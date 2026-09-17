#!/usr/bin/env python3
"""Scientific cell conversion and matrix assembly used by joint_reproduction/cold.py."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                      # worker processes must be able to import this module
    sys.path.insert(0, HERE)
ARRAY_KEYS = ("models", "items", "LP_mean", "LP_sum", "BIN", "CONF")
FLOAT_KEYS = ("LP_mean", "LP_sum", "BIN", "CONF")
BENCHES = ("arc", "hellaswag")
MIN_VALID_CELLS_PER_MODEL = 50        # original harvest_oll_full >=50 valid rows criterion

# Reviewed small converter (standalone numpy-only module; reviewed exact on 20 real models /
# 107560 model-item pairs, 0 mismatches vs the original `_harvest_rows`). Pin, do not copy.
CELL_RECOVER_REL = "cell_recover.py"
CELL_RECOVER_SHA256_PIN = "cc74c5dfb4e7eb29675b8c71611be04d4469845784da913a34586085cfe250f7"
ROUNDING_CONTRACT = {
    "LP_mean": "round(gold_loglik / max(gold_tokens, 1), 6)",
    "LP_sum": "round(gold_loglik, 4)",
    "BIN": "int(char_norm_argmax == gold)",
    "CONF": "round(softmax(char_norm_loglik)[char_norm_argmax], 6)",
    "cast": "float32 at assembly",
    "char_norm": "loglik_k / max(char_len_k, 1); argmax ties -> lowest index (numpy argmax)",
    "missing_token_rule": "token count < 1 -> 1 (original cont_tokens missing/empty fallback)",
    "missing_cells": "NaN preserved; never imputed; metric_acc/metric_accnorm never substituted",
    "min_valid_items_per_model": 50,
    "policy_item_mask": ("fixed scientific ambiguity policy ids are removed from each model's raw rows "
                         "before conversion and dropped from the output item axis explicitly "
                         "(reason recorded); canonical item order is otherwise unchanged; masked "
                         "cells are never filled or imputed"),
    "nonfinite_loglik_rule": ("a non-finite loglik inside the first n_options columns is a MISSING "
                              "cell (NaN in LP_mean/LP_sum/BIN/CONF); counted + id-hashed in the build "
                              "manifest; never fabricated, never imputed; a model left with <50 valid "
                              "cells stays in the panel as a fully missing (data_free) column"),
}


class BuilderError(RuntimeError):
    """Raised on any manifest/schema/ordering/provenance violation. Never silently repaired."""


_CELL_RECOVER_CACHE: dict[str, tuple[str, object]] = {}


def project_root(start: str) -> str:
    p = os.path.abspath(start)
    for _ in range(12):
        if os.path.isdir(os.path.join(p, "Imports")):
            return p
        nxt = os.path.dirname(p)
        if nxt == p:
            break
        p = nxt
    raise BuilderError(f"could not locate project root (dir containing 'Imports') above {start}")


PROJECT_ROOT = os.path.dirname(HERE)
CELL_RECOVER_PATH = os.path.join(HERE, CELL_RECOVER_REL)


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(buf) -> str:
    return hashlib.sha256(buf).hexdigest()


def load_cell_recover(path: str = CELL_RECOVER_PATH, *, expected_sha256: str | None = None):
    """Import the reviewed small converter by path (no sys.path pollution, no giant harness)."""
    cached = _CELL_RECOVER_CACHE.get(path)
    if cached is not None and (expected_sha256 is None or cached[0] == expected_sha256):
        return cached[1]
    if not os.path.isfile(path):
        raise BuilderError(f"reviewed converter not found: {path}")
    want = expected_sha256 or CELL_RECOVER_SHA256_PIN
    got = sha256_file(path)
    if want and got != want:
        raise BuilderError(
            f"cell_recover sha256 mismatch: got {got}, pinned {want} ({path}). "
            "Re-review the converter, then pass --cell-recover-sha256 explicitly and record why."
        )
    spec = importlib.util.spec_from_file_location("cell_recover_pinned", path)
    if spec is None or spec.loader is None:
        raise BuilderError(f"cannot import converter from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cell_recover_pinned"] = mod   # stable name so exceptions survive pickling
    spec.loader.exec_module(mod)
    if not all(hasattr(mod, a) for a in ("load_raw_npz", "recover_cells", "read_order_file")):
        raise BuilderError(f"{path} lacks expected API (load_raw_npz/recover_cells/read_order_file)")
    _CELL_RECOVER_CACHE[path] = (want, mod)
    return mod


# ------------------------------------------------------------------------------- manifest
def nonfinite_loglik_mask(raw: dict) -> np.ndarray:
    """Rows with a non-finite loglik inside the scored options (first `n_options` columns).

    These cells are MISSING by data rule (see module docstring): the original fit masks NaN cells
    out and the accepted core pipeline classifies such rows as invalid (`incomplete_real_loglik`).
    Returned as a boolean row mask; nothing is modified.
    """
    n = raw["item_ids"].shape[0]
    nopt = np.asarray(raw["n_options"]).astype(np.int64)
    loglik = np.asarray(raw["loglik"], dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        K = int(nopt[i])
        if K < 1 or K > loglik.shape[1]:
            raise BuilderError(f"row {i}: n_options={K} out of range 1..{loglik.shape[1]}")
        mask[i] = not bool(np.isfinite(loglik[i, :K]).all())
    return mask


def _drop_rows(raw: dict, keep: np.ndarray) -> dict:
    """Row-filtered copy of a raw NPZ dict (scalars and 0-d fields are carried unchanged)."""
    n = raw["item_ids"].shape[0]
    out = {}
    for k, v in raw.items():
        arr = np.asarray(v)
        out[k] = arr[keep] if (arr.ndim >= 1 and arr.shape[0] == n) else v
    return out


def _convert_entry(payload) -> dict:
    """Worker: recover one model's cells. Returns per-model sparse (item ids + 4 values).

    The fixed scientific policy mask (if any) is applied HERE, on the raw rows, BEFORE the reviewed
    converter runs: masked rows are removed and their count/id hash recorded.  The converter itself
    and its arithmetic are untouched.
    """
    entry, cr_path, cr_sha, mask_ids = payload
    mask_set = set(mask_ids or ())
    mod = load_cell_recover(cr_path, expected_sha256=cr_sha)
    got = sha256_file(entry["path"])
    if got != entry["source_sha256"]:
        raise BuilderError(
            f"{entry['repo']}: raw source sha256 mismatch (manifest {entry['source_sha256']}, file {got}) - "
            "the fixed scientific coverage was frozen on a different file"
        )
    try:
        raw = mod.load_raw_npz(entry["path"])
    except Exception as e:  # noqa: BLE001
        raise BuilderError(f"{entry['repo']}: {type(e).__name__}: {e}") from e
    def _scalar(key):
        try:
            return str(np.asarray(raw[key]).reshape(-1)[0]) if key in raw else ""
        except Exception:  # noqa: BLE001
            return ""
    repo_in_file, bench_in_file = _scalar("repo"), _scalar("bench")
    if repo_in_file and repo_in_file != entry["repo"]:
        raise BuilderError(f"{entry['path']}: in-file repo {repo_in_file!r} != manifest repo {entry['repo']!r}")
    if bench_in_file and bench_in_file != entry["bench"]:
        raise BuilderError(f"{entry['path']}: in-file bench {bench_in_file!r} != manifest bench {entry['bench']!r}")
    ids_all = [str(x) for x in raw["item_ids"]]
    if len(ids_all) < 50:      # original harvest_oll_full L157 criterion, checked on the raw file
        raise BuilderError(f"{entry['repo']}: only {len(ids_all)} harvested rows (<50) - refusing")
    n_masked_rows = 0
    masked_ids_present: list[str] = []
    masked_rows_hash = ""
    if mask_set:
        is_masked = np.array([cid in mask_set for cid in ids_all], dtype=bool)
        n_masked_rows = int(is_masked.sum())
        masked_ids_present = sorted(set(cid for cid, m in zip(ids_all, is_masked) if m))
        if n_masked_rows:
            masked_rows_hash = sha256_bytes("\n".join(masked_ids_present).encode())
            raw = _drop_rows(raw, ~is_masked)
    missing = nonfinite_loglik_mask(raw)
    n_missing = int(missing.sum())
    n_raw_rows = len(ids_all)
    missing_ids_hash = ""
    missing_policy = ""
    if n_missing:
        missing_ids = [cid for cid, m in zip(ids_all, missing) if m]
        missing_ids_hash = sha256_bytes("\n".join(missing_ids).encode())
        keep = ~missing
        n_valid = int(keep.sum())
        if n_valid < 50:
            # Keep the model in the panel (no silent cohort truncation) as a fully missing column.
            missing_policy = "data_free_model_below_50_valid_cells_after_nonfinite_missingness"
            return {
                "repo": entry["repo"], "path": entry["path"], "source_sha256": got,
                "item_ids": np.array([], dtype=object), "values": np.zeros((0, 4), dtype=np.float64),
                "n_cells": 0, "k_min": 0, "k_max": 0, "n_tokens_fallback_cells": 0,
                "n_char_guard_applied": 0, "raw_values_hash": "", "cells_hash": "",
                "n_raw_rows": n_raw_rows, "n_valid_rows": n_valid, "n_missing_nonfinite": n_missing,
                "missing_ids_hash": missing_ids_hash, "missing_policy": missing_policy,
                "n_masked_rows": n_masked_rows, "masked_ids_present": masked_ids_present,
                "masked_rows_hash": masked_rows_hash,
            }
        raw = _drop_rows(raw, keep)
    try:
        recs, checks = mod.recover_cells(raw)
    except Exception as e:  # noqa: BLE001
        raise BuilderError(f"{entry['repo']}: {type(e).__name__}: {e}") from e
    ids = [cid for cid, _ in recs]
    vals = np.array([v for _, v in recs], dtype=np.float64)
    if vals.shape[1] != 4:
        raise BuilderError(f"{entry['repo']}: unexpected value width {vals.shape}")
    return {
        "repo": entry["repo"], "path": entry["path"], "source_sha256": got,
        "item_ids": np.array(ids, dtype=object), "values": vals,
        "n_cells": int(checks["n_cells"]), "k_min": int(checks["k_min"]), "k_max": int(checks["k_max"]),
        "n_tokens_fallback_cells": int(checks["n_tokens_fallback_cells"]),
        "n_char_guard_applied": int(checks["n_char_guard_applied"]),
        "raw_values_hash": checks["raw_values_hash"], "cells_hash": checks["cells_hash"],
        "n_raw_rows": n_raw_rows, "n_valid_rows": int(checks["n_cells"]),
        "n_missing_nonfinite": n_missing, "missing_ids_hash": missing_ids_hash,
        "missing_policy": missing_policy,
        "n_masked_rows": n_masked_rows, "masked_ids_present": masked_ids_present,
        "masked_rows_hash": masked_rows_hash,
    }


def convert_models(entries: list[dict], *, workers: int, cr_path: str, cr_sha: str,
                   mask_ids: tuple[str, ...] = ()) -> list[dict]:
    payloads = [({"repo": e["repo"], "bench": e["bench"], "path": e["path"],
                  "source_sha256": e["source_sha256"]}, cr_path, cr_sha, mask_ids) for e in entries]
    if workers <= 1:
        return [_convert_entry(p) for p in payloads]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_convert_entry, payloads, chunksize=1))


def assemble(entries: list[dict], results: list[dict], items_order: list[str], *,
             coverage_min: float, min_model_presence: float, bench: str,
             mask_ids: tuple[str, ...] = (),
             min_valid_cells_per_model: int = MIN_VALID_CELLS_PER_MODEL) -> tuple[dict, dict]:
    """Fill float32 (nI x nM) arrays. Explicit ordering only; unknown/missing -> hard fail."""
    by_repo = {r["repo"]: r for r in results}
    if len(by_repo) != len(results):
        raise BuilderError("duplicate conversion results")
    missing = [e["repo"] for e in entries if e["repo"] not in by_repo]
    if missing:
        raise BuilderError(f"no conversion result for {len(missing)} model(s): {missing[:5]}")
    models = [e["repo"] for e in entries]  # entries are pre-sorted by repo
    iidx = {it: i for i, it in enumerate(items_order)}
    nI, nM = len(items_order), len(models)
    mask_set = set(mask_ids or ())
    mask_missing_from_axis = sorted(i for i in mask_set if i not in iidx)
    if mask_missing_from_axis:
        raise BuilderError(
            f"{len(mask_missing_from_axis)} fixed scientific mask id(s) are not on the canonical item "
            f"axis (e.g. {mask_missing_from_axis[:5]}) - the axis and the policy do not match")
    arrays = {k: np.full((nI, nM), np.nan, np.float32) for k in FLOAT_KEYS}
    unknown: list[str] = []
    for j, repo in enumerate(models):
        r = by_repo[repo]
        for cid, v in zip(r["item_ids"], r["values"]):
            i = iidx.get(str(cid))
            if i is None:
                unknown.append(str(cid))
                continue
            arrays["LP_mean"][i, j] = np.float32(v[0])
            arrays["LP_sum"][i, j] = np.float32(v[1])
            arrays["BIN"][i, j] = np.float32(v[2])
            arrays["CONF"][i, j] = np.float32(v[3])
    if unknown:
        raise BuilderError(
            f"{len(unknown)} cell id(s) absent from the supplied canonical item order "
            f"(e.g. {sorted(set(unknown))[:5]}) - ordering is supplied, never inferred"
        )
    present = ~np.isnan(arrays["LP_mean"])          # (nI, nM)
    masked_bool = np.array([it in mask_set for it in items_order], dtype=bool)
    if int(masked_bool.sum()) != len(mask_set):
        raise BuilderError("internal: masked item bookkeeping inconsistent with the axis")
    item_cov = present[~masked_bool].mean(1) if (~masked_bool).any() else np.zeros(0)
    keep = np.zeros(nI, dtype=bool)
    keep[~masked_bool] = item_cov >= coverage_min          # mask ids are ALWAYS dropped (policy)
    if not keep.any():
        raise BuilderError(f"no item reaches >= {coverage_min:.2%} model presence")
    model_presence = present[keep].mean(0)
    bad = [(models[j], float(model_presence[j])) for j in range(nM) if model_presence[j] < min_model_presence]
    if bad:
        raise BuilderError(
            f"{len(bad)} model(s) below min-model-presence {min_model_presence:.2%} on kept items: "
            f"{bad[:5]} - do not truncate the panel silently; fix coverage or lower the threshold explicitly"
        )
    out = {"models": np.array(models), "items": np.array([it for it, k in zip(items_order, keep) if k])}
    for k in FLOAT_KEYS:
        out[k] = arrays[k][keep]
    # Never retain a model with fewer than the original >=50 finite observed cells: a data-free or
    # degenerate latent column must fail the build loudly (root FCV repair requirement).
    finite_per_model = (~np.isnan(out["LP_mean"])).sum(0)
    thin = [(models[j], int(finite_per_model[j])) for j in range(nM)
            if int(finite_per_model[j]) < min_valid_cells_per_model]
    if thin:
        raise BuilderError(
            f"{len(thin)} model(s) have < {min_valid_cells_per_model} finite observed cells in the "
            f"assembled matrix (e.g. {thin[:5]}) - re-derive the analysis panel, do not keep "
            "data-free columns")
    masked_rows_total = int(sum(r.get("n_masked_rows", 0) for r in results))
    masked_ids_seen = sorted(set().union(*[set(r.get("masked_ids_present", ())) for r in results])) \
        if results else []
    stats = {
        "bench": bench,
        "n_models": nM,
        "n_items_input_axis": nI,
        "n_items_kept": int(keep.sum()),
        "item_mask": {
            "n_ids": len(mask_set),
            "ids": sorted(mask_set),
            "n_ids_present_on_axis": int(masked_bool.sum()),
            "ids_absent_from_axis": mask_missing_from_axis,
            "ids_absent_from_every_raw_file": [i for i in sorted(mask_set) if i not in masked_ids_seen],
            "ids_observed_in_raw_files": masked_ids_seen,
            "n_masked_raw_rows_removed": masked_rows_total,
            "row_mask_rule": ROUNDING_CONTRACT["policy_item_mask"],
        },
        "n_items_dropped_masked": int(masked_bool.sum()),
        "items_masked_by_policy": [it for it, m in zip(items_order, masked_bool) if m],
        "coverage_min": float(coverage_min),
        "min_model_presence": float(min_model_presence),
        "min_valid_cells_per_model": int(min_valid_cells_per_model),
        "model_presence_min": float(np.min(model_presence)),
        "model_presence_median": float(np.median(model_presence)),
        "item_coverage_min": float(np.min(item_cov)) if item_cov.size else None,
        "item_coverage_median": float(np.median(item_cov)) if item_cov.size else None,
        "n_items_dropped_low_coverage": int(sum(1 for it, k, m in zip(items_order, keep, masked_bool)
                                                if not k and not m)),
        "dropped_items_low_coverage": [it for it, k, m in zip(items_order, keep, masked_bool)
                                       if not k and not m],
        "n_cells_filled": int(present.sum()),
        "n_cells_total": int(nI * nM),
        "nan_cells_kept": {k: int(np.isnan(out[k]).sum()) for k in FLOAT_KEYS},
        "per_model_cells_min": int(present.sum(0).min()),
        "per_model_cells_max": int(present.sum(0).max()),
        "per_model_finite_cells_min": int(finite_per_model.min()),
        "per_model_finite_cells_median": int(np.median(finite_per_model)),
        "nonfinite_loglik_missing": {
            "rule": ROUNDING_CONTRACT["nonfinite_loglik_rule"],
            "n_cells_missing": int(sum(r.get("n_missing_nonfinite", 0) for r in results)),
            "n_models_with_missing_cells": int(sum(1 for r in results if r.get("n_missing_nonfinite"))),
            "n_models_data_free": int(sum(1 for r in results if r.get("missing_policy"))),
            "data_free_models": sorted(r["repo"] for r in results if r.get("missing_policy")),
        },
        "shapes": {k: list(out[k].shape) for k in FLOAT_KEYS},
        "dtypes": {k: str(out[k].dtype) for k in ("models", "items", *FLOAT_KEYS)},
        "array_sha256": {k: sha256_bytes(np.ascontiguousarray(out[k]).tobytes()) for k in ARRAY_KEYS},
        "model_items": {r["repo"]: {"n_cells": r["n_cells"], "k_min": r["k_min"], "k_max": r["k_max"],
                                    "n_raw_rows": r.get("n_raw_rows"),
                                    "n_missing_nonfinite": r.get("n_missing_nonfinite", 0),
                                    "missing_ids_hash": r.get("missing_ids_hash", ""),
                                    "n_masked_rows": r.get("n_masked_rows", 0),
                                    "masked_rows_hash": r.get("masked_rows_hash", ""),
                                    "data_free": bool(r.get("missing_policy"))}
                        for r in results},
    }
    return out, stats


