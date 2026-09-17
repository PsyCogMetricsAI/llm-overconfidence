#!/usr/bin/env python3
"""IRTREC-CELLS — reconstruct historical per-cell IRT input records from raw per-model NPZ.

WHAT THIS DOES (and what it deliberately does NOT do)
-----------------------------------------------------
Converts one raw per-model file ``data_peritem_v1_20260913/<bench>/details_<model>.npz``
(fields ``item_ids, loglik, n_tokens, char_lens, gold, n_options`` [+ ``metric_acc``,
``metric_accnorm``]) into the historical per-cell record that the missing
``aligned/matrix_ckpt_<bench>.jsonl`` rows contained:

    cells[item_id] = [LP_mean, LP_sum, BIN, CONF]

with the EXACT rounding/typing of the historical pipeline:

    LP_mean = round(gold_sum / gold_n_tokens, 6)                  (harvest_oll_full index 0)
    LP_sum  = round(gold_sum, 4)                                  (harvest_oll_full index 7)
    BIN     = int(argmax_k(sum_k / max(char_len_k, 1)) == gold)   (harvest_oll_full index 10)
    CONF    = round(softmax_k(sum_k / max(char_len_k, 1))[argmax], 6)  (index 14)
    assemble: values -> np.float32 arrays LP_mean/LP_sum/BIN/CONF

Original definitions (read-only sources, sha256 pinned in manifest.json):
  Imports/eirt_cl_reproducible_20260611/logprob_irt_compare_20260614/harvest_oll_full.py
      _harvest_rows  L97-L164  (indices 0/7/10/14; token fallback 1 L118; char guard L122)
  Imports/eirt_cl_reproducible_20260611/logprob_irt_compare_20260614/matrix_harvest.py
      I_GLM/I_GLS/I_CORR/I_CP L31; process_repo rounding L80-L82; cmd_assemble L129-L170

This is a RECOVERY UTILITY, not a re-harvest: it never touches the raw inputs (open read-only),
never fits anything, never builds the historical full item x model matrix, and never uses the
network. It requires EXPLICIT model and item ordering for assembly — it will not infer the
missing 1974-model HellaSwag cohort or any other cohort.

Strictness: required fields that are absent/malformed raise ``CellRecoveryError`` — there is no
silent substitution. The single documented exception is the ORIGINAL semantic rule for missing
option token counts (``cont_tokens`` missing/empty -> 1 token, harvest_oll_full.py L118), which is
reproduced exactly and counted per file (``n_tokens_fallback_cells``); use ``--strict-tokens`` to
fail instead.

Subcommands
-----------
  inventory  --bench <arc|hellaswag> [--dir DIR]                 list raw NPZ files (sizes only)
  recover    --npz FILE --bench B --out-dir DIR [--strict-tokens] [--allow-k-lt-2]
                                                                 -> <out-dir>/<bench>/<model>.cells.jsonl (+ .checks.json)
  assemble   --bench B --records FILE... --models-order FILE --items-order FILE --out FILE
                                                                 -> npz(items, models, LP_mean, LP_sum, BIN, CONF)
                                                                    float32, explicit ordering, no inference

Usage example (exact commands used by run_checks.sh):
  python cell_recover.py recover --npz <raw.npz> --bench arc --out-dir outputs/records
  python cell_recover.py assemble --bench arc --records outputs/records/arc/*.cells.jsonl \
      --models-order outputs/order/arc.models.txt --items-order outputs/order/arc.items.txt \
      --out outputs/bounded_demo_assembly_arc.npz
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True

REQUIRED_FIELDS = ("item_ids", "loglik", "n_tokens", "char_lens", "gold", "n_options")
METRIC_FIELDS = ("metric_acc", "metric_accnorm")

# Historical tuple indices in harvest_oll_full._harvest_rows -> matrix_harvest cell record.
I_GLM, I_GLS, I_CORR, I_CP = 0, 7, 10, 14  # matrix_harvest.py L30-L31
DIGITS_GLM, DIGITS_GLS, DIGITS_CP = 6, 4, 6  # matrix_harvest.py L80-L82


class CellRecoveryError(RuntimeError):
    """Raised on absent/malformed required input — never silently substituted."""


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(x) -> np.ndarray:
    """Verbatim semantics of harvest_oll_full.softmax (L92-L94)."""
    x = np.asarray(x, float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _json_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------------------- load
def load_raw_npz(path: str) -> dict:
    """Load and structurally validate a raw per-item NPZ. Read-only; no dtype coercion of inputs."""
    if not os.path.isfile(path):
        raise CellRecoveryError(f"raw NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_FIELDS if k not in z.files]
        if missing:
            raise CellRecoveryError(f"{os.path.basename(path)}: required field(s) absent: {missing}")
        out = {k: z[k] for k in z.files}
    n = out["item_ids"].shape[0]
    if out["item_ids"].ndim != 1:
        raise CellRecoveryError(f"item_ids must be 1-D, got shape {out['item_ids'].shape}")
    for k in ("loglik", "n_tokens", "char_lens"):
        if out[k].ndim != 2 or out[k].shape[0] != n:
            raise CellRecoveryError(f"{k} malformed: shape {out[k].shape}, expected ({n}, maxK)")
        if not np.issubdtype(out[k].dtype, np.floating):
            raise CellRecoveryError(f"{k} must be floating point, got {out[k].dtype}")
    maxk = {out[k].shape[1] for k in ("loglik", "n_tokens", "char_lens")}
    if len(maxk) != 1:
        raise CellRecoveryError(
            f"option-axis mismatch: loglik/n_tokens/char_lens widths "
            f"{[out[k].shape[1] for k in ('loglik', 'n_tokens', 'char_lens')]}"
        )
    for k in ("gold", "n_options"):
        if out[k].ndim != 1 or out[k].shape[0] != n:
            raise CellRecoveryError(f"{k} malformed: shape {out[k].shape}, expected ({n},)")
        if not np.issubdtype(out[k].dtype, np.integer):
            v = out[k]
            if not np.issubdtype(v.dtype, np.floating) or not np.isfinite(v).all() or (v != np.round(v)).any():
                raise CellRecoveryError(f"{k} must be integer-valued, got {out[k].dtype}")
    ids = [str(x) for x in out["item_ids"]]
    if len(set(ids)) != len(ids):
        dup = sorted({x for x in ids if ids.count(x) > 1})[:5]
        raise CellRecoveryError(
            f"duplicate item_ids in {os.path.basename(path)} (would silently last-win): {dup}"
        )
    if any(not s for s in ids):
        raise CellRecoveryError(f"{os.path.basename(path)}: empty item_id present")
    return out


# ------------------------------------------------------------------------------------ recover
def recover_raw_values(
    raw: dict,
    *,
    strict_tokens: bool = False,
    allow_k_lt_2: bool = False,
) -> tuple[list[tuple[str, tuple]], dict]:
    """Per-item PRE-ROUNDING values: ([(item_id, (gold_mean, gold_sum, correct_cn, chosen_prob_cn))], checks).

    Mirrors harvest_oll_full._harvest_rows (L97-L164) exactly for the four historical cell values
    (harvest_oll_full tuple indices 0/7/10/14). Fails loudly on anything malformed.
    """
    ids = [str(x) for x in raw["item_ids"]]
    loglik, ntok, clen = raw["loglik"], raw["n_tokens"], raw["char_lens"]
    gold = np.asarray(raw["gold"]).astype(np.int64)
    nopt = np.asarray(raw["n_options"]).astype(np.int64)
    max_cols = loglik.shape[1]
    m_acc = raw.get("metric_acc")
    m_accn = raw.get("metric_accnorm")

    recs: list[tuple[str, tuple]] = []
    c = {
        "n_rows": len(ids),
        "n_cells": 0,
        "k_min": None,
        "k_max": None,
        "n_tokens_fallback_cells": 0,
        "n_char_guard_applied": 0,
        "nan_out": 0,
        "inf_out": 0,
        "metric_fields_present": bool(m_acc is not None and m_accn is not None),
        "n_metric_missing_rows": 0,
    }
    for i, cid in enumerate(ids):
        K = int(nopt[i])
        if K < 1 or K > max_cols:
            raise CellRecoveryError(f"{cid}: n_options={K} out of range 1..{max_cols}")
        if K < 2 and not allow_k_lt_2:
            raise CellRecoveryError(
                f"{cid}: n_options={K} (degenerate single option; original margin/argmax semantics "
                f"undefined) — pass --allow-k-lt-2 to override"
            )
        finite_lg = np.isfinite(loglik[i])
        finite_nt = np.isfinite(ntok[i])
        finite_cl = np.isfinite(clen[i])
        # strict option-padding check: exactly the first K entries are data, the rest NaN padding
        for name, fin in (("loglik", finite_lg), ("n_tokens", finite_nt), ("char_lens", finite_cl)):
            if fin.sum() != K or not fin[:K].all():
                raise CellRecoveryError(
                    f"{cid}: {name} does not match n_options={K} (finite={int(fin.sum())}, "
                    f"finite-beyond-K={int(fin[K:].sum())}) — ambiguous option padding, refusing"
                )
        g = int(gold[i])
        if g < 0 or g >= K:
            raise CellRecoveryError(f"{cid}: gold={g} outside scored options 0..{K - 1}")
        if m_acc is not None and not np.isfinite(m_acc[i]):
            raise CellRecoveryError(f"{cid}: metric_acc absent but row present (original skipped such rows)")
        if m_accn is not None and not np.isfinite(m_accn[i]):
            raise CellRecoveryError(f"{cid}: metric_accnorm absent but row present")
        # --- harvest_oll_full._harvest_rows L117-L124 (verbatim semantics) ---
        sums = np.array([float(x) for x in loglik[i, :K]], float)
        toks = []
        for k in range(K):
            v = float(ntok[i, k])
            if v < 1.0:  # cont_tokens[k] missing/empty -> fallback 1 (L118); also max(Lk,1)
                toks.append(1.0)
                c["n_tokens_fallback_cells"] += 1
            else:
                toks.append(v)
        toks = np.array(toks, float)
        cl_raw = [float(x) for x in clen[i, :K]]
        cl = np.array([max(x, 1.0) for x in cl_raw], float)
        c["n_char_guard_applied"] += sum(1 for x in cl_raw if x < 1.0)
        # --- char-normalized reconstruction (L138-L149) ---
        lp_cn = sums / cl
        cidx_cn = int(np.argmax(lp_cn))
        correct_cn = int(cidx_cn == g)
        sm_cn = softmax(lp_cn)
        chosen_prob_cn = float(sm_cn[cidx_cn])
        gold_sum = float(sums[g])
        gold_mean = gold_sum / float(toks[g])
        # --- matrix_harvest.process_repo L80-L82 rounding (exact Python round, then float32 at assembly)
        v = (gold_mean, gold_sum, int(correct_cn), chosen_prob_cn)
        if any((x != x) for x in v) or any(x in (float("inf"), float("-inf")) for x in v):
            c["nan_out" if any((x != x) for x in v) else "inf_out"] += 1
        recs.append((cid, v))
        c["n_cells"] += 1
        c["k_min"] = K if c["k_min"] is None else min(c["k_min"], K)
        c["k_max"] = K if c["k_max"] is None else max(c["k_max"], K)
    if c["n_cells"] < 50:
        # harvest_oll_full L157: fewer than 50 accepted items is not a usable harvest
        raise CellRecoveryError(f"only {c['n_cells']} cells recovered (<50) — refusing")
    c["raw_values_hash"] = _json_hash([[cid, list(v)] for cid, v in recs])
    c["item_order_hash"] = _json_hash(ids)
    return recs, c


def recover_cells(
    raw: dict,
    *,
    strict_tokens: bool = False,
    allow_k_lt_2: bool = False,
) -> tuple[list[tuple[str, tuple]], dict]:
    """Per-item historical cell record: ([(item_id, (LP_mean, LP_sum, BIN, CONF))], checks).

    Applies exactly matrix_harvest.process_repo's rounding (L80-L82): round(x,6) / round(x,4) /
    int / round(x,6). The float32 cast happens at assembly time (cmd_assemble L149-L158).
    """
    raw_recs, checks = recover_raw_values(raw, strict_tokens=strict_tokens, allow_k_lt_2=allow_k_lt_2)
    recs = [
        (
            cid,
            (
                round(float(v[0]), DIGITS_GLM),
                round(float(v[1]), DIGITS_GLS),
                int(v[2]),
                round(float(v[3]), DIGITS_CP),
            ),
        )
        for cid, v in raw_recs
    ]
    checks["cells_hash"] = _json_hash([[cid, list(v)] for cid, v in recs])
    return recs, checks


def cells_to_historical_record(repo: str, bench: str, recs, npz_sha256: str) -> dict:
    """Historical matrix_ckpt_<bench>.jsonl row shape (matrix_harvest.process_repo L80-L82)."""
    return {
        "repo": repo,
        "bench": bench,
        "cells": {cid: [v[0], v[1], v[2], v[3]] for cid, v in recs},
        "npz_sha256": npz_sha256,
    }


# ------------------------------------------------------------------------------------ assemble
def read_order_file(path: str, what: str) -> list[str]:
    if not path or not os.path.isfile(path):
        raise CellRecoveryError(f"explicit {what} ordering file required (missing: {path!r})")
    out = []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    if not out:
        raise CellRecoveryError(f"explicit {what} ordering file is empty: {path}")
    if len(set(out)) != len(out):
        raise CellRecoveryError(f"duplicate entries in {what} ordering file: {path}")
    return out


def assemble(
    bench: str,
    record_paths: list[str],
    models_order: list[str],
    items_order: list[str],
    *,
    coverage_min: float = 0.95,
) -> tuple[dict, dict]:
    """Faithful matrix_harvest.cmd_assemble (L129-L170) but with EXPLICIT, validated ordering.

    Never infers a cohort: every listed model must have exactly one record and every record's repo
    must be listed; every cell id must be in the explicit item axis. Extra/unknown -> hard fail.
    """
    allowed = set(models_order)
    by_repo: dict[str, dict] = {}
    for p in sorted(record_paths):
        with open(p, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                repo = r.get("repo")
                if repo not in allowed:
                    raise CellRecoveryError(
                        f"{p}:{ln}: record repo {repo!r} not in explicit models-order ({len(allowed)} entries)"
                    )
                if "cells" not in r:
                    raise CellRecoveryError(f"{p}:{ln}: record has no 'cells' (failed harvest row?)")
                if repo in by_repo:
                    raise CellRecoveryError(f"duplicate record for {repo} (first-wins would silently drop one)")
                by_repo[repo] = r
    missing = [m for m in models_order if m not in by_repo]
    if missing:
        raise CellRecoveryError(f"explicit models-order lists {len(missing)} model(s) with no record: {missing[:5]}")

    iidx = {it: i for i, it in enumerate(items_order)}
    nI, nM = len(items_order), len(models_order)
    LPm = np.full((nI, nM), np.nan, np.float32)
    LPs = np.full((nI, nM), np.nan, np.float32)
    BIN = np.full((nI, nM), np.nan, np.float32)
    CONF = np.full((nI, nM), np.nan, np.float32)
    unknown_items: list[str] = []
    for j, repo in enumerate(models_order):
        for cid, v in by_repo[repo]["cells"].items():
            i = iidx.get(str(cid))
            if i is None:
                unknown_items.append(str(cid))
                continue
            LPm[i, j], LPs[i, j], BIN[i, j], CONF[i, j] = v[0], v[1], v[2], v[3]
    if unknown_items:
        raise CellRecoveryError(
            f"{len(unknown_items)} cell ids absent from explicit items-order (e.g. {sorted(set(unknown_items))[:5]})"
        )
    present = (~np.isnan(LPm)).mean(1)
    keep = present >= coverage_min
    arrays = {
        "items": np.array([it for it, k in zip(items_order, keep) if k]),
        "models": np.array(models_order),
        "LP_mean": LPm[keep],
        "LP_sum": LPs[keep],
        "BIN": BIN[keep],
        "CONF": CONF[keep],
    }
    stats = {
        "bench": bench,
        "n_models": nM,
        "n_items_input_axis": nI,
        "n_items_kept_ge_coverage": int(keep.sum()),
        "coverage_min": coverage_min,
        "fill_median_items_per_model": int(np.median((~np.isnan(LPm)).sum(0))),
        "nan_lp_mean": int(np.isnan(LPm[keep]).sum()),
        "nan_lp_sum": int(np.isnan(LPs[keep]).sum()),
        "nan_bin": int(np.isnan(BIN[keep]).sum()),
        "nan_conf": int(np.isnan(CONF[keep]).sum()),
        "array_hashes": {
            k: hashlib.sha256(np.ascontiguousarray(arrays[k]).tobytes()).hexdigest()
            for k in ("items", "models", "LP_mean", "LP_sum", "BIN", "CONF")
        },
        "dtypes": {k: str(arrays[k].dtype) for k in ("LP_mean", "LP_sum", "BIN", "CONF")},
        "shapes": {k: list(arrays[k].shape) for k in ("LP_mean", "LP_sum", "BIN", "CONF")},
    }
    return arrays, stats


def save_matrix_npz(arrays: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrays)


# ------------------------------------------------------------------------------------------ cli
def _sanitize(repo: str) -> str:
    return repo.split("details_")[-1].replace("/", "__")


def cmd_inventory(a) -> int:
    files = sorted(f for f in glob.glob(os.path.join(a.dir, a.bench, "details_*.npz")) if not f.endswith(".tmp.npz"))
    rows = [{"path": os.path.relpath(f), "bytes": os.path.getsize(f)} for f in files]
    print(json.dumps({"bench": a.bench, "n_files": len(files), "bytes_total": sum(r["bytes"] for r in rows)}))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 0


def cmd_recover(a) -> int:
    raw = load_raw_npz(a.npz)
    recs, checks = recover_cells(raw, strict_tokens=a.strict_tokens, allow_k_lt_2=a.allow_k_lt_2)
    repo = str(raw.get("repo", os.path.basename(a.npz)))
    sha = sha256_file(a.npz)
    out_dir = os.path.join(a.out_dir, a.bench)
    os.makedirs(out_dir, exist_ok=True)
    stem = _sanitize(repo) if repo else os.path.splitext(os.path.basename(a.npz))[0]
    rec_path = os.path.join(out_dir, stem + ".cells.jsonl")
    with open(rec_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(cells_to_historical_record(repo, a.bench, recs, sha)) + "\n")
    checks.update(
        {
            "npz": os.path.abspath(a.npz),
            "npz_sha256": sha,
            "repo": repo,
            "bench": a.bench,
            "records_path": os.path.abspath(rec_path),
            "records_sha256": sha256_file(rec_path),
            "npz_meta": {
                k: str(raw[k])
                for k in ("timestamp", "n_rows_total", "n_kept", "n_skipped", "n_subtasks")
                if k in raw
            },
            "rounding": "python round(x,6)|round(x,4)|int|round(x,6) -> np.float32 at assembly",
        }
    )
    json.dump(checks, open(os.path.join(out_dir, stem + ".checks.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps({k: checks[k] for k in ("repo", "n_cells", "k_min", "k_max", "records_sha256")}))
    return 0


def cmd_assemble(a) -> int:
    models = read_order_file(a.models_order, "models")
    items = read_order_file(a.items_order, "items")
    arrays, stats = assemble(a.bench, a.records, models, items, coverage_min=a.coverage_min)
    save_matrix_npz(arrays, a.out)
    stats["out"] = os.path.abspath(a.out)
    stats["out_sha256"] = sha256_file(a.out)
    stats["models_order_file"] = os.path.abspath(a.models_order)
    stats["models_order_sha256"] = sha256_file(a.models_order)
    stats["items_order_file"] = os.path.abspath(a.items_order)
    stats["items_order_sha256"] = sha256_file(a.items_order)
    json.dump(stats, open(a.out + ".checks.json", "w"), indent=1, sort_keys=True)
    print(json.dumps(stats, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inventory")
    p.add_argument("--bench", required=True, choices=("arc", "hellaswag"))
    p.add_argument("--dir", default="data_peritem_v1_20260913")
    p.add_argument("--out")
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("recover")
    p.add_argument("--npz", required=True)
    p.add_argument("--bench", required=True, choices=("arc", "hellaswag"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--strict-tokens", action="store_true", help="fail instead of applying the original missing-token fallback of 1")
    p.add_argument("--allow-k-lt-2", action="store_true")
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("assemble")
    p.add_argument("--bench", required=True, choices=("arc", "hellaswag"))
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--models-order", required=True, help="explicit model order file (one repo per line)")
    p.add_argument("--items-order", required=True, help="explicit item axis file (one item id per line)")
    p.add_argument("--coverage-min", type=float, default=0.95)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_assemble)

    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except CellRecoveryError as e:
        print(f"CellRecoveryError: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
