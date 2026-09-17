#!/usr/bin/env python3
"""Public re-harvest of item-level ARC / HellaSwag records with pinned upstream identity.

Use --write-raw to reconstruct verified files without the archived --raw directory.
All arrays and scalar metadata are checked against the public semantic identity index.
--raw retains the original exact-container and array-comparison audit.

The 2026-09-13 harvest recorded the upstream repository and the exact parquet path for every scan
member but no upstream revision. This tool re-fetches selected members from the Hub with

  * the frozen upstream revision pin for the repository, read from
    `upstream_metadata/UPSTREAM_PIN_INDEX.json.gz` (metadata-only pins, each with its acquisition
    time and the tree-listing availability of the exact path). No fresh HEAD is queried by default;
    `--revision` overrides a pin for a deliberate one-off check. A missing pin, a pinned path
    recorded unavailable, or a path without a pin is a hard failure - never a fallback to HEAD - and
  * the exact parquet path recovered by `tools/make_upstream_source_index.py`, plus
  * pinned gold-set commits from `upstream_metadata/UPSTREAM_GOLD_SET_PINS.json`,

re-extracts the arrays with the original harvester's semantics, and verifies the result against the
original local npz member **semantically** (keys, shapes, dtypes, NaN/inf masks and exact values).
NPZ ZIP bytes are never compared: the archive is not byte-stable and a byte-identity claim would be
false. Every attempt is written to `<out>/REHARVEST_MANIFEST.json`; a missing parquet at the pinned
revision, a raw npz whose hash differs from the bundled manifest, or any array difference is an
actionable failure (`verdict: FAIL`, non-zero exit), never a skip.

Run with the harvest environment (`hydro`-style venv with datasets + pyarrow + huggingface_hub);
the HF cache must live on a local filesystem such as /tmp (NFS rejects the pipe names the Hub cache
uses). No bulk download is implied: this tool fetches exactly one parquet per selected member.

  python tools/reharvest_peritem.py --raw data_peritem_v1_20260913 \
      --out runs/reharvest_check --model details_gpt2 --model details_meta-llama__Llama-2-7b-hf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core_reproduction"))
sys.path.insert(0, str(ROOT / "tools"))

from array_identity import array_digest  # noqa: E402

def _default_index() -> Path:
    gz = ROOT / "upstream_metadata" / "UPSTREAM_SOURCE_INDEX.json.gz"
    return gz if gz.is_file() else ROOT / "upstream_metadata" / "UPSTREAM_SOURCE_INDEX.json"


DEFAULT_INDEX = _default_index()
DEFAULT_PINS = ROOT / "upstream_metadata" / "UPSTREAM_GOLD_SET_PINS.json"


def _default_pin_index() -> Path:
    gz = ROOT / "upstream_metadata" / "UPSTREAM_PIN_INDEX.json.gz"
    return gz if gz.is_file() else ROOT / "upstream_metadata" / "UPSTREAM_PIN_INDEX.json"


DEFAULT_PIN_INDEX = _default_pin_index()
RAW_MANIFEST = ROOT / "RAW_INPUT_MANIFEST.json"
MAX_METADATA_WORKERS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path):
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------------
# Original harvester semantics (faithful copies; provenance recorded in the output manifest).
# Source: Imports/eirt_cl_reproducible_20260611/logprob_irt_compare_20260614/atlas_harvest_peritem.py
#         (bench_parquets/_latest/_decode/_ntoks/extract_std/cells_to_npz) and
#         .../harvest_oll_full.py (normalisation + ARC/HellaSwag resolvers + get_acc).
# Only delta: the two gold-set loads take revision=<pinned commit>.
# --------------------------------------------------------------------------------------------------

def _latest(cands):
    return sorted(cands)[-1] if cands else None


def bench_parquets(files, bench):
    pq = [f for f in files if f.endswith(".parquet")]
    if bench == "arc":
        c = [f for f in pq if ("arc:challenge" in f.lower() or "arc_challenge" in f.lower())]
        return {None: _latest(c)} if c else {}
    if bench == "hellaswag":
        c = [f for f in pq if "hellaswag" in f.lower()]
        return {None: _latest(c)} if c else {}
    raise ValueError(f"this release's scan holds only arc and hellaswag members, not {bench!r}")


def _ts_of(path: str) -> str:
    mo = re.search(r"(\d{4}-\d\d-\d\dT[\d\-.]+)", path)
    return mo.group(1) if mo else ""


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def get_acc(r):
    m = r.get("metrics")
    if isinstance(m, dict):
        a = m.get("acc")
        an = m.get("acc_norm", a)
    else:
        a = r.get("acc")
        an = r.get("acc_norm", r.get("acc"))
    if a is None:
        return None, None
    return int(round(float(a))), int(round(float(an if an is not None else a)))


def _decode(rows):
    """Some repos serialize list columns as JSON strings - decode in place (frozen convention)."""
    for r in rows:
        for col in ("predictions", "cont_tokens", "metrics", "choices", "gold"):
            v = r.get(col)
            if isinstance(v, str):
                try:
                    r[col] = json.loads(v)
                except Exception:
                    pass
        p = r.get("predictions")
        if isinstance(p, list) and any(x is None for x in p):
            r["predictions"] = None


def _ntoks(r, K):
    ct = r.get("cont_tokens")
    out = []
    for k in range(K):
        Lk = len(ct[k]) if (ct and k < len(ct) and ct[k]) else 1
        out.append(float(max(Lk, 1)))
    return out


class GoldSets:
    """ARC / HellaSwag resolvers, copied from harvest_oll_full.py with pinned revisions."""

    def __init__(self, pins: dict, pin_metadata=None):
        self.pins = pins
        self.pin_metadata = pin_metadata or json.loads(DEFAULT_PINS.read_text())["pins"]
        self._arc = None
        self._hs = None

    def _load(self, repo, config, split):
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        revision = self.pins[repo]
        path = (f"{config}/{split}-00000-of-00001.parquet" if config else f"data/{split}-00000-of-00001.parquet")
        info = self.pin_metadata[repo]
        assert info["revision"] == revision, "GOLD_REVISION_MISMATCH"
        records = {r["path"]:r for r in info["snapshot_files"]}
        assert path in records, ("GOLD_PARQUET_NOT_PINNED", path)
        local = Path(hf_hub_download(repo_id=repo, filename=path, repo_type="dataset", revision=revision))
        assert sha256_file(local) == records[path]["sha256"], ("GOLD_PARQUET_HASH_MISMATCH", path)
        return pq.read_table(local).to_pylist()

    def arc(self):
        if self._arc is None:
            d = self._load("allenai/ai2_arc", "ARC-Challenge", "test")
            by_q, by_id = {}, {}
            for ex in d:
                g = ex["choices"]["label"].index(ex["answerKey"])
                cid = ex["id"]
                clens = [len(t) for t in ex["choices"]["text"]]
                by_q[norm(ex["question"])] = (g, cid, clens)
                by_id[cid] = (g, cid, clens)

            def res(r):
                ex = r.get("example", "") or ""
                k = norm(re.split(r"Question:", ex)[-1].split("Answer:")[0].strip())
                if k in by_q:
                    return by_q[k]
                if ex.strip() in by_id:
                    return by_id[ex.strip()]
                fp = r.get("full_prompt", "") or ""
                k2 = norm(re.split(r"Question:", fp)[-1].split("Answer:")[0].strip())
                return by_q.get(k2)
            self._arc = res
        return self._arc

    def hellaswag(self):
        if self._hs is None:
            d = self._load("Rowan/hellaswag", None, "validation")

            def pre(t):
                t = t.strip().replace(" [title]", ". ")
                t = re.sub(r"\[.*?\]", "", t)
                return t.replace("  ", " ")
            by_q, by_id = {}, {}
            for ex in d:
                q = norm(pre(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()))
                clens = [len(pre(e)) for e in ex["endings"]]
                by_q[q] = (int(ex["label"]), str(ex["ind"]), clens)
                by_id[str(ex["ind"])] = (int(ex["label"]), str(ex["ind"]), clens)

            def res(r):
                ex = r.get("example", "") or ""
                k = norm(ex.split("\n\n")[-1].strip())
                if k in by_q:
                    return by_q[k]
                return by_id.get(str(ex).strip())
            self._hs = res
        return self._hs


def extract_std(rows, bench, gold: GoldSets):
    """ARC / HellaSwag single-gold extraction (frozen semantic: last row wins per item id)."""
    resfn = gold.arc() if bench == "arc" else gold.hellaswag()
    cells = {}
    n_skip = 0
    for r in rows:
        pl = r.get("predictions")
        if not pl or not isinstance(pl, list):
            n_skip += 1
            continue
        gi = resfn(r)
        if gi is None:
            n_skip += 1
            continue
        g, cid, clens = gi
        if g >= len(pl) or len(clens) != len(pl):
            n_skip += 1
            continue
        a, an = get_acc(r)
        if a is None:
            n_skip += 1
            continue
        K = len(pl)
        cells[cid] = dict(loglik=[float(x) for x in pl], n_tokens=_ntoks(r, K),
                          char_lens=[float(max(c, 1)) for c in clens], gold=int(g), n_opt=K,
                          m_acc=float(a), m_accnorm=(float(an) if an is not None else np.nan))
    return cells, n_skip


def _pad(list_of_lists, maxK):
    a = np.full((len(list_of_lists), maxK), np.nan)
    for i, row in enumerate(list_of_lists):
        a[i, :len(row)] = row
    return a


def cells_to_npz(cells, bench):
    cids = list(cells.keys())
    K = max((c["n_opt"] for c in cells.values()), default=0)
    return dict(
        item_ids=np.array(cids),
        loglik=_pad([cells[c]["loglik"] for c in cids], K),
        n_tokens=_pad([cells[c]["n_tokens"] for c in cids], K),
        char_lens=_pad([cells[c]["char_lens"] for c in cids], K),
        gold=np.array([cells[c]["gold"] for c in cids], np.int64),
        n_options=np.array([cells[c]["n_opt"] for c in cids], np.int64),
        metric_acc=np.array([cells[c]["m_acc"] for c in cids], float),
        metric_accnorm=np.array([cells[c]["m_accnorm"] for c in cids], float),
        is_greedy_available=np.array(0, np.int8),
    )


# --------------------------------------------------------------------------------------------------

def compare_arrays(original, rebuilt: dict) -> tuple[dict, list]:
    records, failures = {}, []
    for key in sorted(set(original.files) | set(rebuilt)):
        if key not in original.files:
            failures.append({"array": key, "problem": "extra array in reharvest"})
            continue
        if key not in rebuilt:
            failures.append({"array": key, "problem": "missing array in reharvest"})
            continue
        a, b = original[key], np.asarray(rebuilt[key])
        rec = {"shape": [int(x) for x in b.shape], "dtype": b.dtype.str,
               "original_digest": array_digest(a), "reharvest_digest": array_digest(b)}
        if a.shape != b.shape:
            failures.append({"array": key, "problem": "shape mismatch", "original": list(a.shape),
                             "reharvest": list(b.shape)})
            rec["match"] = False
            records[key] = rec
            continue
        if a.dtype != b.dtype:
            failures.append({"array": key, "problem": "dtype mismatch", "original": a.dtype.str,
                             "reharvest": b.dtype.str})
            rec["match"] = False
            records[key] = rec
            continue
        if a.dtype.kind in "fc":
            with np.errstate(invalid="ignore"):
                nan_ok = bool(np.array_equal(np.isnan(a), np.isnan(b)))
                inf_ok = bool(np.array_equal(np.isposinf(a), np.isposinf(b))) and \
                    bool(np.array_equal(np.isneginf(a), np.isneginf(b)))
            finite = np.isfinite(a)
            if finite.any():
                with np.errstate(invalid="ignore"):
                    maxdiff = float(np.max(np.abs(a[finite] - b[finite])))
            else:
                maxdiff = 0.0
            equal = nan_ok and inf_ok and bool(np.array_equal(a[finite], b[finite]))
            rec.update(nan_mask_equal=nan_ok, inf_mask_equal=inf_ok, max_abs_difference=maxdiff)
        else:
            equal = bool(np.array_equal(a, b))
            rec["exact"] = equal
        rec["match"] = bool(equal)
        if not equal:
            failures.append({"array": key, "problem": "value mismatch",
                             "original_digest": rec["original_digest"],
                             "reharvest_digest": rec["reharvest_digest"]})
        records[key] = rec
    return records, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--pin-index", type=Path, default=DEFAULT_PIN_INDEX,
                        help="frozen metadata revision pin index (default); used instead of any fresh HEAD query")
    parser.add_argument("--raw", type=Path, default=None,
                        help="raw scan root holding <bench>/<file>.npz (the archived records)")
    parser.add_argument("--out", type=Path, required=True, help="acquisition-log directory")
    parser.add_argument("--write-raw", type=Path, help="reconstruct verified NPZs here without a private --raw baseline")
    parser.add_argument("--semantic-identity", type=Path, default=ROOT / "upstream_metadata/RAW_SEMANTIC_IDENTITY.json.gz")
    parser.add_argument("--include-supplemental", action="store_true", help="include the two additional ARC sources for the joint panel")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/hf2"),
                        help="HF_HOME for this run; must be a local filesystem such as /tmp")
    parser.add_argument("--npz", action="append", default=[], help="exact index entry, e.g. arc/details_gpt2.npz")
    parser.add_argument("--model", action="append", default=[], help="substring of the npz file name")
    parser.add_argument("--bench", choices=["arc", "hellaswag"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--revision", default=None, help="force this revision for every selected repo")
    parser.add_argument("--workers", type=int, default=MAX_METADATA_WORKERS)
    args = parser.parse_args()
    workers = max(1, min(args.workers, MAX_METADATA_WORKERS))
    raw = args.raw.resolve() if args.raw else None
    write_raw = args.write_raw.resolve() if args.write_raw else None
    assert raw is not None or write_raw is not None, "provide --raw for byte audit or --write-raw for reconstruction"
    from raw_semantic_identity import load as load_semantic, verify as verify_semantic
    semantic = load_semantic(args.semantic_identity) if write_raw or args.include_supplemental else None
    out = args.out.resolve()
    assert raw is None or raw.is_dir(), ("RAW_SCAN_NOT_FOUND", str(raw))
    assert not str(out).startswith(str(ROOT)), ("OUT_INSIDE_RELEASE_TREE", str(out))
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    index = load_json(args.index.resolve())
    assert index["schema"] == "llm-overconfidence/upstream-source-index/1", index.get("schema")
    entries = [dict(zip(["npz", "repo_index", "timestamp", "parquet"], e)) for e in index["entries"]]
    for e in entries:
        e["repo"] = index["repos"][e["repo_index"]]
        e["bench"] = e["npz"].split("/", 1)[0]
    raw_manifest = json.loads(RAW_MANIFEST.read_text())
    raw_sha = {str(Path(r["path"]).relative_to("data_peritem_v1_20260913")): (r["sha256"], r["benchmark"])
               for r in raw_manifest["files"]}
    if args.include_supplemental:
        supplemental_pins = load_json(args.pin_index.resolve())
        for row in supplemental_pins.get("supplemental", []):
            repo = supplemental_pins["repos"][row[0]]
            ts = re.search(r"(\d{4}-\d\d-\d\dT[\d\-.]+)", row[1]).group(1)  # original supplemental collector timestamp convention
            rel = "arc/" + repo.split("/", 1)[1] + ".npz"
            assert rel not in raw_sha, ("SUPPLEMENT_DUPLICATES_CORE", rel)
            entries.append({"npz": rel, "repo": repo, "bench": "arc", "timestamp": ts, "parquet": row[1]})
            raw_sha[rel] = (semantic["entries"][rel]["source_sha256"], "arc")
    assert args.npz or args.model or args.bench, "select members with --npz/--model/--bench (no bulk default)"
    selected = []
    for e in entries:
        if args.bench and e["bench"] != args.bench:
            continue
        if args.npz and e["npz"] not in args.npz:
            continue
        if args.model and not any(m in e["npz"] for m in args.model):
            continue
        selected.append(e)
    if args.limit is not None:
        selected = selected[:args.limit]
    assert selected, "selection matched no index entry"
    pins = json.loads(args.pins.read_text())["pins"]
    gold_revisions = {repo: rec["revision"] for repo, rec in pins.items()}

    pin_index = None
    rev_by_repo, avail_by_path, avail_source_by_path = {}, {}, {}
    if not args.revision:
        assert args.pin_index.is_file(), ("PIN_INDEX_MISSING", str(args.pin_index))
        pin_index = load_json(args.pin_index.resolve())
        assert pin_index["schema"] == "llm-overconfidence/upstream-pin-index/1", pin_index.get("schema")
        pin_repos = pin_index["repos"]
        for row in pin_index["revisions"]:
            rev_by_repo[pin_repos[row[0]]] = {"revision": row[1], "obtained_utc": row[2],
                                              "source_code": row[3],
                                              "source": pin_index["revision_sources"][row[3]]}
        for row in pin_index["paths"]:
            avail_by_path[(pin_repos[row[0]], row[3])] = row[4]
            if len(row) > 5:
                avail_source_by_path[(pin_repos[row[0]], row[3])] = pin_index["availability_sources"][str(row[5])]
        for row in pin_index.get("supplemental", []):
            avail_by_path[(pin_repos[row[0]], row[1])] = row[3]
            if len(row) > 4:
                avail_source_by_path[(pin_repos[row[0]], row[1])] = pin_index["availability_sources"][str(row[4])]
        print(f"pin index {args.pin_index.name}: {len(rev_by_repo)} revisions, "
              f"{len(avail_by_path)} paths, {len(pin_index.get('failures', []))} build failures", flush=True)

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from importlib.metadata import version as pkg_version

    api = HfApi()
    repos = sorted({e["repo"] for e in selected})

    def repo_metadata(repo):
        requested = args.revision or "main"
        try:
            info = api.repo_info(repo, repo_type="dataset", revision=requested, files_metadata=True)
        except Exception as exc:  # recorded per entry below; never a silent skip
            return repo, {"revision": args.revision, "obtained_utc": utcnow(),
                          "source": "command line (--revision)" if args.revision else "HfApi.repo_info(main)",
                          "error": f"{type(exc).__name__}: {exc}", "files": {}}
        return repo, {"revision": args.revision or info.sha,
                      "obtained_utc": utcnow(),
                      "source": "command line (--revision), resolved against --revision" if args.revision
                                else "HfApi.repo_info(main)",
                      "last_modified": str(getattr(info, "last_modified", "") or ""),
                      "files": {s.rfilename: getattr(s, "size", None) for s in (info.siblings or [])}}

    if args.revision:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            meta = dict(pool.map(repo_metadata, repos))
        print(f"forced revision {args.revision}: resolved metadata for {len(meta)} repos "
              f"with <= {workers} concurrent calls", flush=True)
    else:
        meta = {repo: {"revision": None, "obtained_utc": None, "source": None, "files": {}, "error": None}
                for repo in repos}
        print("frozen pin index mode: no upstream HEAD metadata is queried", flush=True)

    gold = GoldSets(gold_revisions, pins)
    log = {"schema": "llm-overconfidence/reharvest-log/1", "recorded_utc": utcnow(),
           "tool": "tools/reharvest_peritem.py", "tool_sha256": sha256_file(Path(__file__)),
           "index": str(args.index.resolve()), "index_sha256": sha256_file(args.index.resolve()),
           "raw_scan_root": str(raw) if raw else None,
           "reconstructed_raw_root": str(write_raw) if write_raw else None,
           "semantic_identity_sha256": sha256_file(args.semantic_identity) if semantic else None, "raw_manifest_sha256": sha256_file(RAW_MANIFEST),
           "gold_set_pins": {"path": str(args.pins.resolve()), "sha256": sha256_file(args.pins.resolve()),
                             "revisions": gold_revisions},
           "cache_dir": str(cache), "metadata_workers": workers,
           "pin_index": ({"path": str(args.pin_index.resolve()), "sha256": sha256_file(args.pin_index.resolve()),
                          "recorded_utc": pin_index.get("recorded_utc")} if pin_index else None),
           "acquisition": {"mode": ("forced_revision" if args.revision else "frozen_pin_index"),
                           "revision_source": ("command line --revision" if args.revision else
                                               "upstream_metadata/UPSTREAM_PIN_INDEX.json.gz"),
                           "scope": "explicitly selected members only; this is not a full-panel reharvest",
                           "all_members_reharvested": False,
                           "all_bytes_reharvested_claimed": False},
           "python": sys.version, "packages": {"numpy": np.__version__,
                                               "pyarrow": pkg_version("pyarrow"),
                                               "datasets": pkg_version("datasets"),
                                               "huggingface_hub": pkg_version("huggingface-hub")},
           "semantic_comparison": ("per-array key/shape/dtype/NaN/inf-mask/exact-value equality against the "
                                   "original local npz member; npz ZIP bytes are never compared"),
           "exact_original_zip_bytes_claimed": False,
           "entries": [], "failures": []}
    for entry in selected:
        if args.revision:
            revision = meta[entry["repo"]].get("revision")
            revision_obtained = meta[entry["repo"]]["obtained_utc"]
            revision_source = meta[entry["repo"]]["source"]
            path_availability_code = None
            path_availability = "verified against repo_info at the forced revision"
            path_availability_source = "repo_info tree listing at the forced revision"
        else:
            pin = rev_by_repo.get(entry["repo"])
            revision = pin["revision"] if pin else None
            revision_obtained = pin["obtained_utc"] if pin else None
            revision_source = pin["source"] if pin else None
            path_availability_code = avail_by_path.get((entry["repo"], entry["parquet"]))
            path_availability = {0: "unavailable_at_recorded_listing", 1: "present_at_recorded_listing",
                                 2: "unknown_not_checked", None: "no_pin_for_path"}[path_availability_code]
            path_availability_source = avail_source_by_path.get((entry["repo"], entry["parquet"]))
        rec = {"npz": entry["npz"], "bench": entry["bench"], "repo": entry["repo"],
               "npz_timestamp": entry["timestamp"], "parquet_path": entry["parquet"],
               "revision": revision, "revision_obtained_utc": revision_obtained,
               "revision_source": revision_source, "path_availability": path_availability,
               "path_availability_code": path_availability_code,
               "path_availability_source": path_availability_source}
        try:
            if args.revision:
                assert not meta[entry["repo"]].get("error"), ("REVISION_METADATA_FAILED", meta[entry["repo"]]["error"])
            else:
                assert revision, ("NO_PIN_FOR_REPO", entry["repo"])
                assert path_availability_code is not None, ("NO_PIN_FOR_PATH", entry["npz"], entry["parquet"])
                assert path_availability_code == 1, ("PINNED_PATH_RECORDED_UNAVAILABLE", entry["npz"],
                                                     entry["parquet"], path_availability)
            expected_sha, bench = raw_sha[entry["npz"]]
            assert bench == entry["bench"], ("RAW_MANIFEST_BENCH_MISMATCH", entry["npz"])
            local_npz = raw / entry["npz"] if raw else None
            if local_npz is not None:
                assert local_npz.is_file(), ("RAW_NPZ_MISSING", str(local_npz))
                actual_sha = sha256_file(local_npz)
                assert actual_sha == expected_sha, ("RAW_NPZ_IDENTITY_MISMATCH", entry["npz"], actual_sha, expected_sha)
                rec.update(npz_sha256=actual_sha)
            else:
                assert semantic["entries"][entry["npz"]]["source_sha256"] == expected_sha, "SEMANTIC_SOURCE_MANIFEST_MISMATCH"
                rec.update(archived_source_sha256=expected_sha, input_check="public per-array semantic identity")
            siblings = meta[entry["repo"]]["files"]
            if args.revision:
                assert entry["parquet"] in siblings, ("PARQUET_PATH_NOT_PRESENT_AT_REVISION", entry["parquet"],
                                                      rec["revision"])
            local = hf_hub_download(repo_id=entry["repo"], filename=entry["parquet"], repo_type="dataset",
                                    revision=rec["revision"])
            local = Path(local)
            rec.update(parquet_sha256=sha256_file(local), parquet_bytes=local.stat().st_size,
                       parquet_bytes_downloaded=local.stat().st_size,
                       parquet_sha256_matches_revision_size=(siblings.get(entry["parquet"]) in (None, local.stat().st_size)
                                                             if args.revision else None))
            rows = pq.read_table(local).to_pylist()
            _decode(rows)
            cells, n_skip = extract_std(rows, entry["bench"], gold)
            assert len(cells) >= 50, ("TOO_FEW_EXTRACTED_ITEMS", len(cells))
            rebuilt = cells_to_npz(cells, entry["bench"])
            rebuilt.update(repo=np.array(entry["repo"]), bench=np.array(entry["bench"]),
                           timestamp=np.array(entry["timestamp"]), n_rows_total=np.array(len(rows), np.int64),
                           n_kept=np.array(len(cells), np.int64), n_skipped=np.array(n_skip, np.int64),
                           n_subtasks=np.array(1, np.int64))
            if semantic and semantic["entries"][entry["npz"]]["scope"] == "supplemental_joint":
                rebuilt.update(provenance=np.array("FULLIRT-COLLECT.attempt-1 supplemental; pinned revision; original parser"),
                               source_file=np.array(entry["parquet"]), source_revision=np.array(rec["revision"]),
                               source_sha256=np.array(rec["parquet_sha256"]))
            if local_npz is not None:
                with np.load(local_npz, allow_pickle=False) as original:
                    records, failures = compare_arrays(original, rebuilt)
            else:
                records = verify_semantic(rebuilt, semantic["entries"][entry["npz"]])
                failures = []
            rec.update(n_rows_total=len(rows), n_kept=len(cells), n_skipped=n_skip,
                       arrays=records, array_failures=failures)
            if failures:
                rec["verdict"] = "FAIL"
            elif write_raw is not None:
                expected = semantic["entries"][entry["npz"]]
                verify_semantic(rebuilt, expected)
                target = write_raw / entry["npz"]
                assert not target.exists(), ("OUTPUT_ALREADY_EXISTS", str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                from make_corrected_raw_view import deterministic_npz_bytes
                target.write_bytes(deterministic_npz_bytes({k:rebuilt[k] for k in expected["keys"]}))
                rec.update(reconstructed_npz=str(target), reconstructed_npz_sha256=sha256_file(target),
                           exact_zip_identity_claimed=False)
        except Exception as exc:  # actionable failure, never a skip
            rec["verdict"] = "FAIL"
            rec["error"] = f"{type(exc).__name__}: {exc}"
        if rec.get("verdict") != "FAIL":
            rec["verdict"] = "PASS"
        log["entries"].append(rec)
        print(f"{rec['verdict']} {entry['npz']} rev={(rec.get('revision') or 'unresolved')[:12]} "
              f"parquet={entry['parquet']}", flush=True)
    log["failures"] = [{"npz": r["npz"], "error": r.get("error"), "array_failures": r.get("array_failures")}
                       for r in log["entries"] if r["verdict"] != "PASS"]
    log["n_entries"] = len(log["entries"])
    log["n_pass"] = sum(1 for r in log["entries"] if r["verdict"] == "PASS")
    log["verdict"] = "PASS" if not log["failures"] else "FAIL"
    log["acquisition"].update(
        n_selected=log["n_entries"], n_semantic_pass=log["n_pass"],
        n_parquet_verified=sum(1 for r in log["entries"] if r.get("parquet_bytes_downloaded")),
        parquet_bytes_verified_total=sum(int(r.get("parquet_bytes_downloaded") or 0) for r in log["entries"]),
        transfer_measurement=("not measured: the Hub cache may serve a previously downloaded file, so these "
                              "byte totals are the verified parquet sizes, not a network-transfer claim"),
        note=("each selected member is fetched at its frozen pin; nothing else is downloaded, and this "
              "log makes no claim that the full 13,494-member scan was re-harvested"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "REHARVEST_MANIFEST.json").write_text(json.dumps(log, indent=1) + "\n")
    print(json.dumps({"verdict": log["verdict"], "n_entries": log["n_entries"], "n_pass": log["n_pass"],
                      "manifest": str(out / "REHARVEST_MANIFEST.json")}, indent=2))
    return 0 if log["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
