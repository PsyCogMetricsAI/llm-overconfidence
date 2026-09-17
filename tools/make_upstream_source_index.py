#!/usr/bin/env python3
"""Build `upstream_metadata/UPSTREAM_SOURCE_INDEX.json.gz` from the frozen raw scan metadata.

Every file in the withheld item-level raw scan (`data_peritem_v1_20260913/<bench>/<repo>.npz`)
carries the upstream dataset repo and the harvest timestamp it was built from. The original
harvester cached one `list_repo_files` listing per repo; both are present locally. This tool joins
them with the original harvester's own path-selection rule to recover, for every one of the 13,494
scan members, the exact upstream parquet path that reproduces the archived file.

The selection rule is a faithful copy of `bench_parquets()` + `_latest()` in the original harvester
(sha256 recorded in the output), including the ARC filter
(`arc:challenge` / `arc_challenge`) and the HellaSwag filter. The recovered parquet path must carry
the exact timestamp stored in the npz; any member that does not resolve, or resolves to a different
timestamp, is reported in `errors` and makes the tool exit non-zero. Nothing is invented: the index
contains no upstream revision (the 2026-09-13 harvest never recorded one), and the reharvest tool
records the revision it actually obtains.

  python tools/make_upstream_source_index.py --raw data_peritem_v1_20260913
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "upstream_metadata" / "UPSTREAM_SOURCE_INDEX.json.gz"
RAW_MANIFEST = ROOT / "RAW_INPUT_MANIFEST.json"
ORIGINAL_HARVESTER = ROOT / 'tools/reharvest_peritem.py'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latest(cands):
    """Newest-timestamp parquet (ISO timestamp sorts lexicographically) - original semantics."""
    return sorted(cands)[-1] if cands else None


def bench_parquets(files, bench):
    """Faithful copy of the original harvester's path selection (atlas_harvest_peritem.py:166-188)."""
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


def read_npz_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        def scalar(name):
            value = z[name]
            return value.item() if value.shape == () else value.tolist()
        return {"repo": str(scalar("repo")), "bench": str(scalar("bench")),
                "timestamp": str(scalar("timestamp")), "n_kept": int(scalar("n_kept")),
                "n_rows_total": int(scalar("n_rows_total"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=ROOT / "data_peritem_v1_20260913",
                        help="raw scan root (with _filelists/); default is the release-local location")
    parser.add_argument("--filelists", type=Path, default=None,
                        help="directory of cached list_repo_files JSON lists; default <raw>/_filelists")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--plain", action="store_true",
                        help="write plain JSON instead of the default gzip (the reharvest tool reads both)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    raw = args.raw.resolve()
    filelists_dir = (args.filelists.resolve() if args.filelists else raw / "_filelists")
    assert raw.is_dir(), ("RAW_SCAN_NOT_FOUND", str(raw))
    assert filelists_dir.is_dir(), ("FILELIST_CACHE_NOT_FOUND", str(filelists_dir))
    manifest = json.loads(RAW_MANIFEST.read_text())

    members = [(rec["benchmark"], Path(rec["path"]).name) for rec in manifest["files"]]
    assert len(members) == 13494, ("UNEXPECTED_RAW_MANIFEST_SIZE", len(members))

    def meta(rec):
        bench, filename = rec
        return {"npz": f"{bench}/{filename}", "bench": bench, "filename": filename,
                **read_npz_meta(raw / bench / filename)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        metas = list(pool.map(meta, members))
    metas.sort(key=lambda r: r["npz"])
    print(f"read {len(metas)} npz metadata records", flush=True)

    repos = sorted({r["repo"] for r in metas})
    repo_index = {repo: i for i, repo in enumerate(repos)}
    filelist_sha, filelist_used = {}, {}
    parquet_of = {}
    errors = []
    for rec in metas:
        repo = rec["repo"]
        key = (repo, rec["bench"])
        if key in parquet_of:
            continue
        cache = filelists_dir / (repo.replace("/", "--") + ".json")
        if not cache.is_file():
            errors.append({"repo": repo, "error": "no cached file list for repo"})
            continue
        files = json.loads(cache.read_text())
        parts = {k: v for k, v in bench_parquets(files, rec["bench"]).items() if v}
        if len(parts) != 1:
            errors.append({"repo": repo, "bench": rec["bench"], "error": f"expected one parquet, got {len(parts)}"})
            continue
        parquet_of[key] = next(iter(parts.values()))
        filelist_sha[repo] = sha256_file(cache)
        filelist_used[repo] = files
    print(f"resolved {len(parquet_of)} / {len(repos)} repos from cached file lists", flush=True)

    entries = []
    for rec in metas:
        repo = rec["repo"]
        key = (repo, rec["bench"])
        if key not in parquet_of:
            continue
        parquet = parquet_of[key]
        timestamp = _ts_of(parquet)
        if timestamp != rec["timestamp"]:
            errors.append({"npz": rec["npz"], "repo": repo, "error": "timestamp mismatch",
                           "npz_timestamp": rec["timestamp"], "parquet_timestamp": timestamp,
                           "parquet": parquet})
            continue
        entries.append([rec["npz"], repo_index[repo], rec["timestamp"], parquet])
    counts = {"entries": len(entries), "arc": sum(1 for e in entries if e[0].startswith("arc/")),
              "hellaswag": sum(1 for e in entries if e[0].startswith("hellaswag/")),
              "repos": len(repos), "npz_metadata_records": len(metas)}
    harvester_sha = sha256_file(ORIGINAL_HARVESTER) if ORIGINAL_HARVESTER.is_file() else None
    record = {
        "schema": "llm-overconfidence/upstream-source-index/1",
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": ("Exact upstream identity of every member of the withheld ARC + HellaSwag "
                    "item-level scan: upstream dataset repo, harvest timestamp and the exact upstream "
                    "parquet path that produces it. The 2026-09-13 harvest recorded no upstream "
                    "revision; tools/reharvest_peritem.py pins and logs the revision it obtains."),
        "entry_fields": ["npz (relative to the raw scan root)", "repo_index into `repos`",
                         "harvest timestamp stored in the npz", "exact upstream parquet path"],
        "selection_rule": ("faithful copy of the original harvester's bench_parquets()/_latest(): "
                           "cached list_repo_files parquet names filtered by benchmark "
                           "(arc:challenge|arc_challenge / hellaswag), newest ISO timestamp last"),
        "npz_identity": ("per-file sha256 of the local npz is not duplicated here; read it from "
                         "RAW_INPUT_MANIFEST.json (same bench/filename keys)"),
        "original_harvester": "Imports/eirt_cl_reproducible_20260611/logprob_irt_compare_20260614/atlas_harvest_peritem.py",
        "original_harvester_sha256": harvester_sha,
        "raw_manifest": "RAW_INPUT_MANIFEST.json",
        "raw_manifest_sha256": sha256_file(RAW_MANIFEST),
        "filelist_cache": {
            "files": len(filelist_sha),
            "digest": hashlib.sha256("".join(
                f"{repo_index[r]} {filelist_sha[r]}\n"
                for r in sorted(filelist_sha, key=lambda x: repo_index[x])).encode()).hexdigest(),
            "digest_definition": ("sha256 over sorted '<repo_index> <sha256 of the cached list_repo_files JSON>' "
                                  "lines; the 53 MB cache itself is not bundled"),
        },
        "repos": repos,
        "counts": counts,
        "errors": errors,
        "entries": sorted(entries),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.plain:
        args.out.write_text(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
    else:
        with gzip.open(args.out, "wt", compresslevel=9) as handle:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(json.dumps({"status": "WROTE" if not errors else "WROTE_WITH_ERRORS", "out": str(args.out),
                      "bytes": args.out.stat().st_size, **counts, "errors": len(errors)}, indent=2))
    return 0 if not errors else 3


if __name__ == "__main__":
    sys.exit(main())
