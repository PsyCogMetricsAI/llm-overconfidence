#!/usr/bin/env python3
"""Build `upstream_metadata/UPSTREAM_PIN_INDEX.json.gz` - a metadata-only upstream revision pin index.

The raw-scan identity index (`UPSTREAM_SOURCE_INDEX.json.gz`) records *where* every archived member
came from (repo, harvest timestamp, exact parquet path) but no upstream revision, because the
2026-09-13 harvest recorded none. This tool pins a revision for every repository in that index
**from metadata only** - no parquet is downloaded and no bulk data transfer happens:

* revisions already established by a download or a reharvest are reused read-only
  (`source_records.jsonl` from the full-panel collection, the local sample reharvest manifest,
  and the two supplemented ARC sources whose cached snapshot revisions were used for the delivered
  parquet);
* for every other repository the current `refs/main` commit is read with `HfApi.repo_info`
  (bounded concurrency, exponential backoff, polite minimum interval) and recorded together with
  its acquisition time;
* per repository the exact parquet path from the identity index is checked against the tree listing
  returned by that metadata call (or against the recent `list_repo_files` recheck for the 309
  full-panel failed repos). Availability is recorded as present / unavailable / unknown; nothing is
  invented and nothing is downloaded.

The index is deliberately honest about its limits: a revision obtained *now* is not the revision the
2026-09-13 harvest used, and the recorded availability is a metadata check, not a byte check. The
reharvest tool consumes this index by default and fails closed when a pin is missing or a pinned path
is recorded unavailable.

Resumable: `--state <jsonl>` keeps one row per repository; re-running skips rows already finished.

  python tools/make_upstream_pin_index.py --index upstream_metadata/UPSTREAM_SOURCE_INDEX.json.gz \
      --state /tmp/pin_state.jsonl --out upstream_metadata/UPSTREAM_PIN_INDEX.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "upstream_metadata" / "UPSTREAM_SOURCE_INDEX.json.gz"
DEFAULT_OUT = ROOT / "upstream_metadata" / "UPSTREAM_PIN_INDEX.json.gz"
REVISION_SOURCES = {
    "sr": "full_panel_collection/source_records.jsonl (parquet downloaded and sha256-recorded)",
    "sample": "local public_portability sample reharvest (parquet downloaded and array-verified)",
    "supp": "full_panel_collection supplement cache snapshot (parquet downloaded and parser-verified)",
    "head": "HfApi.repo_info(refs/main) at build time (metadata only, no download)",
}
AVAILABILITY = {
    0: "absent_from_recorded_listing",
    1: "present_in_recorded_listing",
    2: "unknown_metadata_not_checked",
}
AVAILABILITY_SOURCES = {
    0: "repo_info tree listing at the pinned revision",
    1: "harvest-time list_repo_files cache (2026-09-13), the listing the identity index resolved",
    2: "download/reharvest proof (parquet fetched and hashed)",
    3: "full-panel metadata recheck listing (list_repo_files, 2026-09-16)",
    4: "not checked",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_index(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def _ts_of(path: str) -> str:
    mo = re.search(r"(\d{4}-\d\d-\d\dT[\d\-.]+)", path)
    return mo.group(1) if mo else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--state", type=Path, required=True, help="resumable JSONL state file")
    parser.add_argument("--reuse-source-records", type=Path, default=None)
    parser.add_argument("--reuse-samples", type=Path, default=None)
    parser.add_argument("--reuse-metadata-recheck", type=Path, default=None)
    parser.add_argument("--recheck-filelists", type=Path, default=None)
    parser.add_argument("--supplement-checks", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4, help="metadata workers (hard cap 4)")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/hf2"))
    parser.add_argument("--min-interval", type=float, default=0.15, help="seconds between calls per worker")
    parser.add_argument("--availability-mode", choices=["tree", "listing"], default="listing",
                        help="tree: list each repository at the pinned revision (heavier per request); "
                             "listing: reuse the harvest-time list_repo_files cache for path availability "
                             "and fetch revision metadata only")
    parser.add_argument("--filelist-cache", type=Path, default=None,
                        help="harvest-time list_repo_files cache directory (used by --availability-mode listing)")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--call-timeout", type=float, default=45.0, help="per metadata request timeout (seconds)")
    parser.add_argument("--limit", type=int, default=None, help="debug: only the first N repositories")
    parser.add_argument("--json-summary", action="store_true")
    args = parser.parse_args()
    workers = max(1, min(args.workers, 4))
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    index = load_index(args.index.resolve())
    assert index["schema"] == "llm-overconfidence/upstream-source-index/1", index.get("schema")
    repos = list(index["repos"])
    repo_index = {repo: i for i, repo in enumerate(repos)}
    paths = [dict(zip(["npz", "repo_index", "timestamp", "parquet"], e)) for e in index["entries"]]
    paths_by_repo: dict[int, list] = {}
    for rec in paths:
        rec["bench"] = rec["npz"].split("/", 1)[0]
        paths_by_repo.setdefault(rec["repo_index"], []).append(rec)

    reuse: dict[int, dict] = {}
    if args.reuse_source_records and args.reuse_source_records.is_file():
        for line in args.reuse_source_records.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            i = repo_index.get(row["repo"])
            if i is None:
                continue
            rec = reuse.setdefault(i, {"revision": row["revision"], "source": "sr", "obtained_utc": None,
                                       "paths": {}})
            rec["paths"][row["path"]] = 1
    if args.reuse_samples and args.reuse_samples.is_file():
        sample = json.loads(args.reuse_samples.read_text())
        for entry in sample["entries"]:
            i = repo_index.get(entry["repo"])
            if i is None:
                continue
            rec = reuse.setdefault(i, {"revision": entry["revision"], "source": "sample",
                                       "obtained_utc": entry.get("revision_obtained_utc"), "paths": {}})
            rec["paths"][entry["parquet_path"]] = 1
    supplemental = []
    if args.supplement_checks and args.supplement_checks.is_file():
        checks = json.loads(args.supplement_checks.read_text())["checks"]
        for name, rec in sorted(checks.items()):
            repo = rec["repo"]
            parquet = rec["A_original_parser_reproduction"]["parquet"]
            revision = parquet.split("/snapshots/")[-1].split("/")[0]
            path = parquet.split("/snapshots/" + revision + "/", 1)[-1]
            i = repo_index.get(repo)
            assert i is not None, ("SUPPLEMENT_REPO_NOT_IN_INDEX", repo)
            reuse.setdefault(i, {"revision": revision, "source": "supp",
                                 "obtained_utc": rec.get("created_utc"), "paths": {}})
            reuse[i]["paths"][path] = 1
            supplemental.append({"repo": repo, "repo_index": i, "bench": "arc", "timestamp": _ts_of(path),
                                 "parquet": path, "revision": revision,
                                 "npz_sha256": rec["npz_sha256"],
                                 "npz": name,
                                 "note": "recovered ARC source from the full-panel supplement; the "
                                         "cached snapshot revision was used for the delivered parquet"})
    recheck: dict[int, set] = {}
    if args.reuse_metadata_recheck and args.reuse_metadata_recheck.is_file():
        for line in args.reuse_metadata_recheck.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            i = repo_index.get(row["repo"])
            if i is None:
                continue
            files = set()
            if args.recheck_filelists:
                cache_file = args.recheck_filelists / (row["repo"].replace("/", "--") + ".json")
                if cache_file.is_file():
                    files = set(json.loads(cache_file.read_text()))
            recheck[i] = files

    state_path = args.state
    state_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if state_path.is_file():
        for line in state_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done[row["repo_index"]] = row
    todo = [i for i in range(len(repos)) if i not in done and i not in reuse]
    if args.limit:
        todo = todo[:args.limit]
    print(f"repos={len(repos)} paths={len(paths)} reuse={len(reuse)} rechecked={len(recheck)} "
          f"state_done={len(done)} todo={len(todo)} workers={workers}", flush=True)

    from huggingface_hub import HfApi
    api = HfApi()
    listing_cache: dict[str, set] = {}

    def cached_listing(repo: str) -> set:
        if repo not in listing_cache:
            assert args.filelist_cache is not None, "FILELIST_CACHE_NOT_GIVEN"
            entry = args.filelist_cache / (repo.replace("/", "--") + ".json")
            assert entry.is_file(), ("FILELIST_CACHE_ENTRY_MISSING", repo)
            listing_cache[repo] = set(json.loads(entry.read_text()))
        return listing_cache[repo]

    def fetch(i: int) -> dict:
        repo = repos[i]
        last_error = None
        for attempt in range(args.max_attempts):
            try:
                time.sleep(args.min_interval * (1 + random.random()))
                if i in recheck:
                    info = api.repo_info(repo, repo_type="dataset", revision="main", files_metadata=False,
                                         timeout=args.call_timeout)
                    files = recheck[i]
                    listing_source = "metadata_recheck_2026-09-16 (list_repo_files cache)"
                    availability_source = 3
                elif args.availability_mode == "listing":
                    info = api.repo_info(repo, repo_type="dataset", revision="main", files_metadata=False,
                                         timeout=args.call_timeout)
                    files = cached_listing(repo)
                    listing_source = "harvest-time list_repo_files cache (2026-09-13)"
                    availability_source = 1
                else:
                    info = api.repo_info(repo, repo_type="dataset", revision="main", files_metadata=True,
                                         timeout=args.call_timeout)
                    files = {s.rfilename for s in (info.siblings or [])}
                    listing_source = "repo_info(files_metadata) at the pinned revision"
                    availability_source = 0
                path_availability = {rec["parquet"]: (1 if rec["parquet"] in files else 0)
                                     for rec in paths_by_repo.get(i, [])}
                return {"repo": repo, "repo_index": i, "ok": True, "revision": info.sha,
                        "obtained_utc": utcnow(), "listing_source": listing_source,
                        "availability_source": availability_source,
                        "path_availability": path_availability, "n_files": len(files),
                        "last_modified": str(getattr(info, "last_modified", "") or "")}
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                text = str(exc).lower()
                wait = min(2.0 ** attempt * 2.0, 60.0)
                match = re.search(r"retry after (\d+) second", text)
                if match:
                    wait = min(int(match.group(1)) + 3, 90)
                time.sleep(wait + random.uniform(0, min(wait, 5)))
        return {"repo": repos[i], "repo_index": i, "ok": False, "error": last_error,
                "obtained_utc": utcnow()}

    if todo:
        with state_path.open("a") as state, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, i): i for i in todo}
            for n, future in enumerate(as_completed(futures), 1):
                row = future.result()
                state.write(json.dumps(row) + "\n")
                state.flush()
                done[row["repo_index"]] = row
                if n % 250 == 0 or n == len(todo):
                    print(f"  metadata {n}/{len(todo)}", flush=True)

    reuse_files: dict[int, set] = {}
    reuse_errors: dict[int, str] = {}
    for i in sorted(reuse):
        needed = {rec["parquet"] for rec in paths_by_repo.get(i, [])}
        if needed <= set(reuse[i]["paths"]):
            continue
        last_error = None
        for attempt in range(args.max_attempts):
            try:
                time.sleep(args.min_interval * (1 + random.random()))
                reuse_files[i] = set(api.list_repo_files(repos[i], repo_type="dataset",
                                                         revision=reuse[i]["revision"]))
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2.0 ** attempt * 2.0, 60.0))
        if last_error:
            reuse_errors[i] = last_error
    print(f"reuse availability checked for {len(reuse_files)} repos "
          f"({len(reuse_errors)} metadata errors)", flush=True)

    revisions, failures, availability = [], [], {}
    for i in range(len(repos)):
        if i in reuse:
            rec = reuse[i]
            revisions.append([i, rec["revision"], rec["obtained_utc"] or "", "sr" if rec["source"] == "sr"
                              else "sample" if rec["source"] == "sample" else "supp"])
        elif done.get(i, {}).get("ok"):
            row = done[i]
            revisions.append([i, row["revision"], row["obtained_utc"], "head"])
            for path, code in (row.get("path_availability") or {}).items():
                availability[(i, path)] = (code, row.get("availability_source", 0))
        else:
            row = done.get(i, {})
            failures.append([i, row.get("error", "not queried"), row.get("obtained_utc", "")])
    for i in sorted(reuse):
        for path in set(list(reuse[i]["paths"]) + [rec["parquet"] for rec in paths_by_repo.get(i, [])]):
            if path in reuse[i]["paths"]:
                availability[(i, path)] = (1, 2)
            elif i in reuse_files:
                availability[(i, path)] = (1 if path in reuse_files[i] else 0, 0)
            else:
                availability[(i, path)] = (2, 4)
        if i in reuse_errors:
            failures.append([i, f"reuse availability check failed: {reuse_errors[i]}", utcnow()])
    path_rows = []
    for rec in paths:
        code, source = availability.get((rec["repo_index"], rec["parquet"]), (2, 4))
        path_rows.append([rec["repo_index"], rec["bench"], rec["timestamp"], rec["parquet"], code, source])
    supplemental_rows = []
    for rec in supplemental:
        code, source = availability.get((rec["repo_index"], rec["parquet"]), (1, 2))
        supplemental_rows.append([rec["repo_index"], rec["parquet"], rec["npz_sha256"], code, source])
    counts = {
        "repos": len(repos),
        "paths": len(path_rows),
        "arc_paths": sum(1 for r in path_rows if r[1] == "arc"),
        "hellaswag_paths": sum(1 for r in path_rows if r[1] == "hellaswag"),
        "supplemental_arc_paths": len(supplemental_rows),
        "revisions_pinned": len(revisions),
        "revisions_reused": sum(1 for r in revisions if r[3] in ("sr", "sample", "supp")),
        "revisions_queried": sum(1 for r in revisions if r[3] == "head"),
        "revision_failures": len(failures),
        "paths_present": sum(1 for r in path_rows if r[4] == 1),
        "paths_absent_from_recorded_listing": sum(1 for r in path_rows if r[4] == 0),
        "paths_unknown": sum(1 for r in path_rows if r[4] == 2),
        "availability_source_counts": {AVAILABILITY_SOURCES[k]: v
                                       for k, v in sorted(Counter(r[5] for r in path_rows).items())},
    }
    record = {
        "schema": "llm-overconfidence/upstream-pin-index/1",
        "recorded_utc": utcnow(),
        "purpose": ("Metadata-only upstream revision pins for every repository of the archived raw scan: "
                    "the exact parquet path of the identity index plus a revision obtained from metadata "
                    "(or already established by a real download/reharvest). No parquet is downloaded by "
                    "this tool; availability is a tree-listing check, never a byte check."),
        "revision_sources": REVISION_SOURCES,
        "availability_codes": AVAILABILITY,
        "availability_sources": AVAILABILITY_SOURCES,
        "sources": {
            "identity_index": {"path": str(args.index.resolve()), "sha256": sha256_file(args.index.resolve())},
            "reuse_source_records": ({"path": str(args.reuse_source_records), "sha256": sha256_file(args.reuse_source_records)}
                                     if args.reuse_source_records and args.reuse_source_records.is_file() else None),
            "reuse_samples": ({"path": str(args.reuse_samples), "sha256": sha256_file(args.reuse_samples)}
                              if args.reuse_samples and args.reuse_samples.is_file() else None),
            "reuse_metadata_recheck": ({"path": str(args.reuse_metadata_recheck),
                                        "sha256": sha256_file(args.reuse_metadata_recheck)}
                                       if args.reuse_metadata_recheck and args.reuse_metadata_recheck.is_file() else None),
            "supplement_checks": ({"path": str(args.supplement_checks), "sha256": sha256_file(args.supplement_checks)}
                                  if args.supplement_checks and args.supplement_checks.is_file() else None),
            "hf_metadata": {"tool": "huggingface_hub HfApi.repo_info", "workers": workers,
                            "min_interval_seconds": args.min_interval, "max_attempts": args.max_attempts,
                            "availability_mode": args.availability_mode,
                            "filelist_cache": str(args.filelist_cache) if args.filelist_cache else None,
                            "parquet_downloads": 0},
        },
        "counts": counts,
        "repos": repos,
        "revisions": revisions,
        "paths": path_rows,
        "supplemental": supplemental_rows,
        "failures": failures,
        "notes": [
            "A revision obtained now is not the revision the 2026-09-13 harvest used; the archived "
            "harvest recorded no revision. Pins are a new, recorded acquisition identity.",
            "path availability comes from a tree listing - repo_info siblings at the pinned revision, "
            "the harvest-time list_repo_files cache, the 309-repo metadata recheck, or an actual "
            "download - and each row records which source was used; it is never a byte-identity check "
            "of the parquet. Harvest-time or unpinned recheck listings do not certify path availability "
            "at the newly recorded revision; retrieval checks that revision directly.",
            "the index covers the 6,748 repositories of the identity index plus the two supplemented "
            "ARC source paths already in the same repository pool; the 309 full-panel metadata-recheck "
            "repos outside this pool are not part of the core source index.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".gz":
        with gzip.open(args.out, "wt", compresslevel=9) as handle:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
    else:
        args.out.write_text(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
    summary = {"status": "WROTE", "out": str(args.out), "bytes": args.out.stat().st_size, **counts}
    print(json.dumps(summary, indent=2) if args.json_summary else json.dumps(summary))
    return 0 if counts["revision_failures"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
