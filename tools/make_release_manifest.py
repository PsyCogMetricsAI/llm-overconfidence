#!/usr/bin/env python3
"""Build or verify manifest/RELEASE_MANIFEST.json for this release tree.

The manifest is a deterministic, sorted list of (path, bytes, sha256) for every bundled file,
plus totals and the recorded exclusion list. Generated run directories and bytecode caches are
excluded; no network access is used.

  python tools/make_release_manifest.py            # write the manifest
  python tools/make_release_manifest.py --check    # verify the tree against the manifest, exit 1 on drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest" / "RELEASE_MANIFEST.json"
CHUNK = 8 * 1024 * 1024
SKIP_DIRS = {".git", "__pycache__", "runs"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def collect() -> list[dict]:
    records = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel == MANIFEST.relative_to(ROOT):
            continue
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        records.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records


def build() -> dict:
    records = collect()
    return {
        "schema": "llm-overconfidence/release-manifest/1",
        "scope": "every bundled file of the local release candidate",
        "root": ".",
        "excluded": [
            "manifest/RELEASE_MANIFEST.json (this manifest)",
            ".git/, __pycache__/, runs/",
            "*.pyc, *.pyo, *.zip",
        ],
        "totals": {"files": len(records), "bytes": sum(r["bytes"] for r in records)},
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    build_result = build()

    if not args.check:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(build_result, indent=1) + "\n")
        print(json.dumps({"status": "WROTE", "manifest": str(args.manifest), **build_result["totals"]}, indent=2))
        return 0

    if not args.manifest.is_file():
        print(json.dumps({"status": "FAIL", "error": "manifest missing; run without --check first"}, indent=2))
        return 2
    recorded = json.loads(args.manifest.read_text())
    recorded_map = {r["path"]: r for r in recorded["files"]}
    current_map = {r["path"]: r for r in build_result["files"]}
    missing = sorted(set(recorded_map) - set(current_map))
    added = sorted(set(current_map) - set(recorded_map))
    changed = sorted(
        path for path in set(recorded_map) & set(current_map)
        if recorded_map[path] != current_map[path]
    )
    report = {
        "status": "PASS" if not (missing or added or changed) else "FAIL",
        "manifest": str(args.manifest),
        "recorded_totals": recorded["totals"],
        "current_totals": build_result["totals"],
        "missing": missing,
        "added": added,
        "changed": changed,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
