#!/usr/bin/env python3
"""Write `upstream_metadata/UPSTREAM_GOLD_SET_PINS.json` from the harvest-time HF cache.

The 2026-09-13 harvest loaded the ARC and HellaSwag gold sets without a `revision=`. The local
Hugging Face cache left over from that harvest still holds the materialised snapshot, whose
`refs/main` commit is also recorded as `sha` in the bundled `upstream_metadata/*.json` records. This
tool reads that cache read-only, cross-checks the two sources against each other, hashes every file
of the snapshot, and writes the pins the public reharvest tool must use. Nothing is invented: if the
cache is absent or disagrees with the bundled record, no pin file is written.

  python tools/make_gold_set_pins.py --hub-cache /tmp/hf2/hub
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "upstream_metadata" / "UPSTREAM_GOLD_SET_PINS.json"
REPOS = {
    "allenai/ai2_arc": {"cache": "datasets--allenai--ai2_arc",
                        "record": "upstream_metadata/allenai__ai2_arc.json"},
    "Rowan/hellaswag": {"cache": "datasets--Rowan--hellaswag",
                        "record": "upstream_metadata/Rowan__hellaswag.json"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hub-cache", type=Path, default=Path("/tmp/hf2/hub"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    hub = args.hub_cache
    assert hub.is_dir(), ("HF_HUB_CACHE_NOT_FOUND", str(hub))
    pins = {}
    for repo, spec in REPOS.items():
        ref = hub / spec["cache"] / "refs" / "main"
        assert ref.is_file(), ("CACHE_REF_MISSING", str(ref))
        revision = ref.read_text().strip()
        snapshot = hub / spec["cache"] / "snapshots" / revision
        assert snapshot.is_dir(), ("CACHE_SNAPSHOT_MISSING", str(snapshot))
        record = json.loads((ROOT / spec["record"]).read_text())
        assert record["sha"] == revision, ("BUNDLED_RECORD_DISAGREES_WITH_CACHE", repo,
                                           record["sha"], revision)
        files = []
        for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
            rel = path.relative_to(snapshot).as_posix()
            files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        pins[repo] = {"revision": revision,
                      "bundled_record": spec["record"],
                      "bundled_record_sha256": sha256_file(ROOT / spec["record"]),
                      "snapshot_files": files}
        print(f"{repo}: {revision} ({len(files)} files)", flush=True)
    out = {"schema": "llm-overconfidence/gold-set-pins/1",
           "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "purpose": ("Gold-set commit pins for tools/reharvest_peritem.py. Recovered from the "
                       "harvest-time Hugging Face cache refs, cross-checked against the bundled "
                       "upstream_metadata records (same commit). The original harvest did not record "
                       "a revision; these pins are what the cache proves was materialised."),
           "pins": pins}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"status": "WROTE", "out": str(args.out),
                      "revisions": {repo: pins[repo]["revision"] for repo in pins}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
