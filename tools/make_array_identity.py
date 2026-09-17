#!/usr/bin/env python3
"""Generate `source_arrays.identity.json` from a verified original `source_arrays.npz`.

The original-input baseline is withheld from the public release (117,410,167 B, single file above
the attachment budget). The identity record it is replaced by carries the exact array-level
assertions the archived raw phase made, so a public run still fails closed on any difference.

  python tools/make_array_identity.py \
      --source /path/to/verified/source_arrays.npz \
      --expect-sha256 dab55008db28dbcb3d39273f6dbef9ae3775dedf44c48485a259b8dd29584756 \
      --out core_reproduction/original_inputs/confidence_validity/source_arrays.identity.json

The tool never writes to the source file. `--expect-sha256` is required in practice: generating an
identity record from an unverified baseline would publish an identity of an unverified object.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core_reproduction"))

from array_identity import SCHEMA, identity_record  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "core_reproduction" / "original_inputs" /
                        "confidence_validity" / "source_arrays.identity.json")
    parser.add_argument("--expect-sha256", default=None,
                        help="expected sha256 of the source npz; the record is only written if it matches")
    parser.add_argument("--note", default=None)
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        print(json.dumps({"status": "FAIL", "error": f"source not found: {source}"}, indent=2))
        return 2
    digest = sha256_file(source)
    if args.expect_sha256 and digest != args.expect_sha256:
        print(json.dumps({"status": "FAIL", "error": "source sha256 mismatch; refusing to write an identity "
                                                     "record for an unverified baseline",
                          "source": str(source), "expected": args.expect_sha256, "actual": digest}, indent=2))
        return 1
    note = args.note or (
        "confidence_validity/source_arrays.npz (archived original-input baseline; withheld from the "
        "public release). The identity below was generated from the copy whose sha256 is recorded here."
    )
    record = identity_record(source, npz_sha256=digest, npz_bytes=source.stat().st_size,
                             recorded_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), note=note)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "WROTE", "schema": SCHEMA, "out": str(out), "source": str(source),
                      "source_sha256": digest, "source_bytes": record["source_bytes"],
                      "arrays": len(record["arrays"]), "n_models": record["n_models"],
                      "n_items": record["n_items"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
