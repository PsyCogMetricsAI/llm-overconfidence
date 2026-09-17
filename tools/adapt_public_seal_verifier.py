#!/usr/bin/env python3
"""Bind the seal verifier to the hashed public scientific protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROTOCOL_PLACEHOLDER = "PUBLIC_PROTOCOL_SHA256"
ORIGINAL_VERIFIER_RELPATH = "confidence_comparison/risk_metrics_v1/reviews/verify_seal_independent.py"

BINDING_MARKER = "public-mechanical-rebinding"
CONSTANT_LINE = f'PLAN_SHA = "{PROTOCOL_PLACEHOLDER}"'
CHECK_BLOCK = ('    if digest(ROOT/"protocol.json") and digest(Path(pr["plan_path"])) != PLAN_SHA:\n'
               '        fail("plan", "plan SHA mismatch")\n')


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def adapt_seal_verifier(text: str, *, source_sha256: str, expected_source_sha256: str,
                        public_protocol_sha256: str, public_protocol_relpath: str,
                        plan_binding_sha256: str) -> tuple[str, dict]:
    """Return (adapted_text, record). Raises AssertionError on any precondition failure."""
    assert source_sha256 == expected_source_sha256, (
        "SEAL_VERIFIER_SOURCE_SHA_MISMATCH", source_sha256, expected_source_sha256)
    assert text.count(CONSTANT_LINE) == 1, ("SEAL_VERIFIER_CONSTANT_NOT_UNIQUE", text.count(CONSTANT_LINE))
    assert text.count(CHECK_BLOCK) == 1, ("SEAL_VERIFIER_CHECK_NOT_UNIQUE", text.count(CHECK_BLOCK))
    assert len(public_protocol_sha256)==64 and all(c in "0123456789abcdef" for c in public_protocol_sha256), "INVALID_PROTOCOL_DIGEST"

    constants = f'PLAN_SHA = "{public_protocol_sha256}"'
    adapted = text.replace(CONSTANT_LINE, constants, 1)
    checks = CHECK_BLOCK
    assert 'fail("plan", "plan SHA mismatch")' in adapted, "ARCHIVED_ASSERTION_WAS_REMOVED"
    assert adapted.count('fail("plan", "plan SHA mismatch")') == 1
    record = {"source_sha256": source_sha256,"public_protocol_sha256":public_protocol_sha256,
              "adapted_sha256":sha256_bytes(adapted.encode()),"adapted_bytes":len(adapted.encode()),
              "scope":"Public protocol hash substitution; all scientific seal checks retained"}
    return adapted, record


def load_binding(migration_root: Path):
    integrity=json.loads((migration_root/'PUBLIC_INTEGRITY.json').read_text())
    public=migration_root/'PROTOCOL.md'
    digest=sha256_bytes(public.read_bytes())
    assert digest==integrity['public_files']['PROTOCOL.md'], 'PUBLIC_PROTOCOL_CHANGED'
    return {'public_protocol':'core_reproduction/PROTOCOL.md'},digest,'PROTOCOL.md'


def expected_source_sha(migration_root: Path) -> str:
    return json.loads((migration_root/'INPUT_HASHES.json').read_text())[ORIGINAL_VERIFIER_RELPATH]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--migration-root", type=Path, required=True,
                        help="release core_reproduction directory holding PUBLIC_INTEGRITY.json")
    parser.add_argument("--source", type=Path, default=None,
                        help="archived verifier source; defaults to <migration-root>/original_inputs/<relpath>")
    parser.add_argument("--dst", type=Path, required=True, help="runtime copy to write")
    parser.add_argument("--record", type=Path, default=None, help="optional JSON record path")
    args = parser.parse_args()
    migration_root = args.migration_root.resolve()
    source = (args.source or migration_root / "original_inputs" / ORIGINAL_VERIFIER_RELPATH).resolve()
    assert source.is_file(), ("SEAL_VERIFIER_SOURCE_MISSING", str(source))
    binding, public_sha, public_rel = load_binding(migration_root)
    text = source.read_text()
    adapted, record = adapt_seal_verifier(
        text, source_sha256=sha256_bytes(text.encode()), expected_source_sha256=expected_source_sha(migration_root),
        public_protocol_sha256=public_sha, public_protocol_relpath=public_rel,
        plan_binding_sha256=sha256_bytes((migration_root / "PUBLIC_INTEGRITY.json").read_bytes()))
    record["public_protocol_release_path"] = binding["public_protocol"]
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(adapted)
    assert sha256_bytes(args.dst.read_bytes()) == record["adapted_sha256"]
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"status": "WROTE", "dst": str(args.dst), "record": str(args.record) if args.record else None,
                      "source_sha256": record["source_sha256"], 
                      "public_protocol_sha256": public_sha, "adapted_sha256": record["adapted_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
