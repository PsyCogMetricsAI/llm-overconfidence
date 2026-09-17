#!/usr/bin/env python3
"""Fail-closed check for the inputs this release needs but does not bundle.

Reads docs/required_inputs.json and reports, per entry, whether it exists at the expected
relative path and matches its recorded identity. Nothing is written and nothing is fetched.

Modes
-----
full      (default) every class=required_external entry must be present and match; exit 1 otherwise.
included  verify only the bundled entries in this tree (integrity check); external absences are
          reported as expected-and-not-included without failing.
public    public recomputation mode: verify every bundled entry, including the public replacements
          for the withheld audit-only inputs (array identity record, public scientific protocol and integrity manifest,
          public summary references + feature digests + bundled reviewer scripts, upstream source
          index), plus cross-file consistency between them. class=required_external entries are
          reported as retrievable-or-withheld without failing; class=local_audit_only entries are
          reported as exact-audit-only. Missing or inconsistent public replacements still fail.

Exit codes: 0 = the requested requirements are satisfied, 1 = at least one is missing/mismatched,
2 = the specification file itself is missing or unreadable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "required_inputs.json"
CHUNK = 8 * 1024 * 1024
PUBLIC_INTEGRITY = ROOT / "core_reproduction" / "PUBLIC_INTEGRITY.json"
PUBLIC_PROTOCOL = ROOT / "core_reproduction" / "PROTOCOL.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(entry: dict) -> dict:
    path = ROOT / entry["path"]
    record = {
        "id": entry["id"],
        "class": entry["class"],
        "status": entry["status"],
        "path": entry["path"],
        "required_for": entry.get("required_for", []),
        "expected_bytes": entry.get("bytes"),
        "exists": path.exists(),
    }
    if path.is_file():
        record["actual_bytes"] = path.stat().st_size
        if entry.get("sha256"):
            record["sha256_match"] = sha256_file(path) == entry["sha256"]
    elif path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        record["actual_files"] = len(files)
        record["actual_bytes"] = sum(p.stat().st_size for p in files)
        if entry.get("sha256_map"):
            map_ok = True
            for name, digest in entry["sha256_map"].items():
                candidate = path / name
                if not candidate.is_file() or sha256_file(candidate) != digest:
                    map_ok = False
            record["sha256_map_match"] = map_ok
    return record


def public_consistency_problems(spec: dict) -> tuple[list, dict]:
    """Cross-file consistency of the public recomputation replacements (fail-closed)."""
    problems, detail = [], {}
    by_id = {entry["id"]: entry for entry in spec["entries"]}

    def read(path: Path, label: str):
        if not path.is_file():
            problems.append({"check": label, "problem": "missing file", "path": str(path.relative_to(ROOT))})
            return None
        try:
            if path.suffix == ".gz":
                import gzip
                with gzip.open(path, "rt") as handle:
                    return json.load(handle)
            return json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            problems.append({"check": label, "problem": f"unreadable JSON: {exc}",
                             "path": str(path.relative_to(ROOT))})
            return None

    identity = read(ROOT / by_id["public_array_identity_record"]["path"], "public_array_identity_record")
    baseline = by_id["archived_source_arrays_baseline"]
    if identity is not None:
        detail["identity_arrays"] = len(identity.get("arrays", {}))
        if identity.get("schema") != "llm-overconfidence/array-identity/1":
            problems.append({"check": "public_array_identity_record", "problem": "unexpected schema"})
        if identity.get("source_sha256") != baseline.get("sha256"):
            problems.append({"check": "public_array_identity_record",
                             "problem": "source_sha256 does not match the archived baseline identity",
                             "identity": identity.get("source_sha256"), "baseline": baseline.get("sha256")})
        if identity.get("source_bytes") != baseline.get("bytes"):
            problems.append({"check": "public_array_identity_record", "problem": "source_bytes mismatch"})

    integrity = read(PUBLIC_INTEGRITY, "public_integrity")
    if integrity is not None:
        for section in ['public_files','corrected_files','source_verifier_files']:
            for rel,want in integrity[section].items():
                path=ROOT/'core_reproduction'/rel
                if not path.is_file() or sha256_file(path)!=want:
                    problems.append({'check':'public_integrity','problem':'input digest mismatch','path':rel})
        if sha256_file(ROOT/'core_reproduction/run_core.py')!=integrity['driver_sha256']:
            problems.append({'check':'public_integrity','problem':'driver digest mismatch'})

    for branch in ("initial", "expanded"):
        base = ROOT / "reference_originals/corrected_historical" / branch
        manifest = read(base / "REFERENCE_MANIFEST.json", "corrected_historical_" + branch)
        if manifest is None:
            continue
        if manifest.get("n_models") != 6701 or manifest.get("n_orgs") != 1330 or len(manifest.get("feature_tables", {})) != 50:
            problems.append({"check": "corrected_historical_" + branch, "problem": "unexpected panel or feature-table count"})
        for rel, want in manifest.get("json_files", {}).items():
            candidate = base / rel
            if not candidate.is_file() or sha256_file(candidate) != want:
                problems.append({"check": "corrected_historical_" + branch, "problem": "reference JSON mismatch", "path": rel})
        detail["corrected_historical_" + branch] = {"models": manifest.get("n_models"), "organizations": manifest.get("n_orgs"), "feature_tables": len(manifest.get("feature_tables", {}))}

    index_entry = by_id.get("upstream_source_index")
    if index_entry:
        index = read(ROOT / index_entry["path"], "upstream_source_index")
        if index is not None:
            counts = index.get("counts", {})
            detail["upstream_index_entries"] = counts.get("entries")
            if counts.get("entries") != 13494 or counts.get("arc") != 6746 or counts.get("hellaswag") != 6748:
                problems.append({"check": "upstream_source_index", "problem": "unexpected member counts", "counts": counts})
            if index.get("errors"):
                problems.append({"check": "upstream_source_index", "problem": "index carries unresolved member errors",
                                 "errors": len(index["errors"])})
            if index.get("original_harvester_sha256") is None:
                problems.append({"check": "upstream_source_index", "problem": "missing original harvester identity"})
    corrected = read(ROOT / "core_reproduction/corrected_inputs/CORRECTED_CORE_ARRAYS.identity.json", "corrected_core_arrays")
    feature_ids = read(ROOT / "core_reproduction/corrected_inputs/CORRECTED_FEATURE_IDENTITIES.json", "corrected_core_features")
    joint = read(ROOT / "joint_reproduction/input_contract/CONTRACT.json", "joint_cold_contract")
    joint_artifacts = read(ROOT / "joint_reproduction/ARTIFACTS.json", "joint_artifacts")
    if corrected is not None:
        detail["corrected_source_shape"] = corrected.get("arrays", {}).get("c", {}).get("shape")
        if detail["corrected_source_shape"] != [6706, 1168]:
            problems.append({"check": "corrected_core_arrays", "problem": "unexpected corrected candidate source dimensions"})
    if feature_ids is not None:
        detail["corrected_feature_tables"] = len(feature_ids.get("files", {}))
        if detail["corrected_feature_tables"] != 50:
            problems.append({"check": "corrected_core_features", "problem": "expected 50 corrected source table identities"})
    if joint is not None:
        for key, rel in [("mask_sha256", "upstream_metadata/UPSTREAM_ITEM_IDENTITY_MASK.json"),
                         ("raw_semantic_sha256", "upstream_metadata/RAW_SEMANTIC_IDENTITY.json.gz")]:
            candidate = ROOT / rel
            if not candidate.is_file() or sha256_file(candidate) != joint.get(key):
                problems.append({"check": "joint_cold_contract", "problem": "input identity mismatch", "path": rel})
        for name, want in joint.get("code_hashes", {}).items():
            candidate = ROOT / "joint_reproduction/original_code" / name
            if not candidate.is_file() or sha256_file(candidate) != want:
                problems.append({"check": "joint_cold_contract", "problem": "original module mismatch", "path": name})
    if joint_artifacts is not None:
        for rel, want in joint_artifacts.get("files", {}).items():
            candidate = ROOT / "joint_reproduction" / rel
            if not candidate.is_file() or sha256_file(candidate) != want:
                problems.append({"check": "joint_artifacts", "problem": "artifact mismatch", "path": rel})
    return problems, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["full", "included", "public"], default="full")
    parser.add_argument("--spec", type=Path, default=SPEC)
    args = parser.parse_args()

    if not args.spec.is_file():
        print(json.dumps({"status": "FAIL", "error": f"specification not found: {args.spec}"}, indent=2))
        return 2
    spec = json.loads(args.spec.read_text())
    wanted = {"full": "required_external", "included": "bundled", "public": "bundled"}[args.mode]

    checked, failures, informational = [], [], []
    for entry in spec["entries"]:
        record = describe(entry)
        if entry["class"] == wanted:
            checked.append(record)
            if not record["exists"]:
                failures.append({**record, "problem": "missing"})
            elif record.get("sha256_match") is False:
                failures.append({**record, "problem": "sha256 mismatch"})
            elif record.get("sha256_map_match") is False:
                failures.append({**record, "problem": "sha256 map mismatch"})
            elif entry.get("files") and record.get("actual_files") is not None and record["actual_files"] != entry["files"]:
                failures.append({**record, "problem": f"expected {entry['files']} files"})
        elif entry["class"] == "bundled" and args.mode == "public":
            checked.append(record)
            if not record["exists"]:
                failures.append({**record, "problem": "missing"})
            elif record.get("sha256_match") is False:
                failures.append({**record, "problem": "sha256 mismatch"})
            elif record.get("sha256_map_match") is False:
                failures.append({**record, "problem": "sha256 map mismatch"})
            elif entry.get("files") and record.get("actual_files") is not None and record["actual_files"] != entry["files"]:
                failures.append({**record, "problem": f"expected {entry['files']} files"})
        else:
            informational.append({k: record[k] for k in ("id", "class", "status", "path", "exists")})

    consistency, consistency_detail = ([], {}) if args.mode != "public" else public_consistency_problems(spec)
    failures.extend(consistency)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "mode": args.mode,
        "scope": (
            "bundled-file integrity"
            if args.mode == "included"
            else "public recomputation: bundled replacements for the withheld audit-only inputs plus cross-file consistency"
            if args.mode == "public"
            else "external inputs required to rerun the documented pipeline"
        ),
        "checked": checked,
        "failures": failures,
        "other_entries": informational,
    }
    if args.mode == "public":
        report["public_replacements"] = consistency_detail
        report["public_flow"] = {
            "raw_scan": ("class=required_external, obtainable with tools/reharvest_peritem.py against "
                         "upstream_metadata/UPSTREAM_PIN_INDEX.json.gz (see docs/04_REQUIRED_INPUTS.md)"),
            "array_baseline": "replaced by the bundled exact array-identity record (--baseline identity)",
            "scientific_protocol": "core_reproduction/PROTOCOL.md",
            "frozen_reference_trees": "replaced by reference_originals/corrected_historical per-branch identities and JSON references",
        }
    if failures and args.mode == "full":
        report["remediation"] = (
            "Obtain the required archival inputs listed in docs/04_REQUIRED_INPUTS.md and place them at "
            "the expected relative paths. This release bundles no download URLs and performs no network "
            "access; a missing input is reported rather than substituted."
        )
    if failures and args.mode == "public":
        report["remediation"] = (
            "The public recomputation path is built from bundled replacements for the withheld audit-only "
            "inputs; a missing or inconsistent replacement is a packaging error, not a missing external "
            "input. Rebuild them from a verified source with tools/make_array_identity.py, "
            "and "
            "tools/make_upstream_source_index.py, and see docs/04_REQUIRED_INPUTS.md."
        )
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
