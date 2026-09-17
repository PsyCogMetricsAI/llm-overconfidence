#!/usr/bin/env python3
"""Build input: build the corrected raw view with the fixed scientific item-identity mask.

The 2026-09-16 upstream-identity audit found 26 item IDs (4 ARC, 22 HellaSwag) whose normalized
prompt collides across the upstream question/answer mapping. The fixed scientific outcome-independent
policy excludes **all** listed IDs wherever they occur. This tool applies exactly that mask to a
local copy of the archived 13,494-file raw scan:

* default bytes mode verifies `RAW_INPUT_MANIFEST.json`; explicit semantic mode checks every
  array against `RAW_SEMANTIC_IDENTITY.json.gz` before applying the same mask;
* rows whose `item_id` is in the mask are removed from the per-row arrays; scalar metadata is kept
  unchanged except `n_kept`, which is rewritten to the truthful post-correction count;
* the four provenance scalars added by the corrected-view builder
  (`correction_policy_id`, `correction_source_sha256`, `correction_n_removed`,
  `correction_removed_ids`) are reproduced exactly;
* files are written with a byte-deterministic npz writer (fixed ZIP timestamps), so the reader can
  check the result against the recorded per-file aggregate digest.

`--verify-against <corrected_raw_dir>` compares every produced file byte-for-byte (sha256 and size)
with a corrected view already present on disk and never writes raw output files. This is the reader
path used by the final corrected analysis. It constructs inputs; it does not estimate results or
modify the paper.

  python tools/make_corrected_raw_view.py --raw data_peritem_v1_20260913 --out runs/corrected_raw
  python tools/make_corrected_raw_view.py --raw data_peritem_v1_20260913 \
      --verify-against /path/toscientific runtime-accepted/corrected_raw
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASK = ROOT / "upstream_metadata" / "UPSTREAM_ITEM_IDENTITY_MASK.json"
DEFAULT_RAW_MANIFEST = ROOT / "RAW_INPUT_MANIFEST.json"
POLICY_ID = "FULLIRT-AMBIGUITY-POLICY-20260916"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_npz_bytes(arrays: dict) -> bytes:
    """Byte-deterministic .npz writer identical to the corrected-view builder."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for key, value in arrays.items():
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            with zf.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.asanyarray(value))
    return buf.getvalue()


def correct_file(raw_bytes: bytes, source_sha256: str, mask_ids: set) -> tuple[bytes, dict]:
    """Return (deterministic npz bytes, record) for one archived member."""
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as archive:
        keys = list(archive.files)
        arrays = {key: archive[key] for key in keys}
    assert "item_ids" in arrays, "MISSING_ITEM_IDS"
    item_ids = np.asarray(arrays["item_ids"]).astype(str)
    n_before = int(item_ids.shape[0])
    row_keys = [k for k in keys if arrays[k].ndim >= 1 and arrays[k].shape[0] == n_before]
    assert row_keys and row_keys[0] == "item_ids", "ITEM_IDS_NOT_FIRST_ROW_ARRAY"
    removed_idx = [i for i, value in enumerate(item_ids.tolist()) if value in mask_ids]
    keep_idx = np.array([i for i in range(n_before) if i not in set(removed_idx)], dtype=np.int64)
    corrected = {key: (np.asanyarray(arrays[key])[keep_idx] if key in row_keys else np.asanyarray(arrays[key]))
                 for key in keys}
    if "n_kept" in arrays:
        n_kept = int(np.asanyarray(arrays["n_kept"]).item())
        corrected["n_kept"] = np.array(n_kept - len(removed_idx),
                                       dtype=np.asanyarray(arrays["n_kept"]).dtype)
    removed_ids = [item_ids.tolist()[i] for i in removed_idx]
    corrected["correction_policy_id"] = np.array(POLICY_ID)
    corrected["correction_source_sha256"] = np.array(source_sha256)
    corrected["correction_n_removed"] = np.array(len(removed_idx), dtype=np.int64)
    corrected["correction_removed_ids"] = np.array("\n".join(removed_ids))
    payload = deterministic_npz_bytes(corrected)
    record = {"n_rows_before": n_before, "n_rows_after": int(keep_idx.size),
              "removed_ids": removed_ids, "output_sha256": sha256_bytes(payload),
              "output_bytes": len(payload)}
    return payload, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="archived raw scan root")
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    parser.add_argument("--input-identity", choices=["bytes", "semantic"], default="bytes", help="bytes preserves archived exact-byte audit; semantic verifies every array of newly acquired NPZs")
    parser.add_argument("--semantic-identity", type=Path, default=ROOT / "upstream_metadata/RAW_SEMANTIC_IDENTITY.json.gz")
    parser.add_argument("--include-supplemental", action="store_true", help="include 2 additional ARC files for the joint panel")
    parser.add_argument("--out", type=Path, default=None, help="output root for the corrected view")
    parser.add_argument("--verify-against", type=Path, default=None,
                        help="compare byte-for-byte against an existing corrected view (no writes)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--npz", action="append", default=[], help="restrict to these relative npz paths")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    raw = args.raw.resolve()
    assert raw.is_dir(), ("RAW_SCAN_NOT_FOUND", str(raw))
    assert args.mask.is_file(), ("ITEM_IDENTITY_MASK_MISSING", str(args.mask))
    mask_record = json.loads(args.mask.read_text())
    assert mask_record["schema"] == "llm-overconfidence/item-identity-mask/1", mask_record.get("schema")
    assert mask_record["policy_id"] == POLICY_ID
    masked = {bench: set(ids) for bench, ids in mask_record["mask_ids"].items()}
    manifest = json.loads(args.raw_manifest.read_text())
    from raw_semantic_identity import load as load_semantic, verify as verify_semantic
    semantic = load_semantic(args.semantic_identity) if args.input_identity == "semantic" or args.include_supplemental else None
    members = []
    for rec in manifest["files"]:
        rel = str(Path(rec["path"]).relative_to("data_peritem_v1_20260913"))
        if args.npz and rel not in args.npz:
            continue
        members.append({"rel": rel, "bench": rec["benchmark"], "sha256": rec["sha256"]})
    if args.include_supplemental:
        for rel, rec in semantic["entries"].items():
            if rec["scope"] != "supplemental_joint" or (args.npz and rel not in args.npz):
                continue
            members.append({"rel": rel, "bench": "arc", "sha256": rec["source_sha256"]})
    if args.limit is not None:
        members = members[:args.limit]
    assert members, "selection matched no manifest member"
    out = args.out.resolve() if args.out else None
    verify = args.verify_against.resolve() if args.verify_against else None
    if out is None and verify is None:
        raise SystemExit("either --out or --verify-against is required")
    print(f"{len(members)} members; out={out} verify={verify}", flush=True)

    results, start = [], time.monotonic()

    def work(rec):
        source = raw / rec["rel"]
        if not source.is_file():
            return {**rec, "ok": False, "error": "source file missing"}
        raw_bytes = source.read_bytes()
        actual = sha256_bytes(raw_bytes)
        if args.input_identity == "bytes" and actual != rec["sha256"]:
            return {**rec, "ok": False, "error": f"source sha256 mismatch: {actual}"}
        try:
            if args.input_identity == "semantic":
                expected = semantic["entries"][rec["rel"]]
                assert expected["source_sha256"] == rec["sha256"], "SEMANTIC_MANIFEST_SOURCE_MISMATCH"
                with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as z:
                    verify_semantic({k:z[k] for k in z.files}, expected)
                    # Restore archived key ordering, preserving exact corrected container output.
                    raw_bytes = deterministic_npz_bytes({k:z[k] for k in expected["keys"]})
            payload, info = correct_file(raw_bytes, rec["sha256"], masked[rec["bench"]])
        except Exception as exc:  # noqa: BLE001
            return {**rec, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result = {**rec, **info, "ok": True, "input_container_sha256": actual, "input_identity_mode": args.input_identity}
        if out:
            target = out / rec["rel"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                return {**rec, "ok": False, "error": "output already exists; select a new output directory"}
            target.write_bytes(payload)
            result["written_sha256"] = sha256_file(target)
        if verify:
            reference = verify / rec["rel"]
            if not reference.is_file():
                result.update(ok=False, error="reference file missing")
            else:
                reference_bytes = reference.read_bytes()
                result["reference_sha256"] = sha256_bytes(reference_bytes)
                result["byte_identical"] = reference_bytes == payload
                if not result["byte_identical"]:
                    result.update(ok=False, error="corrected view is not byte-identical to the reference")
        return result

    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
        for n, result in enumerate(pool.map(work, members), 1):
            results.append(result)
            if n % 500 == 0 or n == len(members):
                print(f"  {n}/{len(members)}", flush=True)
    failures = [r for r in results if not r["ok"]]
    removed_rows = sum(len(r.get("removed_ids", [])) for r in results)
    summary = {
        "schema": "llm-overconfidence/corrected-raw-view/1",
        "status": "FAIL" if failures else "PASS",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy_id": POLICY_ID,
        "mask": {"path": str(args.mask.resolve()), "sha256": sha256_file(args.mask.resolve())},
        "raw_manifest": {"path": str(args.raw_manifest.resolve()), "sha256": sha256_file(args.raw_manifest.resolve())},
        "raw_root": str(raw), "out": str(out) if out else None,
        "input_identity_mode": args.input_identity,
        "semantic_identity_sha256": sha256_file(args.semantic_identity) if semantic else None,
        "provenance_note": "correction_source_sha256 identifies the archived scientific source; input_container_sha256 records actual acquired bytes separately; semantic mode does not claim acquisition ZIP byte identity",
        "input_container_hashes": {r["rel"]:r.get("input_container_sha256") for r in results},
        "verify_against": str(verify) if verify else None,
        "n_files": len(results), "n_rows_removed": removed_rows,
        "n_failed": len(failures),
        "aggregate": {
            "definition": "sha256 over sorted '<rel> <output_sha256>' lines",
            "digest": sha256_bytes("".join(f"{r['rel']} {r['output_sha256']}\n"
                                           for r in sorted(results, key=lambda r: r["rel"]) if "output_sha256" in r).encode()),
        },
        "byte_identical_to_reference": (all(r.get("byte_identical") for r in results) if verify else None),
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "failures": failures[:20],
        "note": "Deterministic corrected input construction; statistical estimation is performed by the separate analysis pipeline",
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_files", "n_rows_removed", "n_failed", "aggregate")} | 
                     {"byte_identical_to_reference": summary["byte_identical_to_reference"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
