#!/usr/bin/env python3
"""Rebind only the archived result verifier's obsolete cohort constants.

The archived source is hash-checked and unchanged. All numerical/statistical,
selection, warm-cost and provenance assertions remain in the runtime copy.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'core_reproduction/original_inputs/confidence_comparison/risk_metrics_v1/reviews/verify_results.py'
REFERENCE = ROOT / 'confidence_comparison/risk_metrics_v1/evaluation_cohort.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapt(core):
    hashes = json.loads((ROOT / 'core_reproduction/INPUT_HASHES.json').read_text())
    key = 'confidence_comparison/risk_metrics_v1/reviews/verify_results.py'
    if sha(ORIGINAL) != hashes[key]:
        raise ValueError('ARCHIVED_RESULT_VERIFIER_CHANGED')
    reference_sha = sha(REFERENCE)
    if reference_sha != '2bcb9324a918ce871ec85ff4af6f8a54c2e7bb77c5dc644edd04eeb5d2410e9b':
        raise ValueError('CORRECTED_REFERENCE_COHORT_CHANGED')
    if sha(core / 'evaluation_cohort.csv') != reference_sha:
        raise ValueError('RUNTIME_CORRECTED_COHORT_CHANGED')
    source = ORIGINAL.read_text()
    before = '    assert set(by)=={r[\'model_id\'] for r in oo} and len(oo)==len(by)==6666'
    after = "\n".join([
        "    # Explicit corrected-cohort identity; every original numerical assertion follows unchanged.",
        f"    assert h(W/'evaluation_cohort.csv')=='{reference_sha}'",
        "    corrected_cohort=rows(W/'evaluation_cohort.csv');expected_by={r['model_id']:r['org_id'] for r in corrected_cohort}",
        "    assert len(corrected_cohort)==len(expected_by)==6701 and len(set(expected_by.values()))==1330",
        "    assert set(by)=={r['model_id'] for r in oo}==set(expected_by) and len(oo)==len(by)==6701",
        "    assert all(r['org_id']==expected_by[r['model_id']] for r in oo)",
    ])
    org_before='assert len(orgs)==1324'
    if source.count(before) != 1 or source.count(org_before) != 1:
        raise ValueError('UNEXPECTED_ORIGINAL_COHORT_ASSERTIONS')
    modified = source.replace(before, after).replace(org_before, 'assert len(orgs)==1330')
    dest = core / 'reviews/verify_results.py'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(modified)
    diff = ''.join(difflib.unified_diff(source.splitlines(True), modified.splitlines(True), fromfile='archived/verify_results.py', tofile='corrected_runtime/verify_results.py'))
    (dest.parent / 'CORRECTED_RESULT_ADAPTER.diff').write_text(diff)
    record = {'scope': 'Corrected cohort identities/counts only; numerical/statistical/selection/cost assertions retained',
              'original_sha256': sha(ORIGINAL), 'adapted_sha256': sha(dest),
              'adapter_sha256': sha(Path(__file__)), 'evaluation_cohort_sha256': reference_sha,
              'old_counts': [6666, 1324], 'corrected_counts': [6701, 1330],
              'diff_sha256': sha(dest.parent / 'CORRECTED_RESULT_ADAPTER.diff')}
    (dest.parent / 'CORRECTED_RESULT_ADAPTER.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core', type=Path, required=True)
    print(json.dumps(adapt(p.parse_args().core.resolve()), indent=2))
