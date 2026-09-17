#!/usr/bin/env python3
"""Clarify two cost-report metadata fields without changing measured values."""
import argparse
import hashlib
import json
from pathlib import Path

STATUS = 'UNMEASURED: no complete acquisition-to-results cold timer; separate stage timings are available'
DISK_SCOPE = 'Total result-tree bytes at the report snapshot, not a new-byte creation delta'


def annotate(core):
    path = Path(core) / 'results/cost_report.json'
    before = path.read_bytes()
    report = json.loads(before)
    if report.get('complete_cold_end_to_end_seconds') is not None or not report.get('complete_cold_status', '').startswith('UNMEASURED'):
        raise ValueError('UNEXPECTED_COLD_COST_SCHEMA')
    if 'new_output_disk_bytes' not in report:
        raise ValueError('MISSING_DISK_SNAPSHOT_FIELD')
    report['complete_cold_status'] = STATUS
    report['new_output_disk_bytes_scope'] = DISK_SCOPE
    # The original key is retained for compatibility; the number is never recalculated.
    path.write_text(json.dumps(report, indent=2)+'\n')
    return {'scope': 'Metadata only; all measured values and bound event hashes retained',
            'before_sha256': hashlib.sha256(before).hexdigest(),
            'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'changed_fields': ['complete_cold_status', 'new_output_disk_bytes_scope']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core', type=Path, required=True)
    print(json.dumps(annotate(p.parse_args().core.resolve()), indent=2))
