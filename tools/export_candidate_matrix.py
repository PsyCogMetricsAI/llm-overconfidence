#!/usr/bin/env python3
"""Serialize the 31 primary comparison records for paper figures/tables.

No estimation, thresholding or status changes occur here.
"""
import argparse
import csv
import json
from pathlib import Path

FIELDS = ['id', 'metric', 'baseline', 'status', 'delta', 'relative_improvement',
          'simultaneous_CI_low', 'simultaneous_CI_high', 'positive_outer_folds', 'reason']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.experiment / 'results/PRIMARY_RESULTS.json').read_text())
    records = data['contrasts']
    assert len(records) == 31 and len({r['id'] for r in records}) == 31
    rows = []
    for record in records:
        row = {key: record.get(key, '') for key in FIELDS}
        interval = record.get('simultaneous_CI')
        row['simultaneous_CI_low'] = interval[0] if interval is not None else ''
        row['simultaneous_CI_high'] = interval[1] if interval is not None else ''
        rows.append(row)
    output = args.experiment / 'CANDIDATE_MATRIX.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'output': str(output), 'rows': len(rows), 'scope': 'primary results serialization only'}))


if __name__ == '__main__':
    main()
