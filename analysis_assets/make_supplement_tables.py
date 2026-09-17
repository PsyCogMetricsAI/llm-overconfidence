#!/usr/bin/env python3
"""Render supplementary tables from frozen experiment outputs; never rerun fits.

Run: python make_supplement_tables.py --experiment VERIFIED_RESULT_DIR
Add --check for byte-for-byte verification without modifying the four table files.
Output defaults to this directory; use --output-dir for isolated generation.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
from pathlib import Path

FINAL = Path(__file__).resolve().parent
EXPERIMENT = FINAL.parent / 'confidence_comparison' / 'risk_metrics_v1'
ROW_END = r'\\' + '\n'
LONG_END = r'\bottomrule\end{longtable}\normalsize' + '\n'

PRIMARY_HEADER = r'''\small
\begin{longtable}{llrrrrl}
\caption{All 31 R50 contrasts. $\Delta$ and interval endpoints are in MSE units multiplied by $10^4$; relative improvement is a percentage. $F_+$ counts positive outer folds.}\label{tab:allresults}\\
\toprule Contrast & Baseline & $100I$ & $10^4\Delta$ & $10^4$ simultaneous CI & $F_+$ & Rule\\\midrule\endfirsthead
\multicolumn{7}{l}{Table \thetable\ continued}\\\toprule Contrast & Baseline & $100I$ & $10^4\Delta$ & $10^4$ simultaneous CI & $F_+$ & Rule\\\midrule\endhead
'''
STABILITY_HEADER = r'''\begin{table}[htbp]\centering\small
\caption{Source availability and descriptive split-half correlation. Valid models are full-metric and paired-half counts, which coincide here; organizations count valid pairs.}\label{tab:stability}
\begin{tabular}{lrrr}\toprule Indicator & Valid models & Organizations & Spearman $\rho$\\\midrule
'''
SECONDARY_HEADER = r'''\small
\begin{longtable}{llrrr}
\caption{Secondary REL10 contrasts: relative MSE improvement (\%), absolute $\Delta$ and unadjusted 95\% percentile CI multiplied by $10^4$. These are descriptive.}\label{tab:secondary}\\
\toprule Contrast & Baseline & $100I$ & $10^4\Delta$ & $10^4$ CI\\\midrule\endfirsthead
\multicolumn{5}{l}{Table \thetable\ continued}\\\toprule Contrast & Baseline & $100I$ & $10^4\Delta$ & $10^4$ CI\\\midrule\endhead
'''
SELECTION_HEADER = r'''\small
\begin{longtable}{llrrr}
\caption{All descriptive selection contrasts. $U_f$ is mean selected-model target R50; $D=U_B-U_f$ is the reduction relative to the named baseline. Intervals are unadjusted 95\% case-organization bootstrap intervals; all reported numbers use risk units multiplied by $10^4$.}\label{tab:selection}\\
\toprule Method & Baseline & $10^4U_f$ & $10^4D$ & $10^4$ CI for $D$\\\midrule\endfirsthead
\multicolumn{5}{l}{Table \thetable\ continued}\\\toprule Method & Baseline & $10^4U_f$ & $10^4D$ & $10^4$ CI for $D$\\\midrule\endhead
'''


def tex_id(value: str) -> str:
    return value.replace('_', r'\_')


def load_json(relative: str):
    return json.loads((EXPERIMENT / relative).read_text())


def check_ids(rows):
    expected = [f'P_{group}{i}' for group, n in [('A', 5), ('B', 7), ('C', 4), ('D', 4)] for i in range(1, n + 1)]
    expected += [f'Q_{group}{i}' for group in ['C', 'D'] for i in range(1, 5)]
    expected += ['J', 'S0', 'Sd']
    if [r['id'] for r in rows] != expected:
        raise ValueError('Frozen contrast IDs/order differ; inspect before rendering.')


def primary_table() -> str:
    with (EXPERIMENT / 'CANDIDATE_MATRIX.csv').open(newline='') as source:
        rows = list(csv.DictReader(source))
    check_ids(rows)
    text = PRIMARY_HEADER
    labels = {'EXPLORATORY_CANDIDATE': 'Pass', 'NO_PREDEFINED_INCREMENT': 'No'}
    for row in rows:
        base = {'B0': '$B_0$', 'Bd': '$B_d$'}[row['baseline']]
        text += (f"{tex_id(row['id'])} & {base} & "
                 f"{100 * float(row['relative_improvement']):.3f} & "
                 f"{1e4 * float(row['delta']):.6f} & "
                 f"$[{1e4 * float(row['simultaneous_CI_low']):.6f}, "
                 f"{1e4 * float(row['simultaneous_CI_high']):.6f}]$ & "
                 f"{row['positive_outer_folds']}/5 & {labels[row['status']]}" + ROW_END)
    return text + LONG_END


def stability_table() -> str:
    rows = load_json('features/source_quality.json')['stability']
    if len(rows) != 20:
        raise ValueError('Expected all 20 source indicators.')
    text = STABILITY_HEADER
    for row in rows:
        if row['n_valid_full'] != row['n_valid_pairs']:
            raise ValueError('Caption requires equal full and paired-half counts.')
        text += (f"{row['metric']} & {row['n_valid_full']:,} & "
                 f"{row['n_orgs_valid_pairs']:,} & {row['half_spearman']:.3f}" + ROW_END)
    return text + r'\bottomrule\end{tabular}\end{table}' + '\n'


def secondary_table() -> str:
    rows = load_json('results/SECONDARY_RESULTS.json')['contrasts']
    check_ids(rows)
    text = SECONDARY_HEADER
    for row in rows:
        text += (f"{tex_id(row['id'])} & {row['baseline']} & "
                 f"{100 * row['relative_improvement']:.3f} & {1e4 * row['delta']:.6f} & "
                 f"$[{1e4 * row['CI95'][0]:.6f}, {1e4 * row['CI95'][1]:.6f}]$" + ROW_END)
    return text + LONG_END


def selection_table() -> str:
    data = load_json('results/selection_results.json')
    rows = data['contrasts']
    check_ids(rows)
    text = SELECTION_HEADER
    for row in rows:
        selected_risk = data['groups'][row['method']]['U']
        text += (f"{tex_id(row['id'])} & {row['baseline']} & {1e4 * selected_risk:.4f} & "
                 f"{1e4 * row['D']:.6f} & "
                 f"$[{1e4 * row['D_CI95'][0]:.6f}, {1e4 * row['D_CI95'][1]:.6f}]$" + ROW_END)
    return text + LONG_END


def main() -> int:
    global EXPERIMENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Report byte-level drift without writing.')
    parser.add_argument("--experiment", default=EXPERIMENT, type=Path, help="Verified experiment result directory.")
    parser.add_argument('--output-dir', type=Path, default=FINAL, help='Destination directory (or reference directory with --check).')
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not args.check:
        output_dir.mkdir(parents=True, exist_ok=True)
    EXPERIMENT = args.experiment.resolve()
    rendered = {
        'supplement_results_table.tex': primary_table(),
        'supplement_stability_table.tex': stability_table(),
        'supplement_secondary_table.tex': secondary_table(),
        'supplement_selection_table.tex': selection_table(),
    }
    mismatch = False
    for filename, content in rendered.items():
        path = output_dir / filename
        old = path.read_text() if path.exists() else ''
        equal = old == content
        if args.check and not equal:
            mismatch = True
            print(''.join(difflib.unified_diff(old.splitlines(True), content.splitlines(True),
                                             fromfile=filename, tofile=filename + ' (regenerated)')))
        elif not args.check:
            path.write_text(content)
        status = 'MATCH' if equal else ('DIFF' if args.check else 'UPDATED')
        print(f'{status} {filename} SHA256={hashlib.sha256(content.encode()).hexdigest()}')
    return int(mismatch)


if __name__ == '__main__':
    raise SystemExit(main())
