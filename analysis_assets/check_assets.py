#!/usr/bin/env python3
"""Check numerical analysis assets without manuscript sources or TeX/PDF tools.

Uses frozen results only; runs the two renderers in a temporary directory and
never fits models or changes reference assets. PDF bytes are not compared because
Matplotlib embeds creation timestamps; PNGs and all table bytes are compared.
This is an artifact consistency check, not independent scientific acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = [f'supplement_{name}_table.tex' for name in
          ('results', 'stability', 'secondary', 'selection')]
FIGURES = [f'figs/fig_{name}.{ext}' for name in ('concept', 'contrasts')
           for ext in ('pdf', 'png')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_primary(experiment, assets):
    require((experiment / 'results/E6_COMPLETE.json').is_file(), 'Experiment scoring incomplete')
    rows = csv_rows(experiment / 'CANDIDATE_MATRIX.csv')
    primary = json.loads((experiment / 'results/PRIMARY_RESULTS.json').read_text())
    contrasts = primary['contrasts']
    require(len(rows) == len(contrasts) == primary['n_contrasts'] == 31, 'Expected 31 primary contrasts')
    raw = {r['id']: r for r in contrasts}
    require(len(raw) == len({r['id'] for r in rows}) == 31, 'Duplicate primary contrast IDs')
    require(set(raw) == {r['id'] for r in rows}, 'Primary matrix/result IDs differ')
    parsed = {}
    for line in (assets / TABLES[0]).read_text().splitlines():
        if re.match(r'^(?:[PQ]\\_|J &|S0 &|Sd &)', line):
            cells = [c.strip() for c in line.rstrip('\\').split('&')]
            key = cells[0].replace('\\_', '_')
            require(key not in parsed, f'Duplicate table contrast {key}')
            parsed[key] = cells
    require(set(parsed) == set(raw), 'Primary table must contain all 31 rows once')
    for row in rows:
        ident = row['id']
        result = raw[ident]
        cells = parsed[ident]
        for key in ('delta', 'relative_improvement'):
            require(float(row[key]) == result[key], f'{ident}: {key} differs')
        for key in ('baseline', 'metric'):
            require(row[key] == (result[key] if result[key] is not None else ''), f'{ident}: {key} differs')
        lo, hi = result['simultaneous_CI']
        require(float(row['simultaneous_CI_low']) == lo and
                float(row['simultaneous_CI_high']) == hi, f'{ident}: interval differs')
        folds = result['positive_outer_folds']
        require(int(row['positive_outer_folds']) == folds, f'{ident}: fold count differs')
        passed = result['relative_improvement'] >= .05 and lo > 0 and folds >= 4
        status = 'EXPLORATORY_CANDIDATE' if passed else 'NO_PREDEFINED_INCREMENT'
        require(row['status'] == result['status'] == status, f'{ident}: support classification differs')
        expected = [ident.replace('_', r'\_'),
                    {'B0': '$B_0$', 'Bd': '$B_d$'}[row['baseline']],
                    f"{100 * result['relative_improvement']:.3f}",
                    f"{1e4 * result['delta']:.6f}",
                    f'$[{1e4 * lo:.6f}, {1e4 * hi:.6f}]$',
                    f'{folds}/5', 'Pass' if passed else 'No']
        require(cells == expected, f'{ident}: primary table transcription differs')
    return rows


def check_provenance(report, experiment, metadata_source, rows, cohort):
    require(report['matrix_sha256'] == digest(experiment / 'CANDIDATE_MATRIX.csv'), 'Figure matrix hash differs')
    selected = [r for r in rows if r['id'].startswith('Q_')]
    keys = ('id', 'metric', 'baseline', 'delta', 'simultaneous_CI_low', 'simultaneous_CI_high', 'status')
    require(report['contrasts'] == [{key: r[key] for key in keys} for r in selected], 'Figure contrast data differs')
    require(report['constructed_checks'] == [
        {'A': .8, 'C': .8, 'O': 0., 'R50': 0.},
        {'A': .8, 'C': .8, 'O': 0., 'R50': .4}], 'Constructed example values differ')
    meta = report['metadata']
    require(meta['evaluation_cohort_sha256'] == digest(experiment / 'evaluation_cohort.csv'), 'Cohort source hash differs')
    require(meta['source_sha256'] == digest(metadata_source), 'Metadata source hash differs')
    require(meta['n_models'] == len(cohort), 'Metadata model count differs')
    require(meta['n_orgs'] == len({r['org_id'] for r in cohort}), 'Metadata organization count differs')
    require(meta['types'] == dict(Counter(r['model_type'] for r in cohort)), 'Metadata type counts differ')
    require(meta['parameter_bins'] == dict(Counter(r['parameter_group'] for r in cohort)), 'Metadata parameter counts differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'confidence_comparison/risk_metrics_v1')
    parser.add_argument('--metadata-source', type=Path, default=ROOT / 'external_inputs/_model_type_labels.csv')
    parser.add_argument('--assets-dir', type=Path, default=HERE, help='Reference asset directory to verify.')
    args = parser.parse_args()
    experiment, metadata_source, assets = (p.resolve() for p in
                                          (args.experiment, args.metadata_source, args.assets_dir))
    rows = check_primary(experiment, assets)
    cohort = csv_rows(assets / 'corrected_cohort_metadata.csv')
    evaluated = csv_rows(experiment / 'evaluation_cohort.csv')
    require(len({r['model_id'] for r in cohort}) == len(cohort), 'Duplicate metadata model IDs')
    require([(r['model_id'], r['org_id']) for r in cohort] ==
            [(r['model_id'], r['org_id']) for r in evaluated], 'Metadata/cohort membership or order differs')
    check_provenance(json.loads((assets / 'FIGURE_GENERATION.json').read_text()),
                     experiment, metadata_source, rows, cohort)
    with tempfile.TemporaryDirectory(prefix='analysis-asset-check-') as temporary:
        output = Path(temporary)
        subprocess.run([sys.executable, str(HERE / 'make_supplement_tables.py'),
                        '--experiment', str(experiment), '--output-dir', str(output)], check=True)
        subprocess.run([sys.executable, str(HERE / 'make_revision_figures.py'),
                        '--experiment', str(experiment), '--metadata-source', str(metadata_source),
                        '--output-dir', str(output)], check=True)
        for name in [*TABLES, 'figs/fig_concept.png', 'figs/fig_contrasts.png', 'corrected_cohort_metadata.csv']:
            require((assets / name).read_bytes() == (output / name).read_bytes(), f'Regeneration differs: {name}')
        for name in FIGURES:
            require((assets / name).is_file() and (assets / name).stat().st_size > 0, f'Missing figure: {name}')
        for name in ('figs/fig_concept.pdf', 'figs/fig_contrasts.pdf'):
            require((assets / name).read_bytes().startswith(b'%PDF-') and
                    (output / name).read_bytes().startswith(b'%PDF-'), f'Invalid PDF header: {name}')
        check_provenance(json.loads((output / 'FIGURE_GENERATION.json').read_text()),
                         experiment, metadata_source, rows, cohort)
    files = [*TABLES, *FIGURES, 'corrected_cohort_metadata.csv', 'FIGURE_GENERATION.json']
    print(json.dumps({'status': 'PASS', 'scope': 'Numerical analysis asset consistency; not independent scientific acceptance',
                      'primary_comparisons_checked': 31, 'tables_regenerated': 4,
                      'png_figures_regenerated': 2, 'pdf_check': 'presence and header only; timestamped PDF bytes not compared',
                      'cohort_models': len(cohort), 'sha256': {name: digest(assets / name) for name in files}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
