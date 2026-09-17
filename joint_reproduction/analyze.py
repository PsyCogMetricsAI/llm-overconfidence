#!/usr/bin/env python3
"""Recompute full-panel statistics from fitted vectors, without local project paths.

The hash-bound original analysis is loaded with only its project-path declarations
rebound. Its calculations are unchanged. Gauge logits use an exactly derived,
manifest-bound item-length vector, so the large response matrix is not needed for
this fitted-result analysis. This is not a cold re-fit of the latent model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arc-fit', type=Path, help='directory containing a completed full ARC fit')
    p.add_argument('--hellaswag-fit', type=Path, help='directory containing a completed full HellaSwag fit')
    a = p.parse_args()
    if bool(a.arc_fit) != bool(a.hellaswag_fit):
        p.error('--arc-fit and --hellaswag-fit must be supplied together')
    fresh = a.arc_fit is not None
    if a.out.exists():
        raise SystemExit('Output already exists; select a new output directory.')
    manifest = json.loads((ROOT / 'ARTIFACTS.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Artifact hash mismatch: {relative}')
    src = ROOT / 'original_code/analyze_joint.py'
    text = src.read_text()
    mod = types.ModuleType('portable_original_joint_analysis')
    mod.__file__ = str(src)
    exec(compile(text, str(src), 'exec'), mod.__dict__)

    def load_length(results_dir, bench):
        rec = manifest['item_length'][bench]
        value = np.load(ROOT / rec['path'], allow_pickle=False)
        run = json.loads((Path(results_dir) / f'joint_run_{bench}_full_2500ep.json').read_text())
        if rec['source_matrix_sha256'] != run['matrix_sha256']:
            raise RuntimeError('Item-length source matrix mismatch')
        if value.shape != (rec['n_items'],) or not np.isfinite(value).all():
            raise RuntimeError('Invalid derived item-length vector')
        return {'Lt': value, 'matrix': 'not_required_for_fitted_result_analysis',
                'matrix_sha256': rec['source_matrix_sha256'],
                'matches_run_json_sha256': True, 'derivation': rec['derivation'],
                'derived_input_sha256': rec['sha256']}

    mod.load_item_length = load_length
    selected = {}
    contract = json.loads((ROOT / 'input_contract/CONTRACT.json').read_text())
    for bench, directory in [('arc', a.arc_fit), ('hellaswag', a.hellaswag_fit)]:
        directory = directory if fresh else ROOT / 'reference_results'
        run_path = directory / f'joint_run_{bench}_full_2500ep.json'
        vector_path = directory / f'joint_vectors_{bench}_full_2500ep.npz'
        run = json.loads(run_path.read_text())
        bc = contract['benches'][bench]
        fixed = contract['rules']
        required_config = {'n_epochs': fixed['epochs'], 'seed': fixed['seed'],
                           'phi': fixed['phi'], 'clamp_eps': fixed['clamp_eps'],
                           'dtype': fixed['dtype'], 'lr': bc['lr'], 'grad_clip': bc['grad_clip']}
        if run.get('status') != 'FULL_COMPLETED' or run.get('mode') != 'full' or run.get('bench') != bench:
            raise RuntimeError('Fit is not a completed full run: ' + bench)
        for key, value in required_config.items():
            if run.get('config', {}).get(key) != value:
                raise RuntimeError('Fit configuration mismatch: ' + bench + ':' + key)
        reference_run = json.loads((ROOT / 'reference_results' / run_path.name).read_text())
        for key in ['response', 'Lt', 'channel']:
            if run.get('config', {}).get(key) != reference_run['config'][key]:
                raise RuntimeError('Fit response definition mismatch: ' + bench + ':' + key)
        if run.get('n_models') != len(bc['analysis_models']) or run.get('n_items') != bc['arrays']['BIN']['shape'][0]:
            raise RuntimeError('Fit panel count mismatch: ' + bench)
        if run.get('matrix_sha256') != bc['matrix_archived_sha256'] or run.get('matrix_array_sha256') != reference_run['matrix_array_sha256']:
            raise RuntimeError('Fit matrix identity mismatch: ' + bench)
        if run.get('matrix_shape_items_models') != bc['arrays']['BIN']['shape']:
            raise RuntimeError('Fit matrix shape mismatch: ' + bench)
        if run.get('original_modules') != {key: contract['code_hashes'][key] for key in ['joint_irt.py', 'beta_irt.py']}:
            raise RuntimeError('Fit original module identity mismatch: ' + bench)
        if digest(vector_path) != run.get('out_npz_sha256'):
            raise RuntimeError('Fit vector hash mismatch: ' + bench)
        with np.load(vector_path, allow_pickle=False) as vectors:
            if vectors['models'].astype(str).tolist() != bc['analysis_models']:
                raise RuntimeError('Fit model order mismatch: ' + bench)
            if int(vectors['fit_epochs']) != fixed['epochs'] or str(vectors['mode']) != 'full':
                raise RuntimeError('Fit vector epoch/mode mismatch: ' + bench)
        selected[bench] = {'run': run_path, 'vectors': vector_path}
    a.out.mkdir(parents=True)
    for files in selected.values():
        for original in files.values():
            shutil.copy2(original, a.out / original.name)
    report = mod.build_report(str(a.out))
    report['portable_adapter'] = {'scope': 'Analysis of supplied completed full fits; this command does not refit' if fresh else 'Bundled fitted-result recomputation; not a cold model fit',
                                  'input_mode': 'supplied_full_fits' if fresh else 'bundled_reference_fits',
                                  'fit_inputs': {bench: {key: {'path': str(path.resolve()), 'sha256': digest(path)} for key, path in files.items()} for bench, files in selected.items()},
                                  'original_analyzer_sha256': digest(src),
                                  'artifact_manifest_sha256': digest(ROOT / 'ARTIFACTS.json'),
                                  'wrapper_sha256': digest(Path(__file__))}
    (a.out / 'analyze_joint_report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    mod.write_tables(report, str(a.out))
    print(json.dumps({'status': 'COMPUTED', 'out': str(a.out), 'n_shared': report['pairing']['n_shared']}))


if __name__ == '__main__':
    main()
