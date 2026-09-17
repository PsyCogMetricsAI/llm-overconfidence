"""E2 fixed three-model source pilot / E3 A-B extraction; no target reader."""

import argparse
import os
import time
from pathlib import Path

from common import (ROOT, digest, dump, load_source, readjson, require_data_release,
                    require_design, timer)
import numpy as np
from metrics_ab import METRIC_IDS, SPLIT_IDS, compute_ab_splits, prepare_ab_items


def _identity(root):
    return dict(E1_bundle_sha256=digest(root / 'E1_BUNDLE.json'),
                protocol_sha256=digest(root / 'protocol.json'),
                formula_registry_sha256=digest(root / 'formula_registry.json'),
                source_arrays_sha256=readjson(root / 'cohort_contract.json')['legacy']['source_arrays_sha256'],
                code_sha256={p: digest(root / 'code' / p) for p in
                             ('metrics_ab.py', 'source_ab.py', 'legacy_primitives.py', 'common.py')})


def _read_arc(rec, root):
    path = Path(rec['arc_path'])
    if path.parent.name != 'arc':
        raise ValueError('source_path_is_not_arc')
    with timer(root, 'source_ab_io', model_id=rec['model_id']):
        if digest(path) != rec['arc_sha256']:
            raise ValueError('raw_arc_hash_mismatch_before_read')
        with np.load(path, allow_pickle=False) as src:
            raw = {key: src[key] for key in ('loglik', 'char_lens', 'gold', 'item_ids', 'n_options')}
        if digest(path) != rec['arc_sha256']:
            raise ValueError('raw_arc_hash_mismatch_after_read')
    return raw


def _compute_model(rec, index, data, item_index, root):
    start = time.perf_counter()
    raw = _read_arc(rec, root)
    with timer(root, 'source_ab_compute', model_id=rec['model_id']):
        items = prepare_ab_items(**raw)
        try:
            cols = np.asarray([item_index[item] for item in items['item_ids']], dtype=np.int64)
        except KeyError as exc:
            raise ValueError('raw_arc_item_not_in_frozen_universe') from exc
        full_valid = np.zeros(len(data['item_ids']), dtype=bool)
        full_valid[cols] = items['valid']
        if not np.array_equal(full_valid, data['valid'][index]):
            raise ValueError('frozen_source_valid_mask_mismatch')
        keep = items['valid']
        for key in ('ell', 'c'):
            if not np.allclose(items[key][keep], data[key][index, cols][keep], atol=1e-12, rtol=0):
                raise ValueError(f'frozen_source_{key}_mismatch')
        if not np.array_equal(items['y'][keep], data['y'][index, cols][keep]):
            raise ValueError('frozen_source_y_mismatch')
        blocks = compute_ab_splits(items, data['source_half'][cols])
    return items, blocks, cols, time.perf_counter() - start


def pilot(root=ROOT):
    """Compute only the first three canonical model IDs; save timing, no features."""
    root = Path(root)
    require_data_release(root)
    identity = _identity(root)
    destination = root / 'resource_pilot_ab.json'
    if destination.exists():
        report = readjson(destination)
        if report['identity'] != identity:
            raise RuntimeError('AB pilot identity changed; retain prior pilot and diagnose')
        return report
    with timer(root, 'source_ab_pilot_load'):
        data, records = load_source(root)
    index = {str(item): i for i, item in enumerate(data['item_ids'])}
    chosen = sorted(range(len(records)), key=lambda i: records[i]['model_id'])[:3]
    results = []
    for i in chosen:
        rec = records[i]
        _, blocks, _, seconds = _compute_model(rec, i, data, index, root)
        results.append(dict(model_id=rec['model_id'], arc_path=rec['arc_path'],
                            arc_sha256=rec['arc_sha256'], wall_seconds=seconds,
                            n_items={s: blocks[s]['n_items'] for s in SPLIT_IDS}))
    if not results:
        raise RuntimeError('empty source cohort')
    # Conservative measured-per-model extrapolation; fixed allowance for output
    # serialization is explicit, not misrepresented as a measured cold runtime.
    estimate = 2 * max(r['wall_seconds'] for r in results) * len(records) + 120
    report = dict(status='SOURCE_ONLY_PILOT', identity=identity,
                  selection='canonical model_id ascending first three', models=results,
                  n_full_models=len(records), estimated_full_seconds=estimate,
                  estimate_rule='2 * maximum three measured raw-read-and-compute times * N + 120s I/O allowance',
                  over_node_budget=estimate > 43200, extra_LLM_calls=0,
                  target_access=False, candidate_effects_tested=False)
    dump(destination, report)
    return report


def run(root=ROOT):
    """Production extraction is unavailable until the independent code release."""
    root = Path(root)
    require_design(root)
    identity = _identity(root)
    feasibility = readjson(root / 'resource_pilot_ab.json')
    if feasibility['identity'] != identity:
        raise RuntimeError('pilot does not match released AB code/input identity')
    if feasibility['over_node_budget']:
        raise RuntimeError('H-BUDGET: AB estimate exceeds 12-hour node budget')
    dest = root / 'features' / 'ab_values.npz'
    manifest_path = root / 'features' / 'ab_manifest.json'
    if dest.exists() or manifest_path.exists():
        raise RuntimeError('AB production artifact already exists; do not overwrite')
    start = time.perf_counter()
    with timer(root, 'source_ab_load'):
        data, records = load_source(root)
    mids = data['model_ids'].astype(str)
    if [rec['model_id'] for rec in records] != mids.tolist():
        raise ValueError('source_model_manifest_order_mismatch')
    global_items = data['item_ids'].astype(str)
    if len(set(global_items.tolist())) != len(global_items):
        raise ValueError('duplicate frozen item ID')
    item_index = {item: i for i, item in enumerate(global_items)}
    shape = (len(records), len(SPLIT_IDS), len(METRIC_IDS))
    values = np.full(shape, np.nan, dtype=np.float64)
    reasons = np.full(shape, 'NOT_COMPUTED', dtype='<U32')
    n_items = np.zeros(shape[:2], dtype=np.int64)
    z = np.full((len(records), len(global_items)), np.nan, dtype=np.float64)
    diagnostics = []
    for i, rec in enumerate(records):
        if time.perf_counter() - start > 43200:
            raise RuntimeError('H-BUDGET: AB node exceeded 12 hours; preserve cost/progress')
        items, blocks, cols, seconds = _compute_model(rec, i, data, item_index, root)
        z[i, cols] = items['z']
        for s, split in enumerate(SPLIT_IDS):
            block = blocks[split]
            n_items[i, s] = block['n_items']
            for j, metric in enumerate(METRIC_IDS):
                values[i, s, j] = block['values'][metric]
                reasons[i, s, j] = block['reasons'][metric]
        diagnostics.append(dict(model_id=rec['model_id'], arc_sha256=rec['arc_sha256'],
                                valid_source_items=blocks['full']['n_items'],
                                validity_reasons=items['validity_reason_counts'],
                                raw_and_compute_wall_seconds=seconds))
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            dump(root / 'runtime' / 'source_ab_progress.json',
                 dict(status='RUNNING', completed_models=i + 1, total_models=len(records),
                      elapsed_seconds=time.perf_counter() - start, identity=identity))
    if not np.array_equal(np.isfinite(values), reasons == ''):
        raise AssertionError('AB values and missing reasons disagree')
    if not np.array_equal(np.isfinite(z), data['valid']):
        raise AssertionError('AB z validity differs from frozen source observations')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name('ab_values.partial.npz')
    with timer(root, 'source_ab_write', n_models=len(records)):
        with temp.open('xb') as stream:
            np.savez(stream, model_ids=mids, org_ids=data['org_ids'].astype(str),
                     item_ids=global_items, metric_ids=np.asarray(METRIC_IDS),
                     split_ids=np.asarray(SPLIT_IDS), values=values, reasons=reasons,
                     n_items=n_items, z=z)
        os.replace(temp, dest)
        dump(root / 'features' / 'ab_diagnostics.json', diagnostics)
    manifest = dict(status='COMPLETE_PENDING_V_SOURCE', identity=identity,
                    file=str(dest.relative_to(root)), sha256=digest(dest),
                    diagnostics_sha256=digest(root / 'features' / 'ab_diagnostics.json'),
                    schema=dict(values=list(values.shape), reasons=list(reasons.shape),
                                n_items=list(n_items.shape), z=list(z.shape),
                                metric_ids=list(METRIC_IDS), split_ids=list(SPLIT_IDS),
                                success_reason='', missing_numeric='NaN'),
                    n_models=len(records), n_items_global=len(global_items),
                    missing_values=int(np.isnan(values).sum()),
                    source_label_count=int(data['valid'].sum()), extra_LLM_calls=0,
                    elapsed_seconds=time.perf_counter() - start)
    dump(manifest_path, manifest)
    dump(root / 'runtime' / 'source_ab_progress.json',
         dict(status='COMPLETE_PENDING_V_SOURCE', completed_models=len(records),
              elapsed_seconds=time.perf_counter() - start, manifest_sha256=digest(manifest_path)))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('pilot', 'run'))
    args = parser.parse_args()
    result = (pilot if args.stage == 'pilot' else run)()
    import json
    print(json.dumps({k: result[k] for k in ('status', 'n_models', 'estimated_full_seconds') if k in result}))
