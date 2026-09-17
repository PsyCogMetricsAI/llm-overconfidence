"""E7 fixed source-to-prediction warm measurements. No target-data reader."""
from common import (ROOT, digest, dump, readjson, rows, write_rows, protocol,
                    require_design, timer)
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp

from metrics_ab import METRIC_IDS as AB_IDS, prepare_ab_items, compute_ab
from metrics_structure import METRIC_IDS as CD_IDS, compute_structure_one
from modeling import predict_package


def base9(raw, items):
    """Exact historical nine source summaries, sharing already prepared c/y."""
    valid, c, y = items['valid'], items['c'], items['y']
    ids = items['item_ids']
    use = np.flatnonzero(valid)
    n = len(use)
    if not n:
        raise ValueError('NO_VALID_SOURCE_ITEMS')
    cc, yy = c[use], y[use].astype(float)
    bins = np.minimum(np.floor(10 * cc).astype(int), 9)
    rel = ece = 0.
    for b in range(10):
        take = bins == b
        if take.any():
            gap = float(cc[take].mean() - yy[take].mean())
            rel += take.mean() * gap ** 2
            ece += take.mean() * abs(gap)
    order = np.lexsort((ids[use], -cc))
    out = dict(A=float(yy.mean()), C=float(cc.mean()), O=float(cc.mean() - yy.mean()),
               REL10=float(rel), R50=float((1 - yy[order[:(n + 1) // 2]]).mean()))
    ll, hh = np.asarray(raw['loglik'], float)[use], np.asarray(raw['char_lens'], float)[use]
    real = np.isfinite(hh) & (hh > 0)
    u = np.full(ll.shape, -np.inf)
    np.divide(ll, hh, out=u, where=real)
    p = np.exp(u - logsumexp(u, axis=1, keepdims=True))
    gold = np.asarray(raw['gold'])[use].astype(int)
    truth = np.zeros_like(p)
    truth[np.arange(n), gold] = 1
    lp = np.zeros_like(p)
    np.log(p, out=lp, where=p > 0)
    pp = np.sort(p, axis=1)
    out.update(ECE10=float(ece), Brier=float(np.mean(np.sum((p - truth) ** 2, axis=1))),
               H=float(np.mean(-np.sum(p * lp, axis=1) / np.log(real.sum(1)))),
               Margin=float(np.mean(pp[:, -1] - pp[:, -2])))
    if not np.isfinite(list(out.values())).all():
        raise ValueError('NONFINITE_BASE_NINE')
    return out


def extract_all(raw, item_ids, source_half, reference):
    """Rebuild base9 + 20 values + all 20 missing flags from one raw ARC record."""
    items = prepare_ab_items(**raw)
    global_ids = np.asarray(item_ids).astype(str)
    halves = np.asarray(source_half)
    if len(set(global_ids.tolist())) != len(global_ids) or halves.shape != global_ids.shape:
        raise ValueError('WARM_SOURCE_AXIS_MISMATCH')
    positions = {item: j for j, item in enumerate(global_ids)}
    try:
        cols = np.array([positions[item] for item in items['item_ids']], dtype=int)
    except KeyError as exc:
        raise ValueError('WARM_RAW_ITEM_OUTSIDE_FROZEN_UNIVERSE') from exc
    aligned = dict(item_ids=global_ids, source_half=halves,
                   valid=np.zeros(len(global_ids), bool), y=np.zeros(len(global_ids), bool))
    for name in ('ell', 'c', 'z'):
        aligned[name] = np.full(len(global_ids), np.nan)
    for name in ('ell', 'c', 'z', 'y', 'valid'):
        aligned[name][cols] = items[name]
    out = base9(raw, items)
    ab = compute_ab(items)
    cd = compute_structure_one(aligned, reference)
    out.update(ab['values'])
    out.update({metric: float(cd['values'][j]) for j, metric in enumerate(CD_IDS)})
    reasons = dict(ab['reasons'])
    reasons.update({metric: str(cd['reasons'][j]) for j, metric in enumerate(CD_IDS)})
    for metric in AB_IDS + CD_IDS:
        out['missing_' + metric] = float(not np.isfinite(out[metric]))
        if (out['missing_' + metric] == 0) != (reasons[metric] == ''):
            raise AssertionError('WARM_VALUE_REASON_MISMATCH')
    return out, reasons


def method_selection(pr, logs):
    """Resolve each endpoint's sealed chosen package; never inspect losses."""
    packages, procedures = {}, {}
    for outer_log in logs:
        for rec in outer_log['packages']:
            key = (int(rec['outer_fold']), rec['group'], rec['endpoint'])
            if key in packages:
                raise ValueError('DUPLICATE_PACKAGE_SELECTION')
            packages[key] = rec
        for rec in outer_log['procedures']:
            key = (int(rec['outer_fold']), rec['procedure'], rec['endpoint'])
            if key in procedures:
                raise ValueError('DUPLICATE_PROCEDURE_SELECTION')
            procedures[key] = rec
    resolved = {}
    folds = sorted({k[0] for k in packages})
    for fold in folds:
        for method in list(pr['feature_groups']) + list(pr['procedures']):
            for ep in pr['endpoints']:
                if method in pr['feature_groups']:
                    chosen = method
                else:
                    rec = procedures[fold, method, ep]
                    chosen = rec['chosen_package']
                    if chosen not in pr['procedures'][method]:
                        raise ValueError('PROCEDURE_CHOICE_OUTSIDE_FROZEN_LIBRARY')
                    if rec['best_estimator'] != packages[fold, chosen, ep]['best_estimator']:
                        raise ValueError('PROCEDURE_ESTIMATOR_MISMATCH')
                resolved[fold, method, ep] = dict(package=chosen,
                                                 estimator=packages[fold, chosen, ep]['best_estimator'])
    return resolved


def fixed_schedule(model_ids, methods, seed=20260914):
    ids = list(map(str, model_ids))
    if len(ids) != len(set(ids)) or len(methods) != len(set(methods)):
        raise ValueError('DUPLICATE_SCHEDULE_ID')
    selected = sorted(ids, key=lambda mid: (hashlib.sha256(f'{seed}|{mid}'.encode()).hexdigest(), mid))[:100]
    rng = np.random.default_rng(seed)
    schedule = []
    for repeat in range(3):
        for mid in selected:
            for position, method in enumerate(rng.permutation(methods)):
                schedule.append(dict(repetition=repeat, model_id=mid,
                                     order_within_model=position, method=str(method)))
    return selected, schedule


def _duration(start):
    return time.perf_counter() - start[0], time.process_time() - start[1]


def _load_raw(rec):
    path = Path(rec['arc_path'])
    if path.parent.name != 'arc':
        raise ValueError('WARM_READER_ONLY_ACCEPTS_ARC')
    if digest(path) != rec['arc_sha256']:
        raise ValueError('WARM_RAW_HASH_BEFORE_READ')
    with np.load(path, allow_pickle=False) as data:
        raw = {key: data[key] for key in ('loglik', 'char_lens', 'gold', 'item_ids', 'n_options')}
    if digest(path) != rec['arc_sha256']:
        raise ValueError('WARM_RAW_HASH_AFTER_READ')
    return raw


def run(root=ROOT):
    root = Path(root)
    node_start = time.perf_counter()
    require_design(root)
    release = readjson(root / 'SCORING_RELEASE.json')
    if release['verdict'] != 'ACCEPT' or release['seal_sha256'] != digest(root / 'PREDICTION_SEAL.json'):
        raise RuntimeError('WARM_REQUIRES_CURRENT_V_SEAL')
    seal = readjson(root / 'PREDICTION_SEAL.json')
    if seal['CODE_RELEASE_sha256'] != digest(root / 'CODE_RELEASE.json'):
        raise RuntimeError('WARM_CODE_RELEASE_NOT_SEALED')
    if seal['source_release_sha256'] != digest(root / 'SOURCE_RELEASE.json'):
        raise RuntimeError('WARM_SOURCE_RELEASE_NOT_SEALED')
    source_release = readjson(root / 'SOURCE_RELEASE.json')
    if source_release['verdict'] != 'ACCEPT' or source_release['features_ready_sha256'] != digest(root / 'features/FEATURES_READY.json'):
        raise RuntimeError('WARM_SOURCE_FEATURE_RELEASE_MISMATCH')
    feature_ready = readjson(root / 'features/FEATURES_READY.json')
    if feature_ready['structure_ready_sha256'] != digest(root / 'features/STRUCTURE_READY.json'):
        raise RuntimeError('WARM_STRUCTURE_RELEASE_MISMATCH')
    if readjson(root / 'features/STRUCTURE_READY.json')['references_ready_sha256'] != digest(root / 'references/REFERENCES_READY.json'):
        raise RuntimeError('WARM_REFERENCE_RELEASE_MISMATCH')
    for name, sha in seal['artifacts'].items():
        if digest(root / name) != sha:
            raise RuntimeError('WARM_SEALED_ARTIFACT_CHANGED:' + name)
    outdir = root / 'cost'
    outdir.mkdir(exist_ok=True)
    measurement_path = outdir / 'warm_measurements.jsonl'
    if measurement_path.exists() or (outdir / 'warm_summary.json').exists():
        raise RuntimeError('warm artifacts already exist; preserve existing measurements')
    pr = protocol(root)
    methods = list(pr['feature_groups']) + list(pr['procedures'])
    if len(methods) != 33 or pr['endpoints'] != ['Y_sel', 'Y_cal']:
        raise ValueError('WARM_FROZEN_METHOD_OR_ENDPOINT_COUNT')
    setup_start = (time.perf_counter(), time.process_time())
    with timer(root, 'warm_setup'):
        # Only sealed predictions are loaded. There is no scoring-target API.
        predictions = rows(root / 'predictions_oof.csv')
        by_prediction = {rec['model_id']: rec for rec in predictions}
        if len(by_prediction) != len(predictions):
            raise ValueError('DUPLICATE_WARM_OOF_MODEL')
        selected, schedule = fixed_schedule(list(by_prediction), methods)
        if not selected:
            raise ValueError('EMPTY_WARM_OOF_COHORT')
        raw_records = {rec['model_id']: rec for rec in rows(root / 'cohort_manifest.csv')}
        source_path = Path(pr['source_arrays'])
        if digest(source_path) != readjson(root / 'cohort_contract.json')['legacy']['source_arrays_sha256']:
            raise ValueError('WARM_SOURCE_METADATA_HASH')
        with np.load(source_path, allow_pickle=False) as source:
            global_ids, halves = source['item_ids'], source['source_half']
        contexts = {int(c['context'][5:].split('/')[0]): c
                    for c in readjson(root / 'split_manifest.json')['projection_contexts']
                    if c['context'].endswith('/test')}
        needed_folds = sorted({int(by_prediction[mid]['outer_fold']) for mid in selected})
        ref_ready = readjson(root / 'references/REFERENCES_READY.json')
        references, reference_provenance = {}, {}
        for fold in needed_folds:
            context = contexts[fold]
            if set(context['reference_orgs']) & set(context['scored_orgs']):
                raise ValueError('WARM_REFERENCE_INCLUDES_SCORED_ORGS')
            relative = 'references/' + context['reference_set_sha256'] + '.npz'
            path = root / relative
            if ref_ready['files'][relative] != digest(path):
                raise ValueError('WARM_REFERENCE_HASH_MISMATCH')
            with np.load(path, allow_pickle=False) as data:
                reference = {k: data[k] for k in ('w', 'b_fit', 'b_eval', 'direction_failures')}
                if not np.array_equal(data['item_ids'], global_ids) or not np.array_equal(data['source_half'], halves):
                    raise ValueError('WARM_REFERENCE_ITEM_AXIS_MISMATCH')
            references[fold] = reference
            reference_provenance[str(fold)] = dict(path=relative, sha256=ref_ready['files'][relative],
                                                  context=context['context'])
        selected_methods = method_selection(pr, readjson(root / 'selection_log.json'))
        estimators = {}
        for (fold, method, ep), choice in selected_methods.items():
            if fold not in needed_folds:
                continue
            stored = choice['estimator']
            key = stored['path']
            if key in estimators:
                if estimators[key]['sha256'] != stored['sha256']:
                    raise ValueError('CONFLICTING_ESTIMATOR_HASH')
                continue
            path = (root / key).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != stored['sha256']:
                raise ValueError('WARM_ESTIMATOR_PATH_OR_HASH')
            estimators[key] = dict(saved=joblib.load(path), sha256=stored['sha256'])
    setup_wall, setup_cpu = _duration(setup_start)
    dump(outdir / 'warm_schedule.json', dict(selected_model_ids=selected, methods=methods,
         seed=20260914, repetitions=3, rows=schedule, selection='SHA256(20260914|model_id)',
         shared_reference_setup=True, shared_estimator_setup=True,
         feature_cache_across_methods=False, raw_reads_per_model_per_method_per_repetition=1,
         raw_and_features_shared_only_between_two_endpoints=True))
    measurements = []
    max_error = 0.
    with measurement_path.open('x') as stream, timer(root, 'warm_measurements', n_models=len(selected), n_methods=33):
        for scheduled in schedule:
            if time.perf_counter() - node_start > 43200:
                raise RuntimeError('H-BUDGET: warm node exceeded 12h; measurements preserved')
            mid, method = scheduled['model_id'], scheduled['method']
            predrow, rec = by_prediction[mid], raw_records[mid]
            fold = int(predrow['outer_fold'])
            if rec['org_id'] != predrow['org_id'] or rec['org_id'] not in contexts[fold]['scored_orgs']:
                raise ValueError('WARM_MODEL_OUTER_CONTEXT_MISMATCH')
            whole_start = (time.perf_counter(), time.process_time())
            tick = (time.perf_counter(), time.process_time())
            raw = _load_raw(rec)
            io_wall, io_cpu = _duration(tick)
            tick = (time.perf_counter(), time.process_time())
            features, _ = extract_all(raw, global_ids, halves, references[fold])
            extract_wall, extract_cpu = _duration(tick)
            row = dict(**scheduled, outer_fold=fold, io_wall_seconds=io_wall,
                       io_cpu_seconds=io_cpu, extract_project_wall_seconds=extract_wall,
                       extract_project_cpu_seconds=extract_cpu, extra_LLM_calls=0)
            for ep in pr['endpoints']:
                chosen = selected_methods[fold, method, ep]
                tick = (time.perf_counter(), time.process_time())
                vector = np.array([[features[name] for name in pr['feature_groups'][chosen['package']]]], dtype=float)
                prediction = float(predict_package(estimators[chosen['estimator']['path']]['saved'], vector)[0])
                predict_wall, predict_cpu = _duration(tick)
                expected = float(predrow[f'{method}_best_{ep}'])
                error = abs(prediction - expected)
                row.update({ep + '_package': chosen['package'], ep + '_prediction': prediction,
                            ep + '_oof_abs_error': error, ep + '_predict_wall_seconds': predict_wall,
                            ep + '_predict_cpu_seconds': predict_cpu})
                max_error = max(max_error, error)
                if not np.isfinite(prediction) or error > 1e-10:
                    finite_row = {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                                  for key, value in row.items()}
                    dump(outdir / 'warm_prediction_mismatch.json', dict(**finite_row, endpoint=ep,
                         expected_prediction=expected, verdict='FAIL', target_labels_read=False))
                    raise AssertionError('WARM_REEXTRACTION_DIFFERS_FROM_SEALED_OOF')
            row['total_wall_seconds'], row['total_cpu_seconds'] = _duration(whole_start)
            row['extract_project_predict_wall_seconds'] = extract_wall + sum(row[ep + '_predict_wall_seconds'] for ep in pr['endpoints'])
            row['extract_project_predict_cpu_seconds'] = extract_cpu + sum(row[ep + '_predict_cpu_seconds'] for ep in pr['endpoints'])
            measurements.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            if len(measurements) % 33 == 0:
                stream.flush()
        stream.flush()
    expected_count = 3 * len(selected) * 33
    if len(measurements) != expected_count:
        raise AssertionError('WARM_MEASUREMENT_COUNT')
    timing_fields = ('total_wall_seconds', 'total_cpu_seconds', 'io_wall_seconds',
                     'io_cpu_seconds', 'extract_project_wall_seconds', 'extract_project_cpu_seconds',
                     'extract_project_predict_wall_seconds', 'extract_project_predict_cpu_seconds')
    per_model = []
    grouped = {}
    for row in measurements:
        grouped.setdefault((row['model_id'], row['method']), []).append(row)
    for (mid, method), group in sorted(grouped.items()):
        if len(group) != 3 or {row['repetition'] for row in group} != {0, 1, 2}:
            raise AssertionError('WARM_REPETITION_COUNT')
        summary = dict(model_id=mid, method=method, repetitions=3)
        for field in timing_fields:
            q = np.quantile([row[field] for row in group], [.25, .5, .75], method='linear')
            summary.update({field + '_q25': float(q[0]), field + '_median': float(q[1]), field + '_q75': float(q[2])})
        per_model.append(summary)
    write_rows(outdir / 'warm_per_model.csv', per_model)
    method_summaries = {}
    for method in methods:
        group = [row for row in per_model if row['method'] == method]
        method_summaries[method] = {field: dict(zip(('q25', 'median', 'q75'), map(float,
            np.quantile([row[field + '_median'] for row in group], [.25, .5, .75], method='linear')))) for field in timing_fields}
    summary = dict(status='COMPLETE_PENDING_V_RESULT', n_models=len(selected), n_methods=33,
                   repetitions=3, n_measurements=len(measurements), endpoints=pr['endpoints'],
                   setup_wall_seconds=setup_wall, setup_cpu_seconds=setup_cpu,
                   reference_provenance=reference_provenance, n_preloaded_estimators=len(estimators),
                   methods=method_summaries, max_oof_prediction_abs_error=max_error,
                   identity=dict(code_release_sha256=digest(root / 'CODE_RELEASE.json'),
                                 seal_sha256=digest(root / 'PREDICTION_SEAL.json'),
                                 scoring_release_sha256=digest(root / 'SCORING_RELEASE.json')),
                   files={str(p.relative_to(root)): digest(p) for p in
                          (measurement_path, outdir / 'warm_schedule.json', outdir / 'warm_per_model.csv')},
                   cold_cost='not measured by warm procedure', target_labels_read=False,
                   cache_conditions='reference and estimator cached; fresh raw ARC read and all20/M extraction for every method; the two endpoints share that one extraction',
                   cpu_threads=1, extra_LLM_calls=0, node_wall_seconds=time.perf_counter() - node_start)
    dump(outdir / 'warm_summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('run',))
    parser.parse_args()
    result = run()
    print(json.dumps({k: result[k] for k in ('status', 'n_measurements', 'max_oof_prediction_abs_error')}))
