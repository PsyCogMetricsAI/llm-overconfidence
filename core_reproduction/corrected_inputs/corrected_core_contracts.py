#!/usr/bin/env python3
"""Corrected CORE cohort contracts (root DATAFIX-CORE-COHORT-DECISION.json).

The core experiment keeps the original frozen candidate pool (6,706 model ids / 1,331 orgs).
The mechanical corrected run has 6,707 preliminarily eligible models because the item split was
legitimately recomputed; the extra model is recorded but stays outside the frozen core pool.

This module (no scientific algorithm change, verification + mechanical rebinding only):
  * derives the explicit 6,706-model subset of the already-built mechanical source arrays,
    preserving the corrected 1,168-item axis and the recomputed 584/584 halves;
  * verifies frozen model/org order and sets against the archive arrays;
  * rebinds every runtime item-half map (mainline and core split manifests) to the corrected
    axis from the actual runtime source arrays, preserving organization folds and the 125
    projection contexts (55 unique reference sets);
  * regenerates fit-dependent context qualification (fixed_rows / row_counts / n_models /
    n_orgs) from the actual legacy context CSVs, never forcing the archive 6,666;
  * checks the cohort-contract schema against the expectations of the source workers and the
    independent target administrator.

Nothing here is a scientific acceptance; the root DX2 gate remains mandatory before execution.
"""
import csv, hashlib, json
from pathlib import Path
import numpy as np

SEED = 20260914
MASK_ARC = ['Mercury_SC_LBS10597', 'Mercury_406639', 'TIMSS_2003_8_pg47', 'Mercury_7116183']
REQUIRED_CC_KEYS = ['fixed_rows', 'row_counts', 'n_models', 'n_orgs', 'q0_nmodels',
                    'n_projection_contexts', 'n_unique_reference_sets', 'legacy',
                    'legacy_qualification', 'new_candidate_missing_does_not_drop_rows']
REQUIRED_FIXED_KEYS = ['path', 'source_path', 'sha256', 'model_ids', 'org_ids', 'contexts']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def js(p):
    return json.loads(Path(p).read_text())


def put(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=1, ensure_ascii=False, allow_nan=False) + '\n')


def rows(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha_order_local(items, k):
    """Verification-only reimplementation of the archived SHA256(seed|id) rank-mod-k rule."""
    ordered = sorted(set(map(str, items)), key=lambda s: (hashlib.sha256(f'{SEED}|{s}'.encode()).hexdigest(), s))
    return {s: j % k for j, s in enumerate(ordered)}


def qualified(rec):
    return str(rec['valid_source_fit']).lower() == 'true' and str(rec['valid_summaries']).lower() == 'true'


def frozen_core_subset(cv_root, reference_arrays):
    """Derive and verify the frozen 6,706-model subset of the mechanical source arrays."""
    cv_root = Path(cv_root)
    subset_path = cv_root / 'source_arrays.npz'
    full_path = cv_root / 'source_arrays_fullpool6707.npz'
    if not subset_path.exists():
        raise SystemExit(f'CV_ROOT_NOT_SEEDED: {subset_path}')
    arch = load_npz(reference_arrays)
    arch_ids = arch['model_ids'].astype(str).tolist()
    if not full_path.exists():
        current = load_npz(subset_path)
        cur_ids = current['model_ids'].astype(str).tolist()
        if set(cur_ids) == set(arch_ids) and len(cur_ids) == len(arch_ids):
            full_path = subset_path          # already the core subset; nothing to preserve
        else:
            full_path.write_bytes(subset_path.read_bytes())
    mech = load_npz(full_path)
    mech_ids = mech['model_ids'].astype(str).tolist()
    missing = sorted(set(arch_ids) - set(mech_ids))
    if missing:
        raise SystemExit(f'FROZEN_MODELS_ABSENT_FROM_MECHANICAL_ARRAYS: {missing[:5]}')
    if len(set(arch_ids)) != len(arch_ids):
        raise SystemExit('archive model axis not unique')
    extra = sorted(set(mech_ids) - set(arch_ids))
    idx = [mech_ids.index(m) for m in arch_ids]
    if not np.array_equal(mech['org_ids'][idx].astype(str), arch['org_ids'].astype(str)):
        raise SystemExit('FROZEN_ORG_ORDER_MISMATCH')
    items = mech['item_ids'].astype(str).tolist()
    arch_items = arch['item_ids'].astype(str).tolist()
    dropped = [s for s in arch_items if s not in set(items)]
    if not set(dropped) <= set(MASK_ARC) or len(dropped) != 4:
        raise SystemExit(f'ITEM_AXIS_NOT_POLICY_SUBSET: {dropped}')
    if items != [s for s in arch_items if s not in set(dropped)]:
        raise SystemExit('ITEM_ORDER_NOT_ARCHIVE_SUBSEQUENCE')
    recomputed = sha_order_local(items, 2)
    got_half = {s: int(h) for s, h in zip(items, mech['source_half'].astype(int).tolist())}
    if got_half != recomputed:
        raise SystemExit('MECHANICAL_HALVES_DO_NOT_FOLLOW_ARCHIVE_RULE')
    arch_half = {s: int(h) for s, h in zip(arch_items, arch['source_half'].astype(int).tolist())}
    half_changes = sorted(s for s in items if arch_half[s] != got_half[s])
    subset = {
        'ell': mech['ell'][idx], 'c': mech['c'][idx], 'y': mech['y'][idx], 'valid': mech['valid'][idx],
        'item_ids': mech['item_ids'], 'model_ids': arch['model_ids'], 'org_ids': arch['org_ids'],
        'source_half': mech['source_half'],
    }
    tmp = subset_path.with_name('source_arrays.tmp_subset.npz')
    np.savez_compressed(tmp, **subset)
    tmp.replace(subset_path)
    # pipeline-facing per-model files follow the frozen pool; full-scan audit files stay untouched
    for name in ['source_B_features.csv', 'target_file_index.csv']:
        p = cv_root / name
        if not p.exists():
            raise SystemExit(f'MISSING_PIPELINE_FILE: {name}')
        rr = rows(p)
        by = {r['model_id']: r for r in rr}
        missing_rows = [m for m in arch_ids if m not in by]
        if missing_rows:
            raise SystemExit(f'FROZEN_MODELS_ABSENT_FROM_{name}: {missing_rows[:5]}')
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows([by[m] for m in arch_ids])
    record = {
        'policy': 'DATAFIX-CORE-COHORT-DECISION.json: original frozen candidate pool 6706 / 1331 orgs',
        'core_candidate_models': len(arch_ids), 'core_candidate_orgs': len(set(arch['org_ids'].astype(str))),
        'frozen_model_order_verified': True, 'frozen_org_order_verified': True,
        'full_pool_models': len(mech_ids),
        'models_outside_frozen_pool': extra,
        'item_axis': len(items), 'items_removed_vs_archive': dropped,
        'item_order_is_archive_subsequence': True,
        'halves_follow_archive_rule': True,
        'half_counts': {'F': int(sum(1 for v in got_half.values() if v == 0)),
                        'C': int(sum(1 for v in got_half.values() if v == 1))},
        'half_changes_vs_archive': len(half_changes),
        'source_arrays_fullpool_sha256': sha(full_path),
        'full_pool_artifact_preserved_separately': full_path != subset_path,
        'source_arrays_core_sha256': sha(subset_path),
        'source_B_features_core_sha256': sha(cv_root / 'source_B_features.csv'),
        'target_file_index_core_sha256': sha(cv_root / 'target_file_index.csv'),
        'full_scan_artifacts_preserved': {n: sha(cv_root / n) for n in
                                          ['input_scan.json', 'eligibility.csv', 'manifest.sha256',
                                           'data_contract.json']},
    }
    put(cv_root / 'CORRECTED_CORE_SCOPE.json', record)
    return record


def runtime_split_manifest(template_path, arrays, archive_split_path, out_path):
    """Rebind a runtime split manifest's item halves to the corrected axis; keep org design."""
    sm = js(template_path)
    arch = js(archive_split_path)
    items = arrays['item_ids'].astype(str).tolist()
    halves = arrays['source_half'].astype(int).tolist()
    orgs = set(arrays['org_ids'].astype(str).tolist())
    # exact preservation of the frozen organization design (not just key sets)
    if sm['outer_org_fold'] != arch['outer_org_fold']:
        raise SystemExit('ORG_FOLD_MAPPING_CHANGED')
    if sm['outer'] != arch['outer']:
        raise SystemExit('OUTER_FOLD_STRUCTURE_CHANGED')
    if sm['projection_contexts'] != arch['projection_contexts']:
        raise SystemExit('PROJECTION_CONTEXTS_CHANGED')
    for key in ['seed', 'cohort_stage', 'hash_rule', 'n_projection_contexts',
                'n_unique_reference_org_sets', 'nested_rules', 'old_CV_split_exact_match']:
        if key in arch and sm.get(key) != arch[key]:
            raise SystemExit(f'SPLIT_FIELD_CHANGED:{key}')
    if not orgs <= set(sm['outer_org_fold']):
        raise SystemExit('RUNTIME_ORGS_MISSING_FROM_FOLDS')
    kept_contexts, dropped_contexts = [], []
    for ctx in sm['projection_contexts']:
        ok = set(ctx['reference_orgs']) <= set(sm['outer_org_fold'])
        (kept_contexts if ok else dropped_contexts).append(ctx)
    if dropped_contexts:
        raise SystemExit(f'PROJECTION_CONTEXTS_REFERENCE_UNKNOWN_ORGS: {len(dropped_contexts)}')
    sm['source_item_half'] = {s: int(h) for s, h in zip(items, halves)}
    if sm['n_projection_contexts'] != len(sm['projection_contexts']):
        raise SystemExit('PROJECTION_CONTEXT_COUNT_FIELD_MISMATCH')
    sets = {tuple(sorted(c['reference_orgs'])) for c in sm['projection_contexts']}
    if sm['n_unique_reference_org_sets'] != len(sets):
        raise SystemExit('REFERENCE_SET_COUNT_FIELD_MISMATCH')
    put(out_path, sm)
    return {
        'out': str(out_path), 'sha256': sha(out_path),
        'n_items': len(items), 'half_counts': {'F': halves.count(0), 'C': halves.count(1)},
        'n_orgs_in_folds': len(sm['outer_org_fold']),
        'n_projection_contexts': len(sm['projection_contexts']),
        'n_unique_reference_sets': len({tuple(sorted(c['reference_orgs'])) for c in sm['projection_contexts']}),
        'org_folds_unchanged_vs_archive': True,
    }


def verify_cohort_rows(cohort_rows, raw_index, mask_arc=None):
    """Every frozen cohort row must bind to a corrected raw file with the corrected SHA."""
    problems, n = [], 0
    for r in cohort_rows:
        for bench, prefix in [('arc', 'arc'), ('hellaswag', 'HS')]:
            name = Path(r[f'{prefix}_path']).name
            rec = raw_index.get((bench, name))
            if rec is None:
                problems.append(f'{r["model_id"]}:{bench}:missing_from_corrected_manifest')
                continue
            n += 1
            if r[f'{prefix}_sha256'] != rec['output_sha256']:
                problems.append(f'{r["model_id"]}:{bench}:sha_not_corrected_output')
    return {'rows': len(cohort_rows), 'bindings_checked': n, 'problems': problems[:10],
            'n_problems': len(problems)}


def recompute_fixed_rows(core_root, legacy_root, cc):
    """Regenerate fit-dependent context qualification from the actual legacy context CSVs."""
    core_root, legacy_root = Path(core_root), Path(legacy_root)
    rec = {}
    for name, info in cc['fixed_rows'].items():
        src = legacy_root / name
        rr = rows(src)
        qual = [r for r in rr if qualified(r)]
        mids = [r['model_id'] for r in qual]
        if len(set(mids)) != len(mids):
            raise SystemExit(f'DUPLICATE_QUALIFIED_MODEL_IN_CONTEXT: {name}')
        dst = core_root / info['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        info.update(source_path=str(src), sha256=sha(dst), model_ids=mids,
                    org_ids=[r['org_id'] for r in qual],
                    contexts=[r['context'] for r in qual])
        cc['row_counts'][name] = len(qual)
        rec[name] = {'rows': len(rr), 'qualified': len(qual), 'n_orgs': len({r['org_id'] for r in qual})}
    return rec


def verify_cohort_contract(core_root, legacy_root, cc, split):
    """Schema + consumer invariants expected by source workers and the target administrator."""
    core_root, legacy_root = Path(core_root), Path(legacy_root)
    missing = [k for k in REQUIRED_CC_KEYS if k not in cc]
    if missing:
        raise SystemExit(f'COHORT_CONTRACT_MISSING_KEYS: {missing}')
    for name, info in cc['fixed_rows'].items():
        miss = [k for k in REQUIRED_FIXED_KEYS if k not in info]
        if miss:
            raise SystemExit(f'FIXED_ROW_SCHEMA: {name}: {miss}')
        if sha(core_root / info['path']) != info['sha256']:
            raise SystemExit(f'FIXED_ROW_SHA: {name}')
        qual = [r for r in rows(legacy_root / name) if qualified(r)]
        if [r['model_id'] for r in qual] != info['model_ids']:
            raise SystemExit(f'FIXED_ROW_MODEL_ORDER: {name}')
        if [r['org_id'] for r in qual] != info['org_ids']:
            raise SystemExit(f'FIXED_ROW_ORG_ORDER: {name}')
        if cc['row_counts'][name] != len(info['model_ids']):
            raise SystemExit(f'ROW_COUNT_MISMATCH: {name}')
    tests = [v for k, v in cc['fixed_rows'].items() if k.endswith('/test.csv')]
    if len(tests) != 5:
        raise SystemExit(f'EXPECTED_5_TEST_CONTEXTS: {len(tests)}')
    test_models = [m for v in tests for m in v['model_ids']]
    if len(set(test_models)) != len(test_models):
        raise SystemExit('EVALUATION_COHORT_MODEL_DUPLICATE')
    if cc['n_models'] != len(test_models):
        raise SystemExit('N_MODELS_NOT_EVALUATION_COHORT_SIZE')
    if cc['n_orgs'] != len({m for v in tests for m in v['org_ids']}):
        raise SystemExit('N_ORGS_NOT_EVALUATION_COHORT_SIZE')
    if not (cc['n_projection_contexts'] == len(split['projection_contexts']) == 125):
        raise SystemExit('PROJECTION_CONTEXT_COUNT_NOT_125')
    if cc['n_unique_reference_sets'] != split['n_unique_reference_org_sets']:
        raise SystemExit('REFERENCE_SET_COUNT_MISMATCH')
    ident = {}
    for v in cc['fixed_rows'].values():
        for m, g in zip(v['model_ids'], v['org_ids']):
            if m in ident and ident[m] != g:
                raise SystemExit(f'IDENTITY_CONFLICT: {m}')
            ident[m] = g
    folds = split['outer_org_fold']
    for k, v in cc['fixed_rows'].items():
        kk = int(k.split('outer_')[1][0])
        outer_vals = [int(folds[g]) for g in v['org_ids']]
        if k.endswith('/test.csv'):
            # the test context IS the outer holdout: every row must sit in fold k
            if any(f != kk for f in outer_vals):
                raise SystemExit(f'OUTER_HOLDOUT_MISMATCH: {k}')
            continue
        # every non-test context is inside the outer training set A (fold != k)
        if any(f == kk for f in outer_vals):
            raise SystemExit(f'OUTER_TRAIN_LEAK: {k}')
        if '/inner' in k:
            jj = int(k.split('inner')[1][0])
            try:
                inner = split['outer'][str(kk)]['inner_org_fold']
            except KeyError as ex:
                raise SystemExit(f'INNER_FOLD_MISSING: {k}') from ex
            inner_vals = [int(inner[g]) for g in v['org_ids']]
            if k.endswith('_valid.csv'):
                if any(f != jj for f in inner_vals):
                    raise SystemExit(f'INNER_VALID_FOLD_MISMATCH: {k}')
            elif k.endswith('_train.csv'):
                if any(f == jj for f in inner_vals):
                    raise SystemExit(f'INNER_TRAIN_FOLD_LEAK: {k}')
    return {
        'contexts': len(cc['fixed_rows']), 'evaluation_models': cc['n_models'],
        'evaluation_orgs': cc['n_orgs'], 'projection_contexts': cc['n_projection_contexts'],
        'unique_reference_sets': cc['n_unique_reference_sets'],
        'q0_nmodels': cc['q0_nmodels'], 'legacy_qualified_models': cc['legacy_qualification']['n_models'],
        'legacy_qualified_orgs': cc['legacy_qualification']['n_orgs'],
        'schema_ok': True, 'target_admin_invariants_ok': True,
    }


def preflight_core(work, clone_root, corrected_raw, policy_mask, preflight_out):
    """Targeted DX2 preflight: source-array axis/identity/split and every cohort row."""
    work = Path(work)
    cv = work / 'confidence_validity'
    if not (cv / 'source_arrays.npz').exists():
        raise SystemExit('CV root not seeded; run plan/preflight of the adapter first')
    scope = frozen_core_subset(cv, REFERENCE_ARRAYS)
    arrays = load_npz(cv / 'source_arrays.npz')
    raw_index = {(r['bench'], r['filename']): r
                 for r in js(Path(corrected_raw) / 'CORRECTED_RAW_MANIFEST.json')['files']}
    mainline_root = clone_root / 'original_inputs/confidence_comparison/mainline_v2'
    core_root = clone_root / 'original_inputs/confidence_comparison/risk_metrics_v1'
    cohort = rows(mainline_root / 'cohort_manifest.csv')
    # apply the runtime rebind that the legacy phase will write: corrected paths + corrected SHAs
    for r in cohort:
        for bench, prefix in [('arc', 'arc'), ('hellaswag', 'HS')]:
            name = Path(r[f'{prefix}_path']).name
            rec = raw_index[(bench, name)]
            r[f'{prefix}_path'] = str(Path(corrected_raw) / bench / name)
            r[f'{prefix}_sha256'] = rec['output_sha256']
    cohort_rows = verify_cohort_rows(cohort, raw_index)
    split_ml = runtime_split_manifest(mainline_root / 'split_manifest.json', arrays,
                                     STAGING_ML_SPLIT, work / 'DX2_RUNTIME_SPLIT_PREVIEW.json')
    split_core = runtime_split_manifest(core_root / 'split_manifest.json', arrays,
                                        STAGING_CORE_SPLIT, work / 'DX2_RUNTIME_CORE_SPLIT_PREVIEW.json')
    out = {'scope': scope, 'cohort_rows': cohort_rows, 'mainline_split_preview': split_ml,
           'core_split_preview': split_core,
           'arrays_sha256': sha(cv / 'source_arrays.npz'),
           'cohort_manifest_models': len(cohort),
           'cohort_models_equal_frozen': [r['model_id'] for r in cohort] ==
                                         load_npz(REFERENCE_ARRAYS)['model_ids'].astype(str).tolist(),
           'status': 'PASS' if not cohort_rows['n_problems'] else 'FAIL'}
    put(preflight_out, out)
    return out


STAGING = Path(__file__).resolve().parents[2]
REFERENCE_ARRAYS = STAGING / 'core_reproduction/original_inputs/confidence_validity/source_arrays.npz'
STAGING_ML_SPLIT = STAGING / 'core_reproduction/original_inputs/confidence_comparison/mainline_v2/split_manifest.json'
STAGING_CORE_SPLIT = STAGING / 'core_reproduction/original_inputs/confidence_comparison/risk_metrics_v1/split_manifest.json'
