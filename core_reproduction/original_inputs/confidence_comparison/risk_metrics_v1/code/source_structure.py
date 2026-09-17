"""Source-only reference/C/D production stages and a fixed source resource pilot."""
from common import (ROOT, digest, dump, readjson, protocol, require_data,
                    require_design, timer, load_source, event_list)
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from legacy_primitives import set_sha
from metrics_structure import (METRIC_IDS, reference_ease, fit_reference_direction,
                               compute_structure_one)


def ahash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.%s' % os.getpid())
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


class SourceReferences:
    def __init__(self, data, root=ROOT):
        self.root, self.data = Path(root), data
        self.old = Path(protocol(root)['legacy_root'])
        self.orgs = data['org_ids'].astype(str)
        self.mids = data['model_ids'].astype(str)
        self.source_sha = readjson(root / 'cohort_contract.json')['legacy']['source_arrays_sha256']
        self.code_sha = {n: digest(root / 'code' / n) for n in
                         ('metrics_structure.py', 'source_structure.py', 'legacy_primitives.py')}
        self.old_shared = dict(source_sha256=self.source_sha,
                               core_sha256=digest(self.old / 'code/legacy_primitives.py'),
                               A1_bundle_sha256=digest(self.old / 'A1_OUTPUT_HASHES.json'),
                               feature_contract_sha256=digest(self.old / 'feature_contract.json'))
        # Reuse uses exactly the old primitive; modifying it invalidates reuse.
        assert self.code_sha['legacy_primitives.py'] == self.old_shared['core_sha256']

    def identity(self, reference_orgs):
        ref = sorted(set(map(str, reference_orgs)))
        if not ref or not set(ref) <= set(self.orgs):
            raise ValueError('REFERENCE_ORGS_MISSING')
        ix = np.flatnonzero(np.isin(self.orgs, ref))
        mask = self.data['valid'][ix] & np.isfinite(self.data['ell'][ix])
        return dict(reference_orgs=ref, reference_model_ids=self.mids[ix].tolist(),
                    source_sha256=self.source_sha, source_half_sha256=ahash(self.data['source_half']),
                    ordered_item_ids_sha256=ahash(self.data['item_ids']),
                    effective_mask_sha256=ahash(mask), code_sha256=self.code_sha,
                    protocol_sha256=digest(self.root / 'protocol.json')), ix

    def legacy_b(self, ref, ix):
        key = set_sha(ref)
        path = self.old / 'features/reference_cache' / (key + '.npz')
        jp = path.with_suffix('.json')
        meta = readjson(jp)
        half = self.data['source_half'] == 0
        mask = self.data['valid'][ix][:, half] & np.isfinite(self.data['ell'][ix][:, half])
        ids = self.data['item_ids'][half].astype(str)
        expected = dict(reference_orgs=sorted(ref), reference_model_ids=self.mids[ix].tolist(),
                        **self.old_shared,
                        ordered_F_item_ids_sha256=hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest(),
                        effective_mask_sha256=ahash(mask),
                        qualified_item_mask_sha256=ahash(mask.sum(axis=0) >= 20))
        if meta['identity'] != expected or meta['b_file_sha256'] != digest(path):
            raise ValueError('LEGACY_REFERENCE_IDENTITY_MISMATCH')
        with np.load(path, allow_pickle=False) as z:
            if z.files != ['b']:
                raise ValueError('LEGACY_REFERENCE_ARRAY_SCHEMA')
            b = z['b'].copy()
        return b, dict(path=str(path), sha256=digest(path), metadata_sha256=digest(jp),
                       diagnostics=meta['diagnostics'])

    def get(self, context, allow_compute=True):
        score, ref = set(context['scored_orgs']), set(context['reference_orgs'])
        if score & ref:
            raise ValueError('REFERENCE_SCORED_ORG_OVERLAP')
        identity, ix = self.identity(ref)
        key = set_sha(ref)
        if key != context['reference_set_sha256']:
            raise ValueError('REFERENCE_SET_HASH_MISMATCH')
        path = self.root / 'references' / (key + '.npz')
        jp = path.with_suffix('.json')
        if path.exists() or jp.exists():
            if not path.exists() or not jp.exists():
                raise RuntimeError('INCOMPLETE_REFERENCE_CACHE_PAIR')
            meta = readjson(jp)
            if meta['identity'] != identity or meta['npz_sha256'] != digest(path):
                raise RuntimeError('REFERENCE_CACHE_IDENTITY_MISMATCH')
            with timer(self.root, 'structure_reference_cache_read', reference_sha256=key, reused=True):
                with np.load(path, allow_pickle=False) as z:
                    out = {k: z[k] for k in z.files}
            return out, meta
        if not allow_compute:
            raise RuntimeError('REFERENCE_CACHE_NOT_READY')
        with timer(self.root, 'structure_reference_ease', reference_sha256=key,
                   code_sha256=self.code_sha,
                   n_reference_models=len(ix), source_labels=int(self.data['valid'][ix].sum()), extra_LLM_calls=0):
            ease = reference_ease(self.data['y'], self.data['valid'], self.orgs, ref, score)
        outputs, diagnostics, reuse = [], [], None
        for h in (0, 1):
            with timer(self.root, 'structure_reference_direction', reference_sha256=key,
                       code_sha256=self.code_sha,
                       fit_half=h, reused_b=(h == 0), new_ALS=(h == 1),
                       n_reference_models=len(ix), extra_LLM_calls=0) as event:
                try:
                    if h == 0:
                        b, reuse = self.legacy_b(ref, ix)
                    else:
                        b = None
                    out = fit_reference_direction(self.data['ell'][ix], self.data['valid'][ix],
                                                  self.data['source_half'], h, cached_b=b)
                    out['failure'] = ''
                    event['status'] = 'OK'
                except ValueError as exc:
                    # Identity/shape defects are implementation/input failures, not
                    # scientific missingness. Only frozen ALS degeneracies qualify.
                    reason = str(exc)
                    recognized = ('INSUFFICIENT_REFERENCE_ITEMS', 'REFERENCE_MODEL_WITH_LT2_QUALIFIED_ITEMS',
                                  'DEGENERATE_REFERENCE_', 'ALS_NONCONVERGENCE', 'NONFINITE_REFERENCE_PARAMETERS')
                    if not any(reason.startswith(prefix) for prefix in recognized):
                        raise
                    nitems = len(self.data['item_ids'])
                    neval = int(np.sum(self.data['source_half'] == 1 - h))
                    out = dict(b_fit=np.full(nitems, np.nan), b_eval=np.full(nitems, np.nan),
                               a_reference=np.full(len(ix), np.nan), s_reference=np.full(len(ix), np.nan),
                               eval_n_reference=np.zeros(neval, np.int64),
                               eval_denominator=np.full(neval, np.nan),
                               eval_reasons=np.full(neval, reason, dtype='U160'),
                               diagnostics=dict(failure=reason), failure=reason)
                    event.update(status='DIAGNOSTIC_FAILURE', reason=reason)
                outputs.append(out)
                diagnostics.append(out['diagnostics'])
        arrays = dict(w=ease['w'], n_reference=ease['n_reference'], n_reference_orgs=ease['n_reference_orgs'],
                      item_ids=self.data['item_ids'], source_half=self.data['source_half'],
                      reference_model_ids=self.mids[ix], reference_org_ids=self.orgs[ix],
                      b_fit=np.array([o['b_fit'] for o in outputs]), b_eval=np.array([o['b_eval'] for o in outputs]),
                      a_reference=np.array([o['a_reference'] for o in outputs]),
                      s_reference=np.array([o['s_reference'] for o in outputs]),
                      eval_n_reference=np.array([o['eval_n_reference'] for o in outputs]),
                      eval_denominator=np.array([o['eval_denominator'] for o in outputs]),
                      eval_reasons=np.array([o['eval_reasons'] for o in outputs]),
                      direction_failures=np.array([o['failure'] for o in outputs], dtype='U160'))
        with timer(self.root, 'structure_reference_write', reference_sha256=key) as event:
            save_npz(path, **arrays)
            meta = dict(identity=identity, npz_sha256=digest(path), diagnostics=diagnostics,
                        legacy_half0_reuse=reuse, new_ALS_attempts=1, extra_LLM_calls=0,
                        source_only=True, created_by='source metrics')
            dump(jp, meta)
            event['new_disk_bytes'] = path.stat().st_size + jp.stat().st_size
        return arrays, meta


def source_record(data, i, z):
    return dict(ell=data['ell'][i], c=data['c'][i], y=data['y'][i], z=z,
                valid=data['valid'][i], item_ids=data['item_ids'], source_half=data['source_half'])


def all_contexts(root):
    contexts = readjson(root / 'split_manifest.json')['projection_contexts']
    assert len(contexts) == 125 and len({c['reference_set_sha256'] for c in contexts}) == 55
    return contexts


def pilot(root=ROOT):
    require_data(root)
    with timer(root, 'structure_pilot_source_io'):
        data, _ = load_source(root)
    refs, contexts = SourceReferences(data, root), all_contexts(root)
    counts = {c['reference_set_sha256']: int(np.isin(data['org_ids'], c['reference_orgs']).sum()) for c in contexts}
    selected = sorted(contexts, key=lambda c: (-counts[c['reference_set_sha256']], c['reference_set_sha256'], c['context']))[0]
    key = selected['reference_set_sha256']
    pilot_path = root / 'source_structure_pilot.json'
    identity = dict(reference_key=key, reference_identity=refs.identity(selected['reference_orgs'])[0],
                    E1_bundle_sha256=digest(root / 'E1_BUNDLE.json'))
    if pilot_path.exists():
        report = readjson(pilot_path)
        if report['identity'] != identity:
            raise RuntimeError('PILOT_IDENTITY_CHANGED')
        if report['budget_gate']:
            raise RuntimeError('H-BUDGET: structure estimate exceeds 12h')
        return report
    reference, _ = refs.get(selected)
    # First required fit cost remains traceable even after interruption/cache load.
    events = [e for e in event_list(root) if e['stage'] == 'structure_reference_direction'
              and e['reference_sha256'] == key and e.get('code_sha256') == refs.code_sha]
    cold = [e for e in events if e['fit_half'] == 1]
    if len(cold) != 1:
        raise RuntimeError('AMBIGUOUS_OR_MISSING_ORIGINAL_STRUCTURE_PILOT')
    score = np.flatnonzero(np.isin(data['org_ids'], selected['scored_orgs']))[:100]
    start, cpu = time.perf_counter(), time.process_time()
    # z=0 is a timing-only proxy; no candidate values are saved or released.
    with timer(root, 'structure_pilot_projection', n_models=len(score), z_proxy='zeros_timing_only',
               source_labels=int(data['valid'][score].sum()), extra_LLM_calls=0):
        for i in score:
            compute_structure_one(source_record(data, i, np.zeros(len(data['item_ids']))), reference)
    proj_seconds, proj_cpu = time.perf_counter() - start, time.process_time() - cpu
    scored_rows = sum(int(np.isin(data['org_ids'], c['scored_orgs']).sum()) for c in contexts)
    per_ref = sum(e['wall_seconds'] for e in events)
    ease_events = [e for e in event_list(root) if e['stage'] == 'structure_reference_ease'
                   and e['reference_sha256'] == key and e.get('code_sha256') == refs.code_sha]
    per_ref += sum(e['wall_seconds'] for e in ease_events)
    # Factor two and explicit I/O allowance; no claim this is cold legacy ALS.
    estimate = 2 * (55 * per_ref + scored_rows * proj_seconds / len(score)) + 300
    report = dict(identity=identity, first_required_half1_fit=cold[0],
                  measured_reference_components_seconds=per_ref, projection_sample_n=len(score),
                  projection_sample_wall_seconds=proj_seconds, projection_sample_cpu_seconds=proj_cpu,
                  projection_z='zero proxy for timing only; no scientific candidate extraction',
                  total_scored_context_rows=scored_rows, n_unique_references=55,
                  total_new_half1_ALS=55, total_reused_half0_b=55,
                  conservative_structure_seconds=estimate, budget_gate=estimate > 43200,
                  historical_half0_full_cold_cost='separate legacy ledger; not measured by cache access')
    dump(pilot_path, report)
    if report['budget_gate']:
        raise RuntimeError('H-BUDGET: structure estimate exceeds 12h')
    return report


def references(root=ROOT):
    require_design(root)
    report = readjson(root / 'source_structure_pilot.json')
    if report['budget_gate']:
        raise RuntimeError('H-BUDGET: structure estimate exceeds 12h')
    with timer(root, 'structure_reference_source_io'):
        data, _ = load_source(root)
    refs, contexts = SourceReferences(data, root), all_contexts(root)
    if report['identity']['reference_identity'] != refs.identity(report['identity']['reference_identity']['reference_orgs'])[0]:
        raise RuntimeError('PILOT_IDENTITY_CHANGED')
    distinct = {c['reference_set_sha256']: c for c in contexts}
    files, failures = {}, {}
    start = time.perf_counter()
    for key in sorted(distinct):
        if time.perf_counter() - start > 43200:
            raise RuntimeError('H-BUDGET: reference node runtime exceeds 12h')
        arr, meta = refs.get(distinct[key])
        for ext in ('.npz', '.json'):
            p = root / 'references' / (key + ext)
            files[str(p.relative_to(root))] = digest(p)
        failures[key] = arr['direction_failures'].tolist()
        print(json.dumps(dict(stage='reference', key=key, failures=failures[key])), flush=True)
    dump(root / 'references/REFERENCES_READY.json', dict(files=files, failures=failures,
         n_references=len(distinct), code_sha256=refs.code_sha, source_only=True))


def contexts(root=ROOT):
    require_design(root)
    ready = readjson(root / 'references/REFERENCES_READY.json')
    for name, sha in ready['files'].items():
        assert digest(root / name) == sha, name
    with timer(root, 'structure_context_source_io'):
        data, _ = load_source(root)
        zpath = root / 'features/ab_values.npz'
        z_sha = digest(zpath)
        ab_manifest = readjson(root / 'features/ab_manifest.json')
        assert ab_manifest['file'] == str(zpath.relative_to(root))
        assert ab_manifest['sha256'] == z_sha
        with np.load(zpath, allow_pickle=False) as zdata:
            assert np.array_equal(zdata['model_ids'], data['model_ids'])
            assert np.array_equal(zdata['item_ids'], data['item_ids'])
            z = zdata['z']
        assert z.shape == data['ell'].shape and np.isfinite(z[data['valid']]).all()
    refs = SourceReferences(data, root)
    files = {}
    start = time.perf_counter()
    for context in all_contexts(root):
        if time.perf_counter() - start > 43200:
            raise RuntimeError('H-BUDGET: contexts node runtime exceeds 12h')
        ref, meta = refs.get(context, allow_compute=False)
        ix = np.flatnonzero(np.isin(data['org_ids'], context['scored_orgs']))
        path = root / 'features/structure_contexts' / (context['context'].replace('/', '__') + '.npz')
        jp = path.with_suffix('.json')
        identity = dict(context=context['context'], reference_sha256=meta['npz_sha256'],
                        source_sha256=refs.source_sha, z_sha256=z_sha, code_sha256=refs.code_sha,
                        model_ids_sha256=ahash(data['model_ids'][ix]))
        if path.exists() or jp.exists():
            if not path.exists() or not jp.exists():
                raise RuntimeError('INCOMPLETE_CONTEXT_CACHE_PAIR')
            old = readjson(jp)
            if old['identity'] != identity or old['npz_sha256'] != digest(path):
                raise RuntimeError('CONTEXT_CACHE_IDENTITY_MISMATCH')
        else:
            with timer(root, 'structure_context_projection', context=context['context'], n_models=len(ix),
                       reference_sha256=context['reference_set_sha256'],
                       source_labels=int(data['valid'][ix].sum()), extra_LLM_calls=0) as event:
                results = [compute_structure_one(source_record(data, i, z[i]), ref) for i in ix]
                arrays = {key: np.array([r[key] for r in results]) for key in results[0]}
                arrays.update(model_ids=data['model_ids'][ix], org_ids=data['org_ids'][ix],
                              metric_ids=np.array(METRIC_IDS), context=np.array(context['context']),
                              reference_sha256=np.array(meta['npz_sha256']),
                              parameter_ids=np.array(['a', 's', 'sigma', 'theta', 'var_ell']))
                save_npz(path, **arrays)
                dump(jp, dict(identity=identity, npz_sha256=digest(path), n_models=len(ix),
                              row_policy='all Q0 scored models, original Q0 order; old qualification applied by assembler',
                              half_axis_semantics='C: source item half; D: fitted source half h, evaluated on 1-h',
                              finite_counts=np.isfinite(arrays['values']).sum(axis=0).tolist(), source_only=True))
                event['new_disk_bytes'] = path.stat().st_size + jp.stat().st_size
        for p in (path, jp):
            files[str(p.relative_to(root))] = digest(p)
        print(json.dumps(dict(stage='context', context=context['context'], n_models=len(ix))), flush=True)
    dump(root / 'features/STRUCTURE_READY.json', dict(files=files, n_contexts=125, metric_ids=list(METRIC_IDS),
         references_ready_sha256=digest(root / 'references/REFERENCES_READY.json'), z_sha256=z_sha,
         code_sha256=refs.code_sha, source_only=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pilot', 'references', 'contexts'])
    args = parser.parse_args()
    output = dict(pilot=pilot, references=references, contexts=contexts)[args.stage]()
    if output is not None:
        print(json.dumps({key: output[key] for key in ('projection_sample_n', 'n_unique_references',
              'conservative_structure_seconds', 'budget_gate')}), flush=True)
