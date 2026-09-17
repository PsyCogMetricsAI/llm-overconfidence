"""A3 source summaries, finite nested references and context-local qualification."""
from common import *
from legacy_primitives import observations,validity,als_reference,project_source,set_sha
from scipy.special import logsumexp

def summaries(loglik,char_lens,gold,item_ids,n_options=None,extended=True):
    start=time.perf_counter();cpu=time.process_time()
    e,c,y,valid,*_=observations(loglik,char_lens,gold,n_options)
    ids=np.asarray(item_ids).astype(str)
    if len(set(ids))!=len(ids):raise ValueError('duplicate item ID')
    r=np.flatnonzero(valid);n=len(r)
    if n==0:raise ValueError('no valid source items')
    cc=c[r];yy=y[r].astype(float);bb=np.minimum(np.floor(10*cc).astype(int),9)
    rel=ece=0.
    for b in range(10):
        take=bb==b
        if take.any():
            gap=float(cc[take].mean()-yy[take].mean());rel+=take.mean()*gap**2;ece+=take.mean()*abs(gap)
    order=np.lexsort((ids[r],-cc));sel=order[:(n+1)//2]
    out=dict(A=float(yy.mean()),C=float(cc.mean()),O=float(cc.mean()-yy.mean()),REL10=float(rel),R50=float((1-yy[sel]).mean()))
    base_wall=time.perf_counter()-start;base_cpu=time.process_time()-cpu
    if extended:
        ll=np.asarray(loglik,float)[r];hh=np.asarray(char_lens,float)[r];real=np.isfinite(hh)&(hh>0)
        u=np.full(ll.shape,-np.inf);np.divide(ll,hh,out=u,where=real)
        p=np.exp(u-logsumexp(u,axis=1,keepdims=True));gg=np.asarray(gold)[r].astype(int)
        truth=np.zeros_like(p);truth[np.arange(n),gg]=1
        lp=np.zeros_like(p);np.log(p,out=lp,where=p>0)
        pp=np.sort(p,axis=1)
        out.update(ECE10=float(ece),Brier=float(np.mean(np.sum((p-truth)**2,axis=1))),
          H=float(np.mean(-np.sum(p*lp,axis=1)/np.log(real.sum(1)))),Margin=float(np.mean(pp[:,-1]-pp[:,-2])))
    out.update(n_valid_source=n,valid_summaries=bool(np.isfinite(list(out.values())).all()))
    out.update(base_component_wall_seconds=base_wall,base_component_cpu_seconds=base_cpu)
    return out,(e,c,y,valid)

def read_arc(rec):
    path=Path(rec['arc_path'])
    assert path.parent.name=='arc' and digest(path)==rec['arc_sha256']
    with np.load(path,allow_pickle=False) as z:
        raw={k:z[k] for k in ['loglik','char_lens','gold','item_ids','n_options']}
    assert digest(path)==rec['arc_sha256']
    return raw

class References:
    def __init__(self,data,root):
        self.root=root;self.data=data;self.orgs=data['org_ids'].astype(str);self.mids=data['model_ids'].astype(str)
        f=data['source_half']==0;self.ell=data['ell'][:,f];self.valid=data['valid'][:,f];self.items=data['item_ids'][f].astype(str)
        self.cache={};self.meta={};self.path=root/'features/reference_cache';self.path.mkdir(parents=True,exist_ok=True)
    def reference(self,reforgs):
        ref=sorted(set(reforgs));key=set_sha(ref)
        if key in self.cache:return key,self.cache[key]
        take=np.isin(self.orgs,ref);m=self.valid[take]&np.isfinite(self.ell[take]);eligible=m.sum(0)>=20
        arrayhash=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        stamp=dict(reference_orgs=ref,reference_model_ids=self.mids[take].tolist(),source_sha256=readjson(self.root/'cohort_contract.json')['source_arrays_sha256'],
          core_sha256=digest(self.root/'code/legacy_primitives.py'),A1_bundle_sha256=digest(self.root/'A1_OUTPUT_HASHES.json'),
          feature_contract_sha256=digest(self.root/'feature_contract.json'),ordered_F_item_ids_sha256=hashlib.sha256(('\n'.join(self.items)+'\n').encode()).hexdigest(),
          effective_mask_sha256=arrayhash(m),qualified_item_mask_sha256=arrayhash(eligible))
        p=self.path/f'{key}.npz';j=self.path/f'{key}.json'
        if p.exists() or j.exists():
            assert p.exists() and j.exists(),'incomplete reference cache pair'
            meta=readjson(j);assert meta['identity']==stamp and meta['b_file_sha256']==digest(p)
            b=np.load(p,allow_pickle=False)['b']
        else:
            with timer(self.root,'reference_ALS',reference_sha256=key,n_reference_models=int(take.sum())):
                try:b,diag=als_reference(self.ell[take],self.valid[take])
                except Exception as ex:
                    dump(self.path/f'{key}.failure.json',dict(identity=stamp,failure=str(ex)));raise
            np.savez(p,b=b);meta=dict(identity=stamp,diagnostics=diag,b_file_sha256=digest(p));dump(j,meta)
        assert np.array_equal(np.isfinite(b),eligible)
        self.cache[key]=b;self.meta[key]=meta;return key,b
    def project(self,ctx,base):
        score=set(ctx['scored_orgs']);ref=set(ctx['reference_orgs']);assert not score&ref
        key,b=self.reference(ref);ix=np.where(np.isin(self.orgs,list(score)))[0];out=[]
        with timer(self.root,'source_projection',context=ctx['context'],reference_sha256=key,n_models=len(ix)):
            for i in ix:
                r=project_source(self.ell[i],self.valid[i],b)
                r.update(model_id=self.mids[i],org_id=self.orgs[i],reference_sha256=key,context=ctx['context'],**base[self.mids[i]])
                if not r['valid_summaries']:r['exclusion_reason']=r.get('exclusion_reason','')+';NONFINITE_SUMMARIES'
                # Failed-fit diagnostics may contain nonfinite numbers; CSV records explicit strings, never used as inputs.
                out.append(r)
        return out

def validate_qualification(root):
    tests=[];exclusions=[];counts={};context_counts={}
    for k in range(5):
        d=root/'features'/f'outer_{k}';test=rows(d/'test.csv');good=qualified(test);tests+=good
        counts[str(k)]=len({r['org_id'] for r in qualified(rows(d/'train.csv'))})
        for p in sorted(d.glob('*.csv')):
            rr=rows(p);ok=qualified(rr);context_counts[str(p.relative_to(root))]=dict(n_models=len(ok),n_orgs=len({r['org_id'] for r in ok}))
            good_ids={r['model_id'] for r in ok}
            exclusions.extend(dict(model_id=r['model_id'],org_id=r['org_id'],file=str(p.relative_to(root)),context=r['context'],reason=r.get('exclusion_reason','')) for r in rr if r['model_id'] not in good_ids)
    reasons=[]
    if len(tests)<300:reasons.append('QUALIFIED_MODELS_LT300')
    if len({r['org_id'] for r in tests})<30:reasons.append('QUALIFIED_ORGS_LT30')
    if min(counts.values())<20:reasons.append('QUALIFIED_OUTER_TRAIN_ORGS_LT20')
    if any(v['n_models']==0 for v in context_counts.values()):reasons.append('EMPTY_QUALIFIED_CONTEXT')
    assert len({r['model_id'] for r in tests})==len(tests)
    tests.sort(key=lambda r:r['model_id']);write_rows(root/'evaluation_cohort.csv',tests)
    write_rows(root/'exclusions.csv',exclusions,fields=['model_id','org_id','file','context','reason'])
    report=dict(n_models=len(tests),n_orgs=len({r['org_id'] for r in tests}),outer_train_org_counts=counts,contexts=context_counts,reasons=reasons,sufficient=not reasons)
    dump(root/'A3_qualification.json',report)
    if reasons:raise RuntimeError('INCONCLUSIVE: '+','.join(reasons))
    return report

def pilot_identity(root):
    manifest=rows(root/'cohort_manifest.csv');counts={}
    for r in manifest:counts[r['org_id']]=counts.get(r['org_id'],0)+1
    contexts=readjson(root/'split_manifest.json')['projection_contexts']
    largest=max(contexts,key=lambda c:sum(counts.get(g,0) for g in c['reference_orgs']))
    stamp=dict(source_sha256=readjson(root/'cohort_contract.json')['source_arrays_sha256'],
      reference_sha256=largest['reference_set_sha256'],reference_orgs=largest['reference_orgs'],
      source_stage_sha256=digest(root/'code/source_stage.py'),core_sha256=digest(root/'code/legacy_primitives.py'),
      A1_bundle_sha256=digest(root/'A1_OUTPUT_HASHES.json'),DESIGN_RELEASE_sha256=digest(root/'DESIGN_RELEASE.json'))
    return stamp

def validate_pilot(report,identity):
    if report.get('identity')!=identity:raise RuntimeError('pilot identity changed; no cache-time substitution')
    event=report['first_required_fit_event']
    if event.get('stage')!='reference_ALS' or event.get('reference_sha256')!=identity['reference_sha256']:raise RuntimeError('pilot original fit evidence invalid')
    seconds=float(event['wall_seconds'])
    if not np.isfinite(seconds) or seconds<=0:raise RuntimeError('invalid first-fit pilot duration')
    estimate=105*seconds+300
    if report['conservative_source_seconds']!=estimate or report['scope_gate']!=(estimate>43200):raise RuntimeError('pilot gate fields inconsistent')
    if estimate>43200:raise RuntimeError('G-SCOPE: source fits predicted >12h; cached fit does not release gate')
    return report

def pilot(root=ROOT):
    require_design(root);identity=pilot_identity(root);path=root/'source_resource_pilot.json'
    if path.exists():
        report=validate_pilot(readjson(path),identity)
        print('existing original pilot identity/cost/gate revalidated; no refit',flush=True);return report
    with timer(root,'source_io',component='pilot_source_array'):data,_=load_source(root)
    ref=References(data,root);ref.reference(identity['reference_orgs'])
    # Recover only the actual original necessary-fit event if interrupted after
    # caching. A cache-read duration is never accepted as an ALS cost estimate.
    events=[e for e in event_list(root) if e['stage']=='reference_ALS' and e['reference_sha256']==identity['reference_sha256']]
    if len(events)!=1:raise RuntimeError('missing/ambiguous original pilot timing; cached fit cannot bypass gate')
    estimate=105*events[0]['wall_seconds']+300
    report=dict(identity=identity,first_required_fit_event=events[0],conservative_source_seconds=estimate,threshold_seconds=43200,scope_gate=estimate>43200)
    dump(path,report);validate_pilot(report,identity)
    print(json.dumps(dict(source_pilot_seconds=events[0]['wall_seconds'],estimate_seconds=estimate,scope_gate=False)),flush=True)
    return report

def run(root=ROOT):
    require_design(root)
    if (root/'A3_COMPLETE.json').exists():raise RuntimeError('A3 already complete; do not overwrite first-run cost')
    validate_pilot(readjson(root/'source_resource_pilot.json'),pilot_identity(root))
    with timer(root,'source_io',component='frozen_source_array'):
        data,manifest=load_source(root)
    base={};source=[];itemindex={str(x):i for i,x in enumerate(data['item_ids'])}
    for ix,rec in enumerate(manifest):
        with timer(root,'source_io',component='raw_arc',model_id=rec['model_id']):raw=read_arc(rec)
        with timer(root,'source_summaries',group='C',model_id=rec['model_id']) as event:
            stats,obs=summaries(raw['loglik'],raw['char_lens'],raw['gold'],raw['item_ids'],raw['n_options'])
            event.update(base_component_wall_seconds=stats['base_component_wall_seconds'],base_component_cpu_seconds=stats['base_component_cpu_seconds'])
        stats.pop('base_component_wall_seconds');stats.pop('base_component_cpu_seconds')
        # Exact agreement with frozen source observation arrays is an input check, not a target comparison.
        jj=np.array([itemindex[str(x)] for x in raw['item_ids']]);e,c,y,v=obs
        assert np.array_equal(v,data['valid'][ix,jj])
        assert np.allclose(e[v],data['ell'][ix,jj][v],atol=1e-12,rtol=0)
        assert np.allclose(c[v],data['c'][ix,jj][v],atol=1e-12,rtol=0) and np.array_equal(y[v],data['y'][ix,jj][v])
        base[rec['model_id']]=stats;source.append(dict(model_id=rec['model_id'],org_id=rec['org_id'],**stats))
    write_rows(root/'source_features.csv',source)
    ref=References(data,root);sp=readjson(root/'split_manifest.json');assembled={};ctxlog=[]
    for ctx in sp['projection_contexts']:
        r=ref.project(ctx,base);parts=ctx['context'].split('/');k=int(parts[0][5:])
        if parts[1]=='test':name='test'
        elif parts[1]=='train':name='train'
        else:name=parts[1]+'_'+parts[2]
        assembled.setdefault((k,name),[]).extend(r)
        ctxlog.append(dict(context=ctx['context'],n_models=len(r),n_qualified=len(qualified(r)),reference_sha256=ctx['reference_set_sha256']))
    for (k,name),rs in assembled.items():write_rows(root/'features'/f'outer_{k}'/f'{name}.csv',rs)
    dump(root/'fit_diagnostics.json',dict(references=ref.meta,projection_contexts=ctxlog,n_distinct_references=len(ref.meta)))
    q=validate_qualification(root)
    paths=sorted((root/'features').glob('outer_*/*.csv'))+[root/'evaluation_cohort.csv',root/'source_features.csv',root/'A3_qualification.json']
    dump(root/'features/FEATURES_READY.json',dict(files={str(p.relative_to(root)):digest(p) for p in paths},A1_bundle_sha256=digest(root/'A1_OUTPUT_HASHES.json')))
    dump(root/'cost_source.json',dict(events=[e for e in event_list(root) if e['stage'] in ['source_io','source_summaries','reference_ALS','source_projection']],source_labels=sum(int(r['n_valid_source']) for r in source),n_reference_fits=len(ref.meta),n_projection_contexts=len(ctxlog),extra_LLM_calls=0))
    dump(root/'A3_COMPLETE.json',dict(qualification=q,source_features_sha256=digest(root/'source_features.csv')))
    print(json.dumps(q),flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','run']);args=ap.parse_args()
    (pilot if args.stage=='pilot' else run)()
