"""Join source-only metrics to the exact legacy context-qualified row lists."""
from common import *
from scipy.stats import spearmanr
def run(root=ROOT):
    require_design(root);pr=protocol(root);cc=readjson(root/'cohort_contract.json')
    assert not (root/'features/FEATURES_READY.json').exists()
    am=readjson(root/'features/ab_manifest.json');assert digest(root/am['file'])==am['sha256']
    sm=readjson(root/'features/STRUCTURE_READY.json')
    for n,h in sm['files'].items():assert digest(root/n)==h,n
    with np.load(root/am['file'],allow_pickle=False) as z:ab={k:z[k] for k in ['model_ids','metric_ids','values','reasons','n_items']}
    assert ab['metric_ids'].tolist()==pr['standard_ids'];abidx={m:i for i,m in enumerate(ab['model_ids'].astype(str))}
    cache={};files={};coverage={};half_rows=[];reason_counts={j:{} for j in pr['metric_ids']}
    with timer(root,'source_assembly'):
        for filename,fr in cc['fixed_rows'].items():
            old=qualified(rows(root/fr['path']));assert [r['model_id'] for r in old]==fr['model_ids']
            out=[]
            for r0 in old:
                r=dict(r0);context=r['context'];mid=r['model_id'];ai=abidx[mid]
                if context not in cache:
                    p=root/'features/structure_contexts'/(context.replace('/','__')+'.npz')
                    assert sm['files'][str(p.relative_to(root))]==digest(p)
                    with np.load(p,allow_pickle=False) as z:st={k:z[k] for k in ['model_ids','metric_ids','values','reasons','half_values','half_reasons','n_C','n_D_fit','n_D_eval']}
                    assert st['metric_ids'].tolist()==pr['structure_ids']
                    cache[context]=(st,{m:i for i,m in enumerate(st['model_ids'].astype(str))})
                st,si=cache[context];ii=si[mid]
                values=np.r_[ab['values'][ai,0],st['values'][ii]];reasons=np.r_[ab['reasons'][ai,0],st['reasons'][ii]]
                assert np.array_equal(np.isfinite(values),reasons=='')
                for j,v,reason in zip(pr['metric_ids'],values,reasons):
                    r[j]=float(v);r['missing_'+j]=int(not np.isfinite(v));r['reason_'+j]=str(reason)
                    if reason:reason_counts[j][str(reason)]=reason_counts[j].get(str(reason),0)+1
                out.append(r)
                if filename.endswith('/test.csv'):
                    hv=np.concatenate([ab['values'][ai,1:3],st['half_values'][ii]],axis=1)
                    half_rows.append((mid,r['org_id'],values,hv))
            write_rows(root/filename,out);files[filename]=digest(root/filename)
            coverage[filename]=dict(n_models=len(out),n_orgs=len({r['org_id'] for r in out}),finite_counts={j:int(sum(np.isfinite(float(r[j])) for r in out)) for j in pr['metric_ids']})
    assert len(half_rows)==len({r[0] for r in half_rows})==cc['n_models']
    h=np.stack([x[3] for x in half_rows]);full=np.stack([x[2] for x in half_rows]);stability=[]
    for j,metric in enumerate(pr['metric_ids']):
        use=np.isfinite(h[:,:,j]).all(1);rho=None
        if use.sum()>=2 and np.ptp(h[use,0,j])>0 and np.ptp(h[use,1,j])>0:rho=float(spearmanr(h[use,0,j],h[use,1,j]).statistic)
        stability.append(dict(metric=metric,n_valid_full=int(np.isfinite(full[:,j]).sum()),n_valid_pairs=int(use.sum()),n_orgs_valid_pairs=len({half_rows[i][1] for i in np.flatnonzero(use)}),half_spearman=rho,weighting='unweighted model-pair descriptive Spearman; no source-based outcome selection'))
    np.savez(root/'features/source_half_values.npz',model_ids=np.array([r[0] for r in half_rows]),org_ids=np.array([r[1] for r in half_rows]),metric_ids=np.array(pr['metric_ids']),values=full,half_values=h)
    dump(root/'features/source_quality.json',dict(coverage=coverage,reason_counts_across_context_rows=reason_counts,stability=stability,half_axis_semantics={'A_B_C':'index h uses source items H_h','D':'index h fits H_h and evaluates H_(1-h); final value requires both directions'},source_only=True))
    files.update({n:digest(root/n) for n in ['features/source_half_values.npz','features/source_quality.json']})
    dump(root/'features/FEATURES_READY.json',dict(files=files,ab_manifest_sha256=digest(root/'features/ab_manifest.json'),structure_ready_sha256=digest(root/'features/STRUCTURE_READY.json'),E1_bundle_sha256=digest(root/'E1_BUNDLE.json'),CODE_RELEASE_sha256=digest(root/'CODE_RELEASE.json'),n_models=cc['n_models'],n_orgs=cc['n_orgs'],n_contexts=125,n_feature_row_files=50,n_metrics=20,status='COMPLETE_PENDING_V_SOURCE'))
    print(json.dumps(dict(status='COMPLETE_PENDING_V_SOURCE',n_models=cc['n_models'],n_metrics=20)),flush=True)
if __name__=='__main__':run()
