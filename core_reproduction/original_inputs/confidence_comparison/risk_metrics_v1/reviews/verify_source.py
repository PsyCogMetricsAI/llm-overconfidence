"""Independent V-SOURCE audit: no production metric imports and no target labels."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv, hashlib, json, time
import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata, spearmanr

W = Path(__file__).resolve().parents[1]
def h(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(path): return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open() as f: return list(csv.DictReader(f))
def nz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}
errors = {}
def eq(got, expected, label, atol=1e-10):
    a, b = np.asarray(got, float), np.asarray(expected, float)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), (label, 'finite mask')
    use = np.isfinite(a)
    error = float(np.max(np.abs(a[use]-b[use]))) if use.any() else 0.
    errors[label.split(':')[0]] = max(errors.get(label.split(':')[0], 0.), error)
    assert error <= atol, (label, error)

def ab_formula(c, y, z, nll, ranks, entropy, ids):
    n = len(c)
    if n == 0: return np.full(12, np.nan)
    y = y.astype(float); e = 1-y
    order = np.lexsort((ids, -c)); acc = y.mean()
    one, zero = c[y == 1], c[y == 0]
    auroc = ((one[:, None] > zero).mean() + .5*(one[:, None] == zero).mean()) if len(one) and len(zero) else np.nan
    resolution = 0.
    for b in range(10):
        use = np.minimum((10*c).astype(int), 9) == b
        if use.any(): resolution += use.mean()*(y[use].mean()-acc)**2
    return np.array([auroc, np.mean(np.cumsum(e[order])/np.arange(1,n+1)), resolution,
        e[order[:int(np.ceil(.1*n))]].mean(), e[order[:int(np.ceil(.25*n))]].mean(),
        nll.mean(), (e*c).mean(), (e*c*c).mean(), z.mean(),
        np.sort(z)[-int(np.ceil(.1*n)):].mean(), ranks.mean(), entropy.mean()])

def c_formula(c,y,z,valid,ids,w):
    take = valid & np.isfinite(w); n = int(take.sum())
    if not n: return np.full(4,np.nan)
    cc,yy,zz,ww = c[take],y[take],z[take],w[take]
    order = np.lexsort((ids[take],ww)); v=(cc-yy)[order]
    bins=4*np.arange(n)//n
    hetero=sum(np.mean(bins==b)*(v[bins==b].mean()-v.mean())**2 for b in range(4)) if n>=4 else np.nan
    return np.array([np.mean(ww*(1-yy)),np.mean(ww*(1-yy)*cc),np.mean(ww*zz),hetero])

def d_formula(ell,valid,halves,ref,direction):
    if str(ref['direction_failures'][direction]): return np.full(4,np.nan), None
    b=ref['b_fit'][direction]; be=ref['b_eval'][direction]
    take=valid & (halves==direction) & np.isfinite(b) & np.isfinite(ell)
    other=valid & (halves==1-direction) & np.isfinite(be) & np.isfinite(ell)
    if take.sum()<100: return np.full(4,np.nan),None
    x,y=b[take],ell[take]; dx=x-x.mean();dy=y-y.mean()
    vx=np.mean(dx*dx);vy=np.mean(dy*dy)
    if vx<=1e-12: return np.full(4,np.nan),None
    slope=-np.mean(dx*dy)/vx; intercept=y.mean()+slope*x.mean()
    sigma=np.sqrt(np.mean((y-intercept+slope*x)**2))
    if slope<=1e-8 or sigma<=1e-8 or vy<=1e-6: return np.full(4,np.nan),None
    pars=np.array([intercept,slope,sigma,intercept/slope,vy])
    if other.sum()<100: return np.full(4,np.nan),pars
    r=(ell[other]-intercept+slope*be[other])/sigma
    bc=be[other]; centered=r-r.mean(); m2=np.mean(centered**2)
    vals=np.full(4,np.nan)
    den=np.sum(r*r)
    if den>1e-12:vals[0]=np.sum(np.minimum(r,0)**2)/den
    if m2>1e-12:vals[1]=np.mean(centered**4)/m2**2-3
    for i,x,y in [(2,rankdata(abs(r)),rankdata(bc)),(3,r,bc**2)]:
        dx,dy=x-x.mean(),y-y.mean();vx,vy=np.mean(dx*dx),np.mean(dy*dy)
        if vx>1e-12 and vy>1e-12: vals[i]=np.mean(dx*dy)/np.sqrt(vx*vy)
    return vals,pars

def run():
    start=time.monotonic();pr=js(W/'protocol.json');split=js(W/'split_manifest.json');cc=js(W/'cohort_contract.json')
    ready=js(W/'features/FEATURES_READY.json');abm=js(W/'features/ab_manifest.json');stm=js(W/'features/STRUCTURE_READY.json');rm=js(W/'references/REFERENCES_READY.json')
    assert ready['CODE_RELEASE_sha256']==h(W/'CODE_RELEASE.json') and ready['E1_bundle_sha256']==h(W/'E1_BUNDLE.json')
    assert ready['ab_manifest_sha256']==h(W/'features/ab_manifest.json') and ready['structure_ready_sha256']==h(W/'features/STRUCTURE_READY.json')
    assert stm['references_ready_sha256']==h(W/'references/REFERENCES_READY.json')
    for doc in [ready,stm,rm]:
        for name,sha in doc['files'].items():assert h(W/name)==sha,name
    assert h(W/abm['file'])==abm['sha256'] and stm['z_sha256']==abm['sha256']
    assert h(pr['source_arrays'])==cc['legacy']['source_arrays_sha256']
    source=nz(pr['source_arrays']);ab=nz(W/abm['file']);records=rows(W/'cohort_manifest.csv')
    mids=source['model_ids'].astype(str);orgs=source['org_ids'].astype(str);ids=source['item_ids'].astype(str);halves=source['source_half'];mi={m:i for i,m in enumerate(mids)}
    assert ab['model_ids'].astype(str).tolist()==mids.tolist() and ab['item_ids'].astype(str).tolist()==ids.tolist()
    assert ab['metric_ids'].tolist()==pr['standard_ids'] and np.array_equal(np.isfinite(ab['values']),ab['reasons']=='')
    assert np.array_equal(np.isfinite(ab['z']),source['valid'])
    # Fixed evenly spaced canonical source models; no outcome-dependent sample.
    sample=np.linspace(0,len(mids)-1,10,dtype=int);positions={item:i for i,item in enumerate(ids)}
    for idx in sample:
        rec=records[idx];assert rec['model_id']==mids[idx] and h(rec['arc_path'])==rec['arc_sha256'];raw=nz(rec['arc_path'])
        cols=np.array([positions[str(item)] for item in raw['item_ids']]);take=source['valid'][idx,cols]
        c=[];y=[];z=[];nll=[];ranks=[];entropy=[];ell=[]
        for ii in np.flatnonzero(take):
            real=np.flatnonzero(np.isfinite(raw['char_lens'][ii]) & (raw['char_lens'][ii]>0))
            u=raw['loglik'][ii,real]/raw['char_lens'][ii,real];g=int(np.flatnonzero(real==int(raw['gold'][ii]))[0]);winner=int(np.argmax(u));lse=logsumexp(u)
            c.append(np.exp(u[winner]-lse));y.append(int(winner==g));z.append(u[winner]-u[g]);nll.append(lse-u[g]);ell.append(u[g]-logsumexp(np.delete(u,g)))
            order=np.lexsort((real,-u));ranks.append(int(np.flatnonzero(order==g)[0])/(len(real)-1))
            rest=np.delete(u,winner);logq=rest-logsumexp(rest);entropy.append(float(-np.sum(np.exp(logq)*logq)/np.log(len(rest))) if len(rest)>1 else 0.)
        arrays=list(map(np.asarray,[c,y,z,nll,ranks,entropy]));validcols=cols[take]
        for key,values in [('c',arrays[0]),('y',arrays[1]),('z',arrays[2]),('ell',ell)]:
            eq(ab['z'][idx,validcols] if key=='z' else source[key][idx,validcols],values,'raw_'+key+':'+mids[idx])
        for s in range(3):
            use=np.ones(len(validcols),bool) if s==0 else halves[validcols]==s-1
            expected=ab_formula(*[a[use] for a in arrays],ids[validcols][use]);eq(ab['values'][idx,s],expected,'AB:'+mids[idx])
            assert ab['n_items'][idx,s]==use.sum()
    contexts=split['projection_contexts'];assert len(contexts)==125 and len(rm['files'])==110
    distinct={c['reference_set_sha256']:c for c in contexts};assert len(distinct)==55
    references={}
    for key,ct in distinct.items():
        meta=js(W/'references'/(key+'.json'));ref=nz(W/'references'/(key+'.npz'));ri=np.flatnonzero(np.isin(orgs,ct['reference_orgs']))
        assert set(ct['reference_orgs']).isdisjoint(ct['scored_orgs'])
        assert meta['identity']['reference_model_ids']==mids[ri].tolist() and meta['identity']['reference_orgs']==sorted(ct['reference_orgs'])
        assert ref['reference_model_ids'].astype(str).tolist()==mids[ri].tolist() and np.array_equal(ref['source_half'],halves) and np.array_equal(ref['item_ids'],source['item_ids'])
        assert h(W/'references'/(key+'.npz'))==meta['npz_sha256']
        for direction in (0,1):
            if not str(ref['direction_failures'][direction]):
                b=ref['b_fit'][direction];good=np.isfinite(b)
                assert not np.isfinite(b[halves!=direction]).any() and not np.isfinite(ref['b_eval'][direction,halves==direction]).any()
                assert abs(b[good].mean())<1e-10 and abs(b[good].std()-1)<1e-10
        references[key]=ref
    # Three deterministic reference sets, independent equal-organization smoothing
    # and fitted-parameter evaluation projection, without refitting the ALS search.
    for key in sorted(distinct)[::max(1,len(distinct)//3)][:3]:
        ct=distinct[key];ref=references[key];ri=np.flatnonzero(np.isin(orgs,ct['reference_orgs']))
        sums=np.zeros(len(ids));counts=np.zeros(len(ids),int);ng=np.zeros(len(ids),int)
        for g in sorted(ct['reference_orgs']):
            take=np.flatnonzero(orgs==g);valid=source['valid'][take];n=valid.sum(0);correct=np.where(valid,source['y'][take],0).sum(0);active=n>0
            sums[active]+=(correct[active]+.5)/(n[active]+1);counts+=n;ng+=active
        expected=np.full(len(ids),np.nan);good=counts>=20;expected[good]=sums[good]/ng[good]
        eq(ref['w'],expected,'reference_w:'+key);assert np.array_equal(counts,ref['n_reference']) and np.array_equal(ng,ref['n_reference_orgs'])
        for direction in (0,1):
            if str(ref['direction_failures'][direction]):continue
            a=[];s=[];b=ref['b_fit'][direction]
            for idx in ri:
                take=source['valid'][idx]&np.isfinite(b);x=b[take];y=source['ell'][idx,take]
                slope=-np.mean((x-x.mean())*(y-y.mean()))/np.var(x);a.append(y.mean()+slope*x.mean());s.append(slope)
            a=np.array(a);s=np.array(s);eq(ref['a_reference'][direction],a,'reference_a:'+key);eq(ref['s_reference'][direction],s,'reference_s:'+key)
            other=halves==1-direction;valid=source['valid'][ri][:,other];ell=source['ell'][ri][:,other]
            den=np.sum(valid*s[:,None]**2,0);num=np.sum(np.where(valid,a[:,None]-ell,0)*s[:,None],0)
            vals=np.full(other.sum(),np.nan);good=(valid.sum(0)>=20)&(den>1e-20);vals[good]=num[good]/den[good]
            eq(ref['b_eval'][direction,other],vals,'reference_b_eval:'+key)
    stcache={};sampled=[]
    for ct in contexts:
        context=ct['context'];st=nz(W/'features/structure_contexts'/(context.replace('/','__')+'.npz'));ref=references[ct['reference_set_sha256']]
        indices=np.flatnonzero(np.isin(orgs,ct['scored_orgs']));assert st['model_ids'].astype(str).tolist()==mids[indices].tolist()
        assert set(ct['reference_orgs']).isdisjoint(ct['scored_orgs']) and st['metric_ids'].tolist()==pr['structure_ids']
        assert np.array_equal(np.isfinite(st['values']),st['reasons']=='') and np.array_equal(np.isfinite(st['half_values']),st['half_reasons']=='')
        validboth=np.isfinite(st['half_values'][:,:,4:]).all(1);assert np.array_equal(np.isfinite(st['values'][:,4:]),validboth)
        eq(st['values'][:,4:][validboth],st['half_values'][:,:,4:].mean(1)[validboth],'D_double_half:'+context)
        # First source model per context; fixed before target access.
        row=0;idx=indices[row];c,y,z,valid=source['c'][idx],source['y'][idx].astype(float),ab['z'][idx],source['valid'][idx]
        cv=c_formula(c,y,z,valid,ids,ref['w']);dv=[]
        for direction in (0,1):
            dh,pars=d_formula(source['ell'][idx],valid,halves,ref,direction);dv.append(dh)
            ch=c_formula(c,y,z,valid&(halves==direction),ids,ref['w']);eq(st['half_values'][row,direction],np.r_[ch,dh],'CD_half:'+context)
            if pars is not None:eq(st['D_parameters'][row,direction],pars,'D_parameters:'+context)
        eq(st['values'][row],np.r_[cv,np.mean(dv,axis=0)],'CD_full:'+context);sampled.append((context,str(mids[idx])))
        stcache[context]=(st,{m:j for j,m in enumerate(st['model_ids'].astype(str))})
    joined=0;coverage={}
    for name,fixed in cc['fixed_rows'].items():
        got=rows(W/name);old=[r for r in rows(W/fixed['path']) if r['valid_source_fit'].lower()=='true' and r['valid_summaries'].lower()=='true']
        assert [r['model_id'] for r in got]==fixed['model_ids'] and [r['org_id'] for r in got]==fixed['org_ids'] and len(got)==len(old)
        values=[]
        for row,prior in zip(got,old):
            assert all(row[key]==value for key,value in prior.items()),(name,row['model_id'],'legacy column')
            st,index=stcache[row['context']];ii=index[row['model_id']];ai=mi[row['model_id']]
            expected=np.r_[ab['values'][ai,0],st['values'][ii]];actual=np.array([float(row[m]) for m in pr['metric_ids']]);eq(actual,expected,'join:'+name)
            assert all(int(row['missing_'+m])==int(not np.isfinite(actual[j])) and (row['reason_'+m]=='')==np.isfinite(actual[j]) for j,m in enumerate(pr['metric_ids']))
            values.append(actual)
        joined+=len(got);coverage[name]={'n_models':len(got),'n_orgs':len({r['org_id'] for r in got}),'finite_counts':dict(zip(pr['metric_ids'],map(int,np.isfinite(values).sum(0))))}
    quality=js(W/'features/source_quality.json');assert quality['coverage']==coverage
    prior_rows=W/'admin/e2_assembly_json_fix/pre_fix_row_hashes.json'
    if prior_rows.exists():
        previous=js(prior_rows)
        prior_map=previous.get('files',previous)
        for name,sha in prior_map.items():assert h(W/name)==sha,('assembly_repair_changed_row_bytes',name)
    half=nz(W/'features/source_half_values.npz');assert len(half['model_ids'])==len(set(half['model_ids'].tolist()))==cc['n_models']
    for j,stat in enumerate(quality['stability']):
        use=np.isfinite(half['half_values'][:,:,j]).all(1);assert stat['n_valid_full']==int(np.isfinite(half['values'][:,j]).sum()) and stat['n_valid_pairs']==int(use.sum())
        assert stat['n_orgs_valid_pairs']==len(set(half['org_ids'][use].tolist()))
        pair=half['half_values'][use,:,j]
        rho=float(spearmanr(pair[:,0],pair[:,1]).statistic) if len(pair)>=2 and np.ptp(pair[:,0])>0 and np.ptp(pair[:,1])>0 else None
        assert (rho is None)==(stat['half_spearman'] is None)
        if rho is not None:eq(rho,stat['half_spearman'],'half_stability:'+pr['metric_ids'][j])
    target=js(W/'targets/TARGETS_READY.json')
    assert target['CODE_RELEASE_sha256']==h(W/'CODE_RELEASE.json') and target['new_effects_computed'] is False
    assert target['n_models']==len(mids) and target['targets_sha256']==target['legacy_targets_sha256']
    assert not list((W/'targets').glob('access_*.jsonl')),'training/scoring access before source release'
    result={'status':'PASS','reviewer':'independent statistical checker','production_modules_imported':False,'target_labels_read':False,'formula_tolerance':1e-10,'max_errors':errors,'n_raw_AB_models':len(sample),'n_AB_split_recomputations':3*len(sample),'n_reference_sets_checked':55,'n_reference_sets_numeric_recomputed':3,'n_contexts_checked':125,'n_CD_models_recomputed':len(sampled),'n_feature_files_checked':len(cc['fixed_rows']),'n_joined_rows_checked':joined,'source_models':len(mids),'evaluation_models':cc['n_models'],'evaluation_orgs':cc['n_orgs'],'source_ready_sha256':h(W/'features/FEATURES_READY.json'),'code_release_sha256':h(W/'CODE_RELEASE.json'),'E1_bundle_sha256':h(W/'E1_BUNDLE.json'),'target_readiness_sha256':h(W/'targets/TARGETS_READY.json'),'audit_script_sha256':h(__file__),'wall_seconds':time.monotonic()-start,'sample_context_models':sampled}
    (W/'reviews/SOURCE_NUMERIC_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sample_context_models'}),flush=True)
if __name__=='__main__':run()
