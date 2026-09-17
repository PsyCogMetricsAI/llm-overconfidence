#!/usr/bin/env python3
"""CV0 char-normalized observation contract. Target scan computes validity only.
All outputs confined to confidence_validity; original caches read-only.
"""
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

OUT=Path(__file__).resolve().parents[1]
PROJECT=Path.cwd()
DATA=PROJECT/'data_peritem_v1_20260913'
SEED=20260914

def sha_order(ids,k):
    ordered=sorted(set(map(str,ids)),key=lambda s:(hashlib.sha256(f'{SEED}|{s}'.encode()).hexdigest(),s))
    return {s:j%k for j,s in enumerate(ordered)}

def recover_identity(z):
    """Accept explicit model identifier or known leaderboard namespace encoding only."""
    if 'model_id' in z.files:
        val=str(z['model_id'].item())
        if '/' in val and len(val.split('/'))==2 and all(val.split('/')):
            return val,val.split('/',1)[0],'explicit_model_id'
    if 'repo' not in z.files:raise ValueError('missing_identity')
    repo=str(z['repo'].item())
    prefixes=('open-llm-leaderboard-old/details_','open-llm-leaderboard/details_')
    for prefix in prefixes:
        if repo.startswith(prefix):
            short=repo[len(prefix):]
            if '__' not in short:raise ValueError('unrecoverable_org_encoding')
            org,name=short.split('__',1)
            if not org or not name or '/' in org or '/' in name:raise ValueError('invalid_encoded_identity')
            return org+'/'+name,org,'leaderboard_details_double_underscore'
    raise ValueError('unrecognized_repo_namespace')

def validity(loglik,char_lens,gold,n_options=None):
    ll=np.asarray(loglik,dtype=np.float64); h=np.asarray(char_lens,dtype=np.float64)
    rawg=np.asarray(gold)
    if ll.ndim!=2 or h.shape!=ll.shape or rawg.shape!=(ll.shape[0],):raise ValueError('shape_mismatch')
    n,k=ll.shape
    real=np.isfinite(h)&(h>0)
    nr=real.sum(1)
    # Cache padding is NaN in both h and L. Inf/zero/negative h is invalid,
    # not a missing option silently dropped from a probability denominator.
    bad_h=np.any((~np.isnan(h))&(~real),axis=1)
    padding_bad=np.any(np.isnan(h)&(~np.isnan(ll)),axis=1)
    complete=np.all((~real)|np.isfinite(ll),axis=1)
    gfinite=np.isfinite(rawg)
    gint=gfinite&(rawg==np.floor(np.where(gfinite,rawg,0)))
    g=np.where(gint,rawg,0).astype(np.int64)
    legal=gint&(g>=0)&(g<k)
    ix=np.arange(n);safe=np.clip(g,0,max(k-1,0))
    legal &= real[ix,safe]
    mismatch=np.zeros(n,dtype=bool)
    if n_options is not None:
        no=np.asarray(n_options)
        if no.shape!=(n,):raise ValueError('n_options_shape')
        mismatch=(~np.isfinite(no))|(no!=nr)
    reason={'bad_char_length':bad_h,'padding_has_loglik':padding_bad,'incomplete_real_loglik':~complete,
            'invalid_gold':~legal,'fewer_than_two_options':nr<2,'n_options_disagrees':mismatch}
    valid=np.ones(n,dtype=bool)
    for a in reason.values():valid &= ~a
    return valid,real,g,{a:int(b.sum()) for a,b in reason.items()},nr

def observations(loglik,char_lens,gold,n_options=None):
    ll=np.asarray(loglik,dtype=np.float64);h=np.asarray(char_lens,dtype=np.float64)
    valid,real,g,reasons,nr=validity(ll,h,gold,n_options)
    n,k=ll.shape
    ell=np.full(n,np.nan);c=np.full(n,np.nan);y=np.zeros(n,dtype=bool)
    r=np.flatnonzero(valid)
    err=0.0
    if len(r):
        u=np.full((len(r),k),-np.inf)
        np.divide(ll[r],h[r],out=u,where=real[r])
        pred=np.argmax(u,axis=1) # first index resolves ties
        p=np.exp(u-logsumexp(u,axis=1,keepdims=True))
        err=float(np.max(np.abs(p.sum(1)-1)))
        c[r]=p[np.arange(len(r)),pred];y[r]=pred==g[r]
        gold_u=u[np.arange(len(r)),g[r]].copy()
        u[np.arange(len(r)),g[r]]=-np.inf
        ell[r]=gold_u-logsumexp(u,axis=1)
        if not np.all(np.isfinite(ell[r])):raise ValueError('finite_inputs_nonfinite_logodds')
        if err>1e-12:raise AssertionError('probability_sum_tolerance')
    return ell,c,y,valid,reasons,nr,err

def dump_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def selftest():
    ll=np.array([[0.,0.,np.nan],[-1000.,-1001.,np.nan],[1.,np.nan,np.nan],[0.,0.,0.]])
    h=np.array([[1.,1.,np.nan],[1.,1.,np.nan],[1.,1.,np.nan],[1.,1.,1.]])
    e,c,y,v,_,_,err=observations(ll,h,np.array([0,1,0,1]),np.array([2,2,2,3]))
    assert v.tolist()==[True,True,False,True]
    assert y.tolist()==[True,False,False,False] and c[0]==.5
    assert abs(e[0])<1e-12 and abs(e[1]+1)<1e-12
    # Correct top-label choice can have negative gold-vs-rest logodds.
    e,c,y,v,*_=observations([[0.,-.1,-.2]],[[1,1,1]],[0],[3])
    assert bool(y[0]) and e[0]<0
    for hh in [np.array([[1.,0.]]),np.array([[1.,np.inf]])]:
        assert not validity([[0.,0.]],hh,[0],[2])[0][0]
    assert not validity([[0.,np.nan]],[[1.,1.]],[0],[2])[0][0]
    return {'status':'PASS','max_probability_sum_error':err,'cases':['ties_smallest_option','stable_softmax','missing_real_option','nonbinary_correctness','invalid_lengths','incomplete_options']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.selftest:
        (OUT/'OBSERVATION_SELFTEST.json').write_text(json.dumps(selftest(),indent=2)+'\n');print('observation tests PASS');return
    selftest();start=time.time();manifest=[];scan=[];source={};target={};schema={};union=set();maxerr=0.;tmp=[];directory_counts={}
    for bench in ['arc','hellaswag']:
        files=sorted((DATA/bench).glob('*.npz'));directory_counts[bench]=len(files)
        for j,f in enumerate(files):
            if f.name.endswith('.tmp.npz'):
                tmp.append(str(f));continue
            before=f.stat();raw=f.read_bytes();after=f.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise RuntimeError('input changed during read: '+str(f))
            sha=hashlib.sha256(raw).hexdigest();manifest.append((sha,str(f)))
            rec={'bench':bench,'path':str(f),'sha256':sha,'model_id':'','org_id':'','n_rows':0,'n_valid':0,'status':'OK','reason':''}
            try:
                z=np.load(io.BytesIO(raw),allow_pickle=False)
                mid,org,rule=recover_identity(z);rec.update(model_id=mid,org_id=org,identity_rule=rule)
                if bench not in schema:schema[bench]={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
                iids=np.asarray(z['item_ids']).astype(str)
                if len(set(iids))!=len(iids):raise ValueError('duplicate_item_id')
                args0=(z['loglik'],z['char_lens'],z['gold'],z['n_options'] if 'n_options' in z.files else None)
                if bench=='arc':
                    e,c,y,v,reasons,nr,err=observations(*args0);maxerr=max(maxerr,err)
                    if mid in source:raise ValueError('duplicate_model_identity')
                    source[mid]=(org,iids,e,c,y,v,str(f),sha);union.update(iids)
                else:
                    v,_,_,reasons,nr=validity(*args0)
                    if mid in target:raise ValueError('duplicate_model_identity')
                    target[mid]=(org,int(v.sum()),str(f),sha)
                rec.update(n_rows=len(iids),n_valid=int(v.sum()),reason_counts=reasons,real_option_counts={str(int(n)):int((nr==n).sum()) for n in np.unique(nr)})
            except Exception as ex:
                if str(ex)=='duplicate_model_identity':
                    raise RuntimeError('duplicate model identity is a cohort ambiguity; stop without selecting either file: '+str(f)) from ex
                rec.update(status='EXCLUDED',reason=f'{type(ex).__name__}: {ex}')
            scan.append(rec)
            if (j+1)%1000==0:print(f'{bench}: metadata/source scan {j+1}/{len(files)} elapsed={time.time()-start:.1f}s',flush=True)
    item_ids=np.array(sorted(union));item_half_map=sha_order(item_ids,2);half=np.array([item_half_map[x] for x in item_ids],dtype=np.int8)
    eligibility=[];keep=[]
    for mid in sorted(set(source)|set(target)):
        src=source.get(mid);tgt=target.get(mid);reasons=[];nf=nc=nt=0
        if src is None:reasons.append('source_absent_or_invalid')
        else:
            org,ids,e,c,y,v,_,_=src;mh=np.array([item_half_map[x] for x in ids]);nf=int((v&(mh==0)).sum());nc=int((v&(mh==1)).sum())
            if nf<100:reasons.append('source_F_lt100')
            if nc<100:reasons.append('source_C_lt100')
        if tgt is None:reasons.append('target_absent_or_invalid')
        else:
            nt=tgt[1]
            if nt<200:reasons.append('target_valid_lt200')
        if src and tgt and src[0]!=tgt[0]:reasons.append('org_mismatch')
        org=src[0] if src else tgt[0]
        eligibility.append(dict(model_id=mid,org_id=org,n_valid_F=nf,n_valid_C=nc,n_valid_target=nt,eligible=not reasons,reasons=';'.join(reasons)))
        if not reasons:keep.append(mid)
    pos={x:i for i,x in enumerate(item_ids)};shape=(len(keep),len(item_ids))
    ell=np.full(shape,np.nan);c=np.full(shape,np.nan);y=np.zeros(shape,dtype=bool);valid=np.zeros(shape,dtype=bool);orgs=[]
    for m,mid in enumerate(keep):
        org,ids,ee,cc,yy,v,*_=source[mid];idx=np.array([pos[x] for x in ids]);ell[m,idx]=ee;c[m,idx]=cc;y[m,idx]=yy;valid[m,idx]=v;orgs.append(org)
    np.savez_compressed(OUT/'source_arrays.npz',ell=ell,c=c,y=y,valid=valid,item_ids=item_ids,model_ids=np.array(keep),org_ids=np.array(orgs),source_half=half)
    dump_csv(OUT/'eligibility.csv',eligibility,['model_id','org_id','n_valid_F','n_valid_C','n_valid_target','eligible','reasons'])
    # Target index is metadata only: never c/y/risk values.
    dump_csv(OUT/'target_file_index.csv',[dict(model_id=m,org_id=target[m][0],path=target[m][2],sha256=target[m][3]) for m in keep],['model_id','org_id','path','sha256'])
    (OUT/'manifest.sha256').write_text(''.join(f'{h}  {f}\n' for h,f in manifest))
    changed=[f for h,f in manifest if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h]
    (OUT/'INPUT_STABILITY.json').write_text(json.dumps({'status':'PASS' if not changed else 'FAIL','changed':changed,'n_checked':len(manifest),'method':'before-after stat around snapshot read; SHA256 entire manifest checked again after scan'},indent=2)+'\n')
    if changed:raise RuntimeError('input hashes changed; inspect INPUT_STABILITY.json')
    (OUT/'input_scan.json').write_text(json.dumps(scan,indent=2)+'\n')
    from score_targets import metrics
    feat=[]
    for m,mid in enumerate(keep):
        v=valid[m];mm,bins=metrics(c[m,v],y[m,v],item_ids[v]);aa=float(y[m,v].mean());cc=float(c[m,v].mean())
        feat.append(dict(model_id=mid,org_id=orgs[m],A=aa,C=cc,O=cc-aa,REL10=mm['Y_cal'],R50=mm['Y_sel'],n_valid_source=int(v.sum())))
    dump_csv(OUT/'source_B_features.csv',feat,['model_id','org_id','A','C','O','REL10','R50','n_valid_source'])
    contract={'model_id':'scientific reproduction','protocol':'contracts/PROTOCOL.md','status':'CV0_COMPLETE_TARGET_SCORING_NOT_RUN','seed':SEED,'schema_examples':schema,
       'formula':{'u':'loglik / char_lens on real positive finite character lengths','c':'softmax(u)[argmax(u)]','y':'argmax(u)==gold; smallest original option index breaks ties','ell':'u_gold-logsumexp(u_nongold)'},
       'invalid_rules':['nonpositive or infinite nonpadding char length','finite loglik in NaN length padding','any real option nonfinite loglik','invalid or noninteger gold','fewer than two real options','n_options disagrees with positive finite char lengths','duplicate item or model ID','unknown org mapping'],
       'identity':'explicit model_id org/name or known leaderboard details_org__name split once at double underscore; preserve single underscores','source_item_split':'all source item IDs SHA256(20260914|id) ascending, rank mod2; 0F/1C; lexicographic ID only hashes tie',
       'arrays_schema':{k:v for k,v in [('shape',shape),('ell','float64; invalid NaN'),('c','float64; invalid NaN'),('y','bool; invalid false, always mask'),('valid','bool'),('item_ids','unicode'),('model_ids','canonical org/model unicode'),('org_ids','unicode'),('source_half','int8 0F/1C')]},
       'directory_counts':directory_counts,'n_unique_source_identities':len(source),'n_unique_target_identities':len(target),'n_common_identities_before_thresholds':len(set(source)&set(target)),'duplicate_identity_rule':'stop; no arbitrary first/last file retained',
       'base_features':{'file':'source_B_features.csv','item_set':'ALL_VALID_ARC (F union C)','columns':['A','C','O','REL10','R50'],'n_valid_source':'audit only, not a feature'},'n_preliminary_models':len(keep),'n_preliminary_orgs':len(set(orgs)),'n_source_items':len(item_ids),'n_F_items':int((half==0).sum()),'n_C_items':int((half==1).sum()),'n_original_files_hashed':len(manifest),'tmp_files_excluded':tmp,
       'max_probability_sum_error':maxerr,'minimum_counts':{'source_F':100,'source_C':100,'target_valid':200},'source_parameter_filters_pending':True,'target_scan':'validity/counts only; no target probability, correctness, REL10 or R50 calculated','target_count_feature_forbidden':True,'runtime_sec':time.time()-start}
    (OUT/'data_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (OUT/'OBSERVATION_SELFTEST.json').write_text(json.dumps(selftest(),indent=2)+'\n')
    print(json.dumps({k:contract[k] for k in ['status','n_preliminary_models','n_preliminary_orgs','n_source_items','max_probability_sum_error','runtime_sec']}),flush=True)
if __name__=='__main__':main()
