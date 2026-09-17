#!/usr/bin/env python3
"""Independent CV3 target scorer and org-whitelisted subprocess service.
Full target values stay in sealed/. Never prints target outcomes.
"""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1]
SEALED=OUT/'sealed'

def metrics(c,y,item_ids):
    c=np.asarray(c,dtype=np.float64);y=np.asarray(y,dtype=np.float64);ids=np.asarray(item_ids).astype(str)
    if c.ndim!=1 or y.shape!=c.shape or ids.shape!=c.shape:raise ValueError('metric_shape')
    if len(c)==0:raise ValueError('empty_evaluation_set')
    if not(np.isfinite(c).all() and np.isfinite(y).all() and ((c>=0)&(c<=1)).all() and np.isin(y,[0,1]).all()):raise ValueError('invalid_metric_input')
    if len(set(ids))!=len(ids):raise ValueError('duplicate_item_id')
    # Search explicit decimal edges: exact edge goes in next bin, c=1 in last.
    b=np.searchsorted(np.arange(1,10,dtype=np.float64)/10,c,side='right')
    bins=[];rel=0.
    for j in range(10):
        z=b==j;n=int(z.sum());cb=float(c[z].mean()) if n else None;yb=float(y[z].mean()) if n else None
        contribution=n/len(c)*(cb-yb)**2 if n else 0.
        rel+=contribution;bins.append(dict(bin=j+1,n=n,cbar=cb,ybar=yb,contribution=contribution))
    order=np.lexsort((ids,-c));k=(len(c)+1)//2
    risk=float(np.mean(1-y[order[:k]]))
    fullrisk=float(np.mean(1-y[order]));acc=float(np.mean(y))
    if abs(fullrisk-(1-acc))>1e-12:raise AssertionError('full_coverage_identity')
    return {'Y_cal':float(rel),'Y_sel':risk,'n_valid':len(c),'k_selected':k},bins

def selftest():
    # Bins .1,.9,1 edge handling, empty bins, nontrivial REL hand calculation.
    m,b=metrics([.1,.1,.9,1.],[0,1,1,0],['a','b','c','d'])
    expected=.5*(.1-.5)**2+.5*(.95-.5)**2
    assert abs(m['Y_cal']-expected)<1e-12 and m['Y_sel']==.5
    assert b[1]['n']==2 and b[9]['n']==2 and b[0]['n']==0
    m,b=metrics([.5,.5,.5],[0,1,1],['z','a','b']);assert m['Y_sel']==0 and m['k_selected']==2
    m,b=metrics([.2,.8],[0,1],['a','b']);assert abs(m['Y_cal']-.04)<1e-12
    try:metrics([],[],[])
    except ValueError:pass
    else:raise AssertionError('empty must reject')
    return {'status':'PASS','cases':['hand_REL10','exact_bin_edges','empty_bins_zero','R50_ties_by_item_ID','odd_N_ceiling','full_coverage_identity','empty_rejected'],'tolerance':1e-12}

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def require_gate():
    """Main controller writes an explicit authorization only after independent ACCEPT."""
    gate=OUT/'TARGET_SCORING_AUTHORIZATION.json'
    if not gate.exists():raise PermissionError('V-DESIGN scoring authorization absent')
    d=json.loads(gate.read_text())
    if d.get('status')!='ACCEPT' or d.get('authorize_full_target_scoring') is not True:raise PermissionError('target scoring not authorized')
    report=Path(d['design_report_path']).resolve()
    if OUT not in report.parents:raise PermissionError('report outside output root')
    if file_sha(report)!=d['design_report_sha256']:raise PermissionError('design report hash changed')
    if 'ACCEPT' not in report.read_text():raise PermissionError('no ACCEPT in design report')
    return d

def full_score():
    gate=require_gate();SEALED.mkdir(parents=True,exist_ok=True)
    from prepare_data import observations,recover_identity,dump_csv
    index=read_csv(OUT/'target_file_index.csv');targets=[];bins_all=[];maxerr=0.;identity_maxerr=0.;start=time.time()
    for rec in index:
        f=Path(rec['path']);raw=f.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=rec['sha256']:raise AssertionError('target input SHA changed')
        import io
        z=np.load(io.BytesIO(raw),allow_pickle=False);mid,org,_=recover_identity(z)
        if (mid,org)!=(rec['model_id'],rec['org_id']):raise AssertionError('target identity changed')
        ell,c,y,v,reasons,nr,err=observations(z['loglik'],z['char_lens'],z['gold'],z['n_options'] if 'n_options' in z else None)
        if int(v.sum())<200:raise AssertionError('target eligibility changed')
        m,b=metrics(c[v],y[v],z['item_ids'][v]);maxerr=max(maxerr,err)
        targets.append(dict(model_id=mid,org_id=org,**m))
        for rr in b:bins_all.append(dict(model_id=mid,org_id=org,**rr))
    dump_csv(SEALED/'targets.csv',targets,['model_id','org_id','Y_cal','Y_sel','n_valid','k_selected'])
    dump_csv(SEALED/'bins.csv',bins_all,['model_id','org_id','bin','n','cbar','ybar','contribution'])
    checks={'status':'CV3_COMPLETE_SEALED','n_models':len(targets),'schema':['model_id','org_id','Y_cal','Y_sel','n_valid','k_selected'],'max_probability_sum_error':maxerr,'metric_unit_tests':selftest(),'targets_sha256':file_sha(SEALED/'targets.csv'),'bins_sha256':file_sha(SEALED/'bins.csv'),'authorization':gate,'runtime_sec':time.time()-start}
    (SEALED/'SCORE_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
    # Parent receives counts, schema, hashes only.
    print(json.dumps({k:checks[k] for k in ['status','n_models','schema','targets_sha256','bins_sha256']}))

def validate_request(req,org_by_model):
    from prepare_data import sha_order
    outer=int(req['outer_fold']);allowed=list(map(str,req['allowed_orgs']));models=list(map(str,req['model_ids']))
    if outer not in range(5) or len(set(models))!=len(models):raise PermissionError('invalid request identifiers')
    split=sha_order(set(org_by_model.values()),5)
    if not set(allowed)<=set(split):raise PermissionError('unknown allowed org')
    if any(split[g]==outer for g in allowed):raise PermissionError('outer holdout org in whitelist')
    if any(m not in org_by_model or org_by_model[m] not in allowed for m in models):raise PermissionError('model outside org whitelist')
    return models

def query(req):
    # Service reads only identity metadata before applying whitelist.
    index=read_csv(OUT/'target_file_index.csv');org_by_model={r['model_id']:r['org_id'] for r in index}
    models=validate_request(req,org_by_model)
    wanted=set(models);result={}
    with (SEALED/'targets.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['model_id'] in wanted:
                if r['org_id']!=org_by_model[r['model_id']]:raise AssertionError('sealed identity mismatch')
                result[r['model_id']]={k:(float(r[k]) if k.startswith('Y_') else r[k]) for k in ['model_id','org_id','Y_cal','Y_sel']}
    if set(result)!=wanted:raise ValueError('missing scored training model')
    log={'time_ns':time.time_ns(),'outer_fold':int(req['outer_fold']),'allowed_orgs':req['allowed_orgs'],'model_ids':models,'context':req.get('context'),'count':len(models),'kind':'training_query'}
    with (OUT/'target_access_log.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
    return [result[m] for m in models]

def scoring_query(req):
    pred=Path(req['prediction_path']).resolve()
    if OUT not in pred.parents:raise PermissionError('predictions outside output root')
    sha=file_sha(pred)
    if sha!=req['prediction_sha']:raise PermissionError('prediction hash mismatch')
    seal=OUT/'prediction.sha256'
    if not seal.exists() or sha not in seal.read_text().split():raise PermissionError('prediction hash not sealed')
    approval=OUT/'SCORING_RELEASE.json'
    if not approval.exists():raise PermissionError('main controller has not released final scoring')
    auth=json.loads(approval.read_text())
    if auth.get('prediction_sha256')!=sha or auth.get('release_final_scoring') is not True:raise PermissionError('release does not match prediction')
    predictions=read_csv(pred)
    models=list(dict.fromkeys(r['model_id'] for r in predictions));wanted=set(models)
    lookup={r['model_id']:r for r in read_csv(SEALED/'targets.csv') if r['model_id'] in wanted}
    if set(lookup)!=wanted:raise AssertionError('missing outcome')
    with (OUT/'target_access_log.jsonl').open('a') as f:f.write(json.dumps({'kind':'final_scoring','prediction_sha256':sha,'n_models':len(models),'time_ns':time.time_ns()})+'\n')
    return [{k:(float(lookup[m][k]) if k.startswith('Y_') else lookup[m][k]) for k in ['model_id','org_id','Y_cal','Y_sel']} for m in models]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['selftest','full-score','query','scoring-query']);args=ap.parse_args()
    if args.command=='selftest':
        result=selftest();OUT.mkdir(parents=True,exist_ok=True);(OUT/'SCORE_SELFTEST.json').write_text(json.dumps(result,indent=2)+'\n');print('metric tests PASS')
    elif args.command=='full-score':full_score()
    else:
        import sys
        req=json.load(sys.stdin);result=query(req) if args.command=='query' else scoring_query(req);json.dump(result,sys.stdout)
if __name__=='__main__':main()
