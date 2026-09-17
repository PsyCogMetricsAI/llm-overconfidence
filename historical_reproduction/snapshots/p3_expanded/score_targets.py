#!/usr/bin/env python3
"""Independent target administrator. No target outcomes are printed by prepare/selftest.
Production opens fixed targets only after contract/design and exact-context checks.
"""
import argparse,csv,hashlib,json,re,sys,time,resource
from pathlib import Path
import numpy as np
from common import ROOT,digest,readjson,rows,write_rows,dump,qualified,require_design
ARTIFACTS={'predictions_oof.csv','selection_log.json','candidates.csv','recommendations.csv'}

def sethash(values):return hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest()

def metrics(c,y,item_ids):
    c=np.asarray(c,dtype=np.float64); y=np.asarray(y,dtype=np.float64); ids=np.asarray(item_ids).astype(str)
    if c.ndim!=1 or y.shape!=c.shape or ids.shape!=c.shape or not len(c):raise ValueError('metric_shape_or_empty')
    if not(np.isfinite(c).all() and np.isfinite(y).all() and ((c>=0)&(c<=1)).all() and np.isin(y,[0,1]).all()):raise ValueError('invalid_metric_input')
    if len(set(ids))!=len(ids):raise ValueError('duplicate_item_id')
    bins=np.minimum(np.floor(10*c).astype(np.int64),9)
    rel=sum(float((bins==j).sum())/len(c)*float((c[bins==j].mean()-y[bins==j].mean())**2) for j in range(10) if (bins==j).any())
    order=np.lexsort((ids,-c)); k=(len(c)+1)//2
    return dict(Y_cal=rel,Y_sel=float((1-y[order[:k]]).mean()),n_valid=len(c),k_selected=k)

def allowed_request(req,root=ROOT):
    if req.get('target')!='hellaswag':raise PermissionError('unknown target')
    outer=req.get('outer_fold'); context=req.get('context')
    if type(outer)!=int or outer not in range(5) or not isinstance(context,str):raise PermissionError('invalid fold/context')
    split=readjson(root/'split_manifest.json'); meta=split['outer'][str(outer)]
    final=re.fullmatch(r'outer([0-4])/final_train',context)
    inner=re.fullmatch(r'outer([0-4])/inner([0-3])/(train|validation)',context)
    if final and int(final[1])==outer:
        filename='train.csv'; permitted=set(meta['train_orgs'])
    elif inner and int(inner[1])==outer:
        j=int(inner[2]); role=inner[3]; filename=f'inner{j}_{"train" if role=="train" else "valid"}.csv'
        permitted={g for g,v in meta['inner_org_fold'].items() if (int(v)!=j if role=='train' else int(v)==j)}
    else:raise PermissionError('unknown or mismatched context')
    name=f'features/outer_{outer}/{filename}'; ready=readjson(root/'features/FEATURES_READY.json')
    if ready['files'].get(name)!=digest(root/name):raise PermissionError('source feature SHA mismatch')
    data=qualified(rows(root/name)); identities={r['model_id']:r['org_id'] for r in data}
    cohort={r['model_id']:r['org_id'] for r in rows(root/'cohort_manifest.csv')}
    if len(identities)!=len(data) or not data:raise PermissionError('duplicate/empty qualified rows')
    if any(cohort.get(m)!=g or g not in permitted or int(split['outer_org_fold'][g])==outer for m,g in identities.items()):raise PermissionError('qualified feature identity or fold violation')
    models=req.get('model_ids',[]); orgs=req.get('allowed_orgs',[])
    if not isinstance(models,list) or not isinstance(orgs,list) or len(set(models))!=len(models) or len(set(orgs))!=len(orgs):raise PermissionError('invalid request set')
    if set(models)!=set(identities) or set(orgs)!=set(identities.values()):raise PermissionError('request not exact qualified context')
    return models,identities

def ready_targets(root=ROOT):
    require_design(root)
    ready=readjson(root/'sealed/TARGETS_READY.json')
    if ready.get('status')!='A4_COMPLETE_SEALED' or ready['design_sha256']!=digest(root/'DESIGN_RELEASE.json'):raise PermissionError('target readiness/design mismatch')
    if ready['targets_sha256']!=digest(root/'sealed/targets.csv'):raise PermissionError('sealed targets changed')
    if ready['input_manifest_sha256']!=digest(root/'input_manifest.sha256'):raise PermissionError('target input manifest changed')
    for path,h in ready['dependency_sha256'].items():
        if digest(path)!=h:raise PermissionError('scoring dependency changed')
    return ready

def target_subset(models,identities,root=ROOT):
    want=set(models); found={}
    # Only independent administrator process reads sealed file; trainer sees exact subset.
    for r in rows(root/'sealed/targets.csv'):
        if r['model_id'] in want:
            m=r['model_id']
            if m in found or r['org_id']!=identities[m]:raise ValueError('duplicate or mismatched target')
            rr={k:r[k] for k in ('model_id','org_id')};rr.update({k:float(r[k]) for k in ('Y_cal','Y_sel')})
            if not all(np.isfinite(rr[k]) and 0<=rr[k]<=1 for k in ('Y_cal','Y_sel')):raise ValueError('invalid target')
            found[m]=rr
    if set(found)!=want:raise ValueError('target missing')
    return [found[m] for m in models]

def audit(event,root=ROOT):
    with (root/'target_access_log.jsonl').open('a') as f:f.write(json.dumps(dict(time_ns=time.time_ns(),**event))+'\n')

def query(req,root=ROOT):
    ready_targets(root);models,identities=allowed_request(req,root)
    out=target_subset(models,identities,root)
    audit(dict(kind='training_query',outer_fold=req['outer_fold'],context=req['context'],n_models=len(models),requested_models_sha256=sethash(models),returned_models_sha256=sethash([r['model_id'] for r in out]),requested_orgs_sha256=sethash(req['allowed_orgs']),returned_orgs_sha256=sethash(set(identities.values()))),root)
    return out

def validate_seal(req,root=ROOT):
    path=Path(req['seal_path']).resolve()
    if path!=(root/'PREDICTION_SEAL.json').resolve() or digest(path)!=req['seal_sha']:raise PermissionError('seal path/hash mismatch')
    auth=readjson(root/'SCORING_RELEASE.json')
    if auth.get('release_final_scoring') is not True or auth.get('seal_sha256')!=req['seal_sha']:raise PermissionError('no matching scoring release')
    seal=readjson(path)
    if set(seal['artifacts'])!=ARTIFACTS:raise PermissionError('missing sealed artifact')
    for name,h in seal['artifacts'].items():
        if digest(root/name)!=h:raise PermissionError('sealed artifact changed: '+name)
    return seal

def scoring_query(req,root=ROOT):
    ready_targets(root);seal=validate_seal(req,root)
    for field,name in [('A1_bundle_sha256','A1_OUTPUT_HASHES.json'),('DESIGN_RELEASE_sha256','DESIGN_RELEASE.json'),('FEATURES_READY_sha256','features/FEATURES_READY.json')]:
        if seal.get(field)!=digest(root/name):raise PermissionError('sealed provenance mismatch')
    if seal.get('all_folds_cells_recommendations_sealed') is not True:raise PermissionError('incomplete seal')
    predictions=rows(root/'predictions_oof.csv'); identities={}
    cohort={r['model_id']:r['org_id'] for r in rows(root/'cohort_manifest.csv')}
    for r in predictions:
        if cohort.get(r['model_id'])!=r['org_id']:raise PermissionError('prediction identity mismatch')
        identities[r['model_id']]=r['org_id']
    split=readjson(root/'split_manifest.json')['outer_org_fold']
    expected={r['model_id']:r['org_id'] for r in rows(root/'evaluation_cohort.csv')}
    if identities!=expected or len(predictions)!=len(expected):raise PermissionError('OOF cohort mismatch/duplicate')
    columns=[f'{g}_{a}_{e}' for g in ['B','C','L','F'] for a in ['ridge','hgb','best'] for e in ['Y_sel','Y_cal']]
    for r in predictions:
        if r.get('target')!='hellaswag' or int(r['outer_fold'])!=int(split[r['org_id']]):raise PermissionError('OOF fold/target mismatch')
        if not all(np.isfinite(float(r[k])) for k in columns):raise PermissionError('missing or invalid OOF cell')
    if set(int(r['outer_fold']) for r in predictions)!=set(range(5)):raise PermissionError('missing outer fold')
    if seal.get('n_models')!=len(expected) or seal.get('n_orgs')!=len(set(expected.values())) or seal.get('n_family_cells')!=16 or seal.get('n_best_cells')!=8:raise PermissionError('seal counts mismatch')
    models=sorted(identities);result=target_subset(models,identities,root)
    audit(dict(kind='final_scoring',seal_sha256=req['seal_sha'],n_models=len(models),models_sha256=sethash(models)),root)
    return result

def checked_dependency(path,manifest):
    path=Path(path)
    expected=manifest.get(str(path))
    if expected is None or digest(path)!=expected:raise PermissionError('unfrozen or changed dependency: '+str(path))
    return expected

def prepare_targets(root=ROOT):
    """Independent full scoring: old actual searchsorted bins differ at machine
    boundaries from this frozen floor contract, so old target values are not reused.
    """
    require_design(root);start=time.perf_counter();cpu_start=time.process_time()
    old=root.parents[1]/'confidence_validity'
    manifest={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/'input_manifest.sha256').read_text().splitlines()}
    dependencies={str(p):checked_dependency(p,manifest) for p in [old/'target_file_index.csv',old/'code/prepare_data.py']}
    index=rows(old/'target_file_index.csv'); cohort=rows(root/'cohort_manifest.csv')
    identities={r['model_id']:r['org_id'] for r in index}
    if len(identities)!=len(index) or identities!={r['model_id']:r['org_id'] for r in cohort}:raise ValueError('target index/cohort identity mismatch')
    import importlib.util,io
    spec=importlib.util.spec_from_file_location('frozen_observations',old/'code/prepare_data.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    target_dir=root/'sealed';target_dir.mkdir(exist_ok=True)
    output=[];input_checks=[]
    for num,rec in enumerate(index,1):
        if manifest.get(rec['path'])!=rec['sha256']:raise PermissionError('target missing from frozen manifest')
        raw=Path(rec['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=rec['sha256']:raise PermissionError('target input SHA changed')
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            mid,org,_=mod.recover_identity(z)
            if (mid,org)!=(rec['model_id'],rec['org_id']):raise ValueError('target identity mismatch')
            _,c,y,v,_,_,_=mod.observations(z['loglik'],z['char_lens'],z['gold'],z['n_options'] if 'n_options' in z else None)
            value=metrics(c[v],y[v],z['item_ids'][v])
        if value['n_valid']<200:raise ValueError('target eligibility changed')
        if digest(rec['path'])!=rec['sha256']:raise PermissionError('target changed during read')
        output.append(dict(model_id=mid,org_id=org,**value));input_checks.append(dict(path=rec['path'],sha256=rec['sha256']))
        if num%500==0:dump(target_dir/'progress.json',dict(status='RUNNING',completed=num,total=len(index)))
    # Recheck import and metadata after processing. No old target outcomes were read.
    for p,h in dependencies.items():
        if digest(p)!=h:raise PermissionError('dependency changed during scoring')
    write_rows(target_dir/'targets.csv',output)
    dump(target_dir/'INPUT_CHECKS.json',input_checks)
    report=dict(model_id='scientific reproduction',status='A4_COMPLETE_SEALED',method='full fixed-snapshot scoring; floor(10*c) clipped at9; no old target value reuse',n_models=len(output),schema=list(output[0]),targets_sha256=digest(target_dir/'targets.csv'),input_checks_sha256=digest(target_dir/'INPUT_CHECKS.json'),dependency_sha256=dependencies,input_manifest_sha256=digest(root/'input_manifest.sha256'),design_sha256=digest(root/'DESIGN_RELEASE.json'),metric_selftest=selftest(),wall_seconds=time.perf_counter()-start)
    dump(target_dir/'TARGETS_READY.json',report)
    (root/'cost').mkdir(exist_ok=True)
    with (root/'cost/events.jsonl').open('a') as f:f.write(json.dumps(dict(stage='target_scoring',wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu_start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,RSS_scope='target administrator process lifetime high-water mark',n_models=len(output)))+'\n')
    print(json.dumps({k:report[k] for k in ['status','n_models','schema','targets_sha256']}))

def selftest():
    a=metrics([.1,.1,.9,1],[0,1,1,0],['a','b','c','d'])
    assert abs(a['Y_cal']-(.5*.4**2+.5*.45**2))<1e-12 and a['Y_sel']==.5
    assert metrics([.5,.5,.5],[0,1,1],['z','a','b'])['Y_sel']==0
    assert abs(metrics([.2,.8],[0,1],['a','b'])['Y_cal']-.04)<1e-12
    # Check every decimal boundary and adjacent representable floats against
    # the exact frozen scalar floor expression, not an interval approximation.
    import math
    for edge in np.arange(1,10,dtype=float)/10:
        for value in [np.nextafter(edge,-np.inf),edge,np.nextafter(edge,np.inf)]:
            c=np.array([value,edge,.05,.95]);y=np.array([0.,1.,1.,0.]);ids=['a','b','c','d']
            scalar=[min(math.floor(10*float(x)),9) for x in c];ref=0.
            for j in range(10):
                select=np.array(scalar)==j
                if select.any():ref+=select.mean()*(c[select].mean()-y[select].mean())**2
            assert abs(metrics(c,y,ids)['Y_cal']-ref)<1e-15
    return dict(status='PASS',cases=['fixed_edge_REL10','R50_canonical_tie','odd_ceil','all9_decimal_edges_and_both_nextafter_floor_contract'])

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['query','scoring-query','prepare-targets','selftest']);a=p.parse_args()
    if a.command=='prepare-targets':prepare_targets()
    elif a.command=='selftest':print(json.dumps(selftest()))
    else:json.dump(query(json.load(sys.stdin)) if a.command=='query' else scoring_query(json.load(sys.stdin)),sys.stdout)
if __name__=='__main__':main()
