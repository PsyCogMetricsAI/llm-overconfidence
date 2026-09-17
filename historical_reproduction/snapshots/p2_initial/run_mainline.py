#!/usr/bin/env python3
"""CV1/CV2/CV4. Target reads exclusively via independent score_api subprocess.
prepare and synthetic tests allowed before V-DESIGN; all real fits require release.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,csv,json,sys,time,subprocess,hashlib
from pathlib import Path
import numpy as np
from cv_core import *
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'contracts/PROTOCOL.md'

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def load_source():
    z=np.load(ROOT/'source_arrays.npz',allow_pickle=False);data={k:z[k] for k in z.files}
    assert data['ell'].shape==data['c'].shape==data['valid'].shape==data['y'].shape
    assert len(set(data['model_ids']))==len(data['model_ids'])
    itemfold=fixed_folds(data['item_ids'],2)
    assert all(int(h)==itemfold[str(i)] for i,h in zip(data['item_ids'],data['source_half']))
    base={}
    with (ROOT/'source_B_features.csv').open() as f:
        for r in csv.DictReader(f):base[r['model_id']]={k:float(r[k]) for k in BASE}
    assert set(base)==set(map(str,data['model_ids']))
    return data,base

def ensure_sample(data,context):
    orgs=sorted(set(map(str,data['org_ids'])));outer=fixed_folds(orgs,5)
    counts=[sum(outer[g]!=k for g in orgs) for k in range(5)]
    reasons=[]
    if len(data['model_ids'])<300:reasons.append('MODELS_LT300')
    if len(orgs)<30:reasons.append('ORGS_LT30')
    if min(counts)<20:reasons.append('OUTER_TRAIN_ORGS_LT20')
    if reasons:
        dump(ROOT/'INSUFFICIENT_SAMPLE.json',dict(verdict='INCONCLUSIVE',status='INSUFFICIENT_SAMPLE',context=context,reasons=reasons,n_models=len(data['model_ids']),n_orgs=len(orgs),outer_train_org_counts=counts,unrun=['CV2_full_fits','CV4_predictions','CV6_target_scoring']))
        raise RuntimeError('INSUFFICIENT_SAMPLE: '+','.join(reasons))

def code_hashes():return {p.name:digest(p) for p in sorted((ROOT/'code').glob('*.py'))}
def require_release():
    p=ROOT/'DESIGN_RELEASE.json'
    if not p.exists():raise RuntimeError('V-DESIGN release absent; real fits/targets forbidden')
    r=json.loads(p.read_text());assert r['verdict']=='ACCEPT';assert r['plan_sha256']==digest(PLAN)
    for name,sha in r['code_sha256'].items():assert digest(ROOT/'code'/name)==sha,f'code changed since review: {name}'
    for name in ['cv_core.py','run_mainline.py','score_mainline.py','score_api.py']:assert name in r['code_sha256']
    return r

def prepare():
    data,_=load_source();orgs=sorted(set(map(str,data['org_ids'])));outer=fixed_folds(orgs,5)
    splits={'seed':SEED,'outer_org_fold':outer,'source_item_half':{str(i):int(v) for i,v in zip(data['item_ids'],data['source_half'])},'outer':{}}
    for k in range(5):
        A={g for g in orgs if outer[g]!=k};H=set(orgs)-A;inner=fixed_folds(A,4)
        splits['outer'][str(k)]={'train_orgs':sorted(A),'test_orgs':sorted(H),'inner_org_fold':inner,'aux_for_outer_training':fixed_folds(A,4),'aux_for_inner_training':{str(j):fixed_folds({g for g in A if inner[g]!=j},4) for j in range(4)}}
    dump(ROOT/'splits.json',splits)
    config={'model_id':'scientific reproduction','plan_sha256':digest(PLAN),'direction':'ARC_to_HellaSwag','models':{'base':BASE,'full':FULL},'endpoints':['Y_cal','Y_sel'],'lambdas':LAMBDAS.tolist(),'outer_folds':5,'inner_folds':4,'aux_training_features_folds':4,'source_sha256':digest(ROOT/'source_arrays.npz'),'source_B_sha256':digest(ROOT/'source_B_features.csv'),'baseline_item_scope':'all ARC valid items F union C','seed':SEED,'threads':1,'cohort_models':len(data['model_ids']),'cohort_orgs':len(orgs),'max_reference_fits':105,'bootstrap_n':2000,'ci_percentiles':[1.25,98.75]}
    dump(ROOT/'run_config.json',config)
    dump(ROOT/'access_protocol.json',{'outer_process_isolation':True,'target_access':'score_api.training_targets -> independent CLI with outer holdout exclusion and exact model/allowed-org whitelist','inner_validation':'May request V outcomes only for inner validation loss; V is within outer training A','outer_holdout':'No H targets read until all five prediction files concatenated and SHA sealed','full_target_reader':'independent scoring process only','reference_cache_key':'SHA of sorted exact reference org set; source SHA verified','no_recursive_tuning':True,'release_file':'DESIGN_RELEASE.json, written by independent reviewer/main controller'})
    dump(ROOT/'resource_plan.json',{'stage':'pre-design','max_reference_fits':105,'source_dimensions':list(data['ell'].shape),'source_F_items':int(np.sum(data['source_half']==0)),'pilot':'After V-DESIGN ACCEPT, exactly outer-0 training reference, timed and cached; no target outcomes. Estimate 105*reference wall time + 300 seconds ridge/io. If >12h STOP G-CV-SCOPE.','threads':1,'memory':'One source matrix plus masked ALS reference arrays; no GPU; caches store difficulty vectors not response arrays.'})
    ensure_sample(data,'prepare')
    print(json.dumps(config))

def write_rows(path,rows):
    if not rows:raise ValueError('NO_ROWS')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def read_rows(path):
    out=[]
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            r['valid_source_fit']=r['valid_source_fit']=='True'
            for key in FULL:
                if r.get(key,'')!='':r[key]=float(r[key])
            out.append(r)
    return out

def pilot():
    require_release();data,base=load_source();ensure_sample(data,'pilot');split=json.loads((ROOT/'splits.json').read_text())['outer']['0'];fe=SourceFeatures(data,base,ROOT/'features/reference_cache',digest(ROOT/'source_arrays.npz'))
    stamp=dict(source_sha256=digest(ROOT/'source_arrays.npz'),algorithm_sha256=digest(ROOT/'code/cv_core.py'),plan_sha256=digest(PLAN),reference_sha256=set_sha(split['train_orgs']))
    saved=ROOT/'pilot_timing.json'
    if saved.exists():
        report=json.loads(saved.read_text())
        if all(report.get(k)==v for k,v in stamp.items()):
            if report['scope_gate']:raise RuntimeError('G-CV-SCOPE: predicted >12h')
            print(json.dumps(dict(report,reused_original_timing=True)));return
    start=time.monotonic();key,b=fe.reference(split['train_orgs']);secs=time.monotonic()-start;est=105*secs+300
    report=dict(stamp,seconds=secs,conservative_total_estimate_seconds=est,threshold_seconds=43200,scope_gate=est>43200,target_reads=0)
    dump(ROOT/'pilot_timing.json',report);print(json.dumps(report))
    if est>43200:raise RuntimeError('G-CV-SCOPE: predicted >12h')

def fit_outer(k):
    require_release();data,base=load_source();ensure_sample(data,'fit_outer');s=json.loads((ROOT/'splits.json').read_text())['outer'][str(k)];A=set(s['train_orgs']);H=set(s['test_orgs'])
    if len(A)<20:raise ValueError('INSUFFICIENT_SAMPLE: outer training orgs <20')
    out=ROOT/'features'/f'outer_{k}';out.mkdir(parents=True,exist_ok=True)
    fe=SourceFeatures(data,base,ROOT/'features/reference_cache',digest(ROOT/'source_arrays.npz'))
    write_rows(out/'train.csv',fe.crossfit(A,f'outer{k}/train'))
    write_rows(out/'test.csv',fe.project(H,A,f'outer{k}/test'))
    inner=s['inner_org_fold']
    for j in range(4):
        U={g for g in A if inner[g]!=j};V=A-U
        write_rows(out/f'inner{j}_train.csv',fe.crossfit(U,f'outer{k}/inner{j}/train'))
        write_rows(out/f'inner{j}_valid.csv',fe.project(V,U,f'outer{k}/inner{j}/valid'))
    dump(out/'fit_diagnostics.json',{'outer_fold':k,'reference_results':fe.reference_log,'projection_contexts':fe.projection_log})
    print(f'outer {k}: source features complete; references {len(fe.reference_log)}',flush=True)

def targets(k,rows,context):
    # Import wrapper only; no target path is accepted or opened by this module.
    from score_api import training_targets
    orgs=sorted({r['org_id'] for r in rows});mids=[r['model_id'] for r in rows]
    data=training_targets(outer_fold=k,allowed_orgs=orgs,model_ids=mids,context=context)
    if isinstance(data,dict):data=data['rows']
    by={r['model_id']:r for r in data};assert set(by)==set(mids)
    for r in rows:assert by[r['model_id']]['org_id']==r['org_id']
    return {ep:np.array([by[m][ep] for m in mids],float) for ep in ['Y_cal','Y_sel']}

def validate_feature_sample():
    test=[];train_org_counts={};excluded=[]
    for k in range(5):
        d=ROOT/'features'/f'outer_{k}'
        testrows=read_rows(d/'test.csv');test+=qualified(testrows)
        train_org_counts[str(k)]=len({r['org_id'] for r in qualified(read_rows(d/'train.csv'))})
        excluded.extend(dict(model_id=r['model_id'],org_id=r['org_id'],outer_fold=k,exclusion_reason=r['exclusion_reason']) for r in testrows if not r['valid_source_fit'])
    reasons=[]
    if len(test)<300:reasons.append('QUALIFIED_MODELS_LT300')
    if len({r['org_id'] for r in test})<30:reasons.append('QUALIFIED_ORGS_LT30')
    if min(train_org_counts.values())<20:reasons.append('QUALIFIED_OUTER_TRAIN_ORGS_LT20')
    report=dict(context='CV2_source_qualification',n_models=len(test),n_orgs=len({r['org_id'] for r in test}),outer_train_org_counts=train_org_counts,sample_sufficient=not reasons,reasons=reasons)
    dump(ROOT/'CV2_qualification.json',report);dump(ROOT/'source_exclusions.json',excluded)
    if reasons:
        dump(ROOT/'INSUFFICIENT_SAMPLE.json',dict(report,verdict='INCONCLUSIVE',status='INSUFFICIENT_SAMPLE',unrun=['CV4_predictions','CV6_target_scoring']))
        raise RuntimeError('INSUFFICIENT_SAMPLE: '+','.join(reasons))

def predict_outer(k):
    require_release();validate_feature_sample();directory=ROOT/'features'/f'outer_{k}';split=json.loads((ROOT/'splits.json').read_text())['outer'][str(k)]
    A=set(split['train_orgs']);H=set(split['test_orgs']);train=qualified(read_rows(directory/'train.csv'));test=qualified(read_rows(directory/'test.csv'))
    assert {r['org_id'] for r in train}<=A and {r['org_id'] for r in test}<=H
    if len({r['org_id'] for r in train})<20:raise ValueError('INSUFFICIENT_SAMPLE: qualified outer training orgs <20')
    if not test:raise ValueError('INSUFFICIENT_SAMPLE: no eligible outer test models')
    losses={(model,ep):[] for model in ['base','full'] for ep in ['Y_cal','Y_sel']};orglabels=[];all_validation_ids=[]
    for j in range(4):
        u=qualified(read_rows(directory/f'inner{j}_train.csv'));v=qualified(read_rows(directory/f'inner{j}_valid.csv'))
        if not u or not v:raise ValueError('INSUFFICIENT_SAMPLE: empty inner qualified set')
        U={r['org_id'] for r in u};V={r['org_id'] for r in v};assert not U&V and U|V<=A
        yu=targets(k,u,f'outer{k}/inner{j}/train');yv=targets(k,v,f'outer{k}/inner{j}/validation')
        orglabels.extend(r['org_id'] for r in v);all_validation_ids.extend(r['model_id'] for r in v)
        for model,names in [('base',BASE),('full',FULL)]:
            for ep in ['Y_cal','Y_sel']:
                errs=[]
                for lam in LAMBDAS:
                    pred,_=ridge_fit_predict(design(u,names),yu[ep],[r['org_id'] for r in u],design(v,names),lam)
                    errs.append((yv[ep]-pred)**2)
                losses[model,ep].append(np.stack(errs))
    # One common fixed inner validation model set for every lambda and model.
    assert len(all_validation_ids)==len(set(all_validation_ids))
    vg=group_weights(orglabels);ytrain=targets(k,train,f'outer{k}/final_train');predrows=[dict(model_id=r['model_id'],org_id=r['org_id'],outer_fold=k) for r in test];selections=[]
    for model,names in [('base',BASE),('full',FULL)]:
        for ep in ['Y_cal','Y_sel']:
            L=np.concatenate(losses[model,ep],axis=1)@vg;idx=select_lambda(L);lam=LAMBDAS[idx]
            pred,fit=ridge_fit_predict(design(train,names),ytrain[ep],[r['org_id'] for r in train],design(test,names),lam)
            for row,value in zip(predrows,pred):row[f'{model}_{ep}']=float(value)
            selections.append(dict(outer_fold=k,model=model,endpoint=ep,selected_lambda=float(lam),inner_losses=L.tolist(),inner_validation_model_ids=all_validation_ids,train_model_ids=[r['model_id'] for r in train],test_model_ids=[r['model_id'] for r in test],fit=fit))
    out=ROOT/'predictions';out.mkdir(exist_ok=True);write_rows(out/f'outer_{k}.csv',predrows);dump(out/f'outer_{k}_selection.json',selections)
    print(f'outer {k}: predicted {len(test)} held-out models; no held-out outcomes read',flush=True)

def seal():
    require_release();rows=[];selections=[];ex=[]
    for k in range(5):
        with (ROOT/'predictions'/f'outer_{k}.csv').open() as f:rows.extend(list(csv.DictReader(f)))
        selections+=json.loads((ROOT/'predictions'/f'outer_{k}_selection.json').read_text())
        ex.extend([dict(model_id=r['model_id'],org_id=r['org_id'],outer_fold=k,exclusion_reason=r['exclusion_reason']) for r in read_rows(ROOT/'features'/f'outer_{k}'/'test.csv') if not r['valid_source_fit']])
    assert len({r['model_id'] for r in rows})==len(rows)
    rows.sort(key=lambda r:r['model_id']);write_rows(ROOT/'predictions_oof.csv',rows);dump(ROOT/'selection_log.json',selections);dump(ROOT/'source_exclusions.json',ex)
    sha=digest(ROOT/'predictions_oof.csv');(ROOT/'prediction.sha256').write_text(sha+'  predictions_oof.csv\n')
    status={'predictions_sha256':sha,'n_models':len(rows),'n_orgs':len({r['org_id'] for r in rows}),'all_five_outer_folds_present':set(int(r['outer_fold']) for r in rows)==set(range(5)),'all_predictions_sealed_before_scoring':True,'sample_sufficient':len(rows)>=300 and len({r['org_id'] for r in rows})>=30}
    dump(ROOT/'PREDICTION_SEAL.json',status);print(json.dumps(status))

def collect_diagnostics():
    out={'outer_folds':{},'n_distinct_references':0};refs=set()
    for k in range(5):
        v=json.loads((ROOT/'features'/f'outer_{k}'/'fit_diagnostics.json').read_text());out['outer_folds'][str(k)]=v;refs.update(v['reference_results'])
    out['n_distinct_references']=len(refs);dump(ROOT/'fit_diagnostics.json',out);validate_feature_sample()

def subprocess_stage(stage,k=None):
    cmd=[sys.executable,str(Path(__file__).resolve()),stage]
    if k is not None:cmd+=['--outer',str(k)]
    subprocess.run(cmd,check=True,env=os.environ.copy())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','pilot','fit-outer','predict-outer','features','predict','seal']);ap.add_argument('--outer',type=int,choices=range(5));args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='pilot':pilot()
    elif args.stage=='fit-outer':fit_outer(args.outer)
    elif args.stage=='predict-outer':predict_outer(args.outer)
    elif args.stage=='features':
        require_release();pilot()
        for k in range(5):subprocess_stage('fit-outer',k)
        collect_diagnostics()
    elif args.stage=='predict':
        require_release()
        for k in range(5):subprocess_stage('predict-outer',k)
        seal()
    else:seal()
