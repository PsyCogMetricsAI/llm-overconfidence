"""A6 frozen-prediction institution inference; final targets only through sealed API."""
from common import *

def percentile(x):
    x=np.asarray(x,float)
    return np.quantile(x,[.025,.975],method='linear').tolist() if x.size and np.isfinite(x).all() else None
def institution_errors(preds,target,cols):
    orgs=sorted({r['org_id'] for r in preds});out={c:[] for c in cols}
    for g in orgs:
        rr=[r for r in preds if r['org_id']==g]
        for col in cols:
            ep='_'.join(col.split('_')[-2:])
            out[col].append(np.mean([(float(target[r['model_id']][ep])-float(r[col]))**2 for r in rr]))
    return orgs,{k:np.array(v) for k,v in out.items()}
def contrast(base,full,draws,orgs,folds):
    lb=float(base.mean());lf=float(full.mean());delta=lb-lf
    db=(base[draws]-full[draws]).mean(1);den=base[draws].mean(1)
    ib=np.divide(db,den,out=np.full_like(db,np.nan),where=den>1e-12)
    byfold=[float((base[np.array([folds[g]==k for g in orgs])]-full[np.array([folds[g]==k for g in orgs])]).mean()) for k in range(5)]
    ci=percentile(db);ici=percentile(ib);I=delta/lb if lb>1e-12 else None
    if I is None or ci is None or ici is None or not np.isfinite(byfold).all():verdict='INCONCLUSIVE'
    elif I>=.05 and ci[0]>0 and sum(d>0 for d in byfold)>=4:verdict='FOLLOWUP_PREDICTIVE_SUPPORT'
    else:verdict='NO_PREDEFINED_PREDICTIVE_SUPPORT'
    return dict(base_MSE=lb,full_MSE=lf,delta=delta,I=I,delta_CI95=ci,I_CI95=ici,outer_fold_delta=byfold,n_positive_outer_folds=sum(d>0 for d in byfold),condition_status=verdict)
def decision_results(candidates,recs,target,primary_status):
    orgs=sorted({r['org_id'] for r in candidates});by={(r['org_id'],r['group']):r for r in recs}
    folds=sorted({int(r['outer_fold']) for r in candidates});n=len(orgs)
    if not n:return dict(status='INSUFFICIENT_CASES',n_case_orgs=0,n_outer_folds=0,groups={},D=None,D_CI95=None,claim='DESCRIPTIVE_ONLY' if primary_status!='FOLLOWUP_PREDICTIVE_SUPPORT' else 'PREDICTION_ONLY'),np.empty((2000,0),int)
    draws=np.random.default_rng(20260914).integers(0,n,size=(2000,n));risk={};regret={}
    for group in ['B','C','L','F']:
        risk[group]=np.array([float(target[by[g,group]['model_id']]['Y_sel']) for g in orgs]);regret[group]=np.array([risk[group][i]-min(float(target[r['model_id']]['Y_sel']) for r in candidates if r['org_id']==g) for i,g in enumerate(orgs)])
    d=risk['C']-risk['F'];D=float(d.mean());ci=percentile(d[draws].mean(1));sufficient=n>=30 and len(folds)>=3
    if not sufficient:status='INSUFFICIENT_CASES'
    elif primary_status=='FOLLOWUP_PREDICTIVE_SUPPORT' and D>=.01 and ci[0]>0:status='FOLLOWUP_DECISION_SUPPORT'
    elif primary_status=='FOLLOWUP_PREDICTIVE_SUPPORT':status='PREDICTION_ONLY'
    else:status='DESCRIPTIVE_ONLY'
    claim='DESCRIPTIVE_ONLY' if primary_status!='FOLLOWUP_PREDICTIVE_SUPPORT' else ('FOLLOWUP_DECISION_SUPPORT' if status=='FOLLOWUP_DECISION_SUPPORT' else 'PREDICTION_ONLY')
    return dict(status=status,claim=claim,n_case_orgs=n,n_outer_folds=len(folds),org_order=orgs,D=D,D_CI95=ci,groups={g:dict(U=float(risk[g].mean()),U_CI95=percentile(risk[g][draws].mean(1)),regret=float(regret[g].mean()),regret_CI95=percentile(regret[g][draws].mean(1))) for g in risk}),draws

def run(root=ROOT):
    require_design(root)
    if (root/'results_comparison.json').exists():raise RuntimeError('results exist; no overwrite')
    from score_api import scoring_targets
    got=scoring_targets(str(root/'PREDICTION_SEAL.json'),digest(root/'PREDICTION_SEAL.json'))
    if isinstance(got,dict):got=got['rows']
    target={r['model_id']:r for r in got};preds=rows(root/'predictions_oof.csv')
    assert all(r['model_id'] in target and target[r['model_id']]['org_id']==r['org_id'] for r in preds)
    pr=protocol(root);cols=[f'{g}_{f}_{e}' for g in pr['feature_groups'] for f in ['ridge','hgb','best'] for e in ['Y_sel','Y_cal']]
    orgs,errors=institution_errors(preds,target,cols);draws=np.random.default_rng(20260914).integers(0,len(orgs),size=(2000,len(orgs)))
    np.save(root/'bootstrap_indices.npy',draws);dump(root/'bootstrap_org_order.json',orgs)
    folds=readjson(root/'split_manifest.json')['outer_org_fold'];matrix=[]
    for col in cols:
        ci=percentile(errors[col][draws].mean(1));matrix.append(dict(target='hellaswag',cell=col,MSE=float(errors[col].mean()),CI95_lower=ci[0],CI95_upper=ci[1],n_models=len(preds),n_orgs=len(orgs)))
    write_rows(root/'effect_matrix.csv',matrix)
    comparisons={}
    for ep in ['Y_sel','Y_cal']:
        for base in ['B','C','L']:
            comparisons[f'F_best_vs_{base}_best/{ep}']=contrast(errors[f'{base}_best_{ep}'],errors[f'F_best_{ep}'],draws,orgs,folds)
        for family in ['ridge','hgb']:
            for base in ['C','L']:comparisons[f'F_{family}_vs_{base}_{family}/{ep}']=contrast(errors[f'{base}_{family}_{ep}'],errors[f'F_{family}_{ep}'],draws,orgs,folds)
    primary=comparisons['F_best_vs_C_best/Y_sel'];decision,ddraws=decision_results(rows(root/'candidates.csv'),rows(root/'recommendations.csv'),target,primary['condition_status'])
    np.save(root/'case_bootstrap_indices.npy',ddraws);dump(root/'decision_results.json',decision)
    result=dict(model_id=MODEL_ID,target='hellaswag',primary='F_best_vs_C_best/Y_sel',prediction_status=primary['condition_status'],decision_status=decision['status'],decision_claim=decision['claim'],n_models=len(preds),n_orgs=len(orgs),comparisons=comparisons,
      interpretation='post-HS followup; only designated primary comparison carries support label; all other condition_status values descriptive, not independent superiority claims',bootstrap_n=2000,CI_percentiles=[2.5,97.5],two_target_consistency='NOT_APPLICABLE',PREDICTION_SEAL_sha256=digest(root/'PREDICTION_SEAL.json'))
    dump(root/'results_comparison.json',result)
    print(json.dumps(dict(prediction=result['prediction_status'],decision=result['decision_status'],primary=primary)),flush=True)

if __name__=='__main__':run()
