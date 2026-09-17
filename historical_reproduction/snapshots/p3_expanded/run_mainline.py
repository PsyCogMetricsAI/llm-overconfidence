"""A5: fold-isolated supervised comparisons and source-only recommendations."""
from common import *
from legacy_primitives import group_weights
from predictors import fit_predict,select_configuration
import argparse,subprocess,sys,joblib

def get_targets(k,data,context,root=ROOT):
    from score_api import training_targets
    mids=[r['model_id'] for r in data];orgs=sorted({r['org_id'] for r in data})
    rr=training_targets(outer_fold=k,allowed_orgs=orgs,model_ids=mids,context=context,target='hellaswag')
    if isinstance(rr,dict):rr=rr['rows']
    by={r['model_id']:r for r in rr};assert set(by)==set(mids) and len(rr)==len(mids)
    assert all(by[r['model_id']]['org_id']==r['org_id'] for r in data)
    return {ep:np.array([float(by[m][ep]) for m in mids]) for ep in ['Y_sel','Y_cal']}

def predict_outer(k,root=ROOT,target_provider=None):
    require_design(root)
    q=readjson(root/'A3_qualification.json');assert q['sufficient']
    ready=readjson(root/'features/FEATURES_READY.json')
    for name,h in ready['files'].items():assert digest(root/name)==h,name
    provider=target_provider or get_targets
    dest=root/'predictions';dest.mkdir(exist_ok=True)
    if (dest/f'outer_{k}.csv').exists():raise RuntimeError('outer predictions already exist; do not overwrite')
    directory=root/'features'/f'outer_{k}';sp=readjson(root/'split_manifest.json')['outer'][str(k)]
    train=qualified(rows(directory/'train.csv'));test=qualified(rows(directory/'test.csv'))
    A=set(sp['train_orgs']);H=set(sp['test_orgs'])
    assert {r['org_id'] for r in train}<=A and {r['org_id'] for r in test}<=H
    assert len({r['org_id'] for r in train})>=20 and test
    pr=protocol(root);configs=pr['candidate_configurations'];groups=pr['feature_groups'];endpoints=['Y_sel','Y_cal']
    errors={(g,e):[] for g in groups for e in endpoints};vg=[];vmids=[]
    for j in range(4):
        u=qualified(rows(directory/f'inner{j}_train.csv'));v=qualified(rows(directory/f'inner{j}_valid.csv'))
        U={r['org_id'] for r in u};V={r['org_id'] for r in v}
        assert u and v and not U&V and U|V<=A
        assert all(sp['inner_org_fold'][g]!=j for g in U) and all(sp['inner_org_fold'][g]==j for g in V)
        yu=provider(k,u,f'outer{k}/inner{j}/train',root);yv=provider(k,v,f'outer{k}/inner{j}/validation',root)
        vg.extend(r['org_id'] for r in v);vmids.extend(r['model_id'] for r in v)
        for group,names in groups.items():
            X=design(u,names);Xv=design(v,names)
            for ep in endpoints:
                E=[]
                for conf in configs:
                    with timer(root,'supervised_inner_tuning',outer=k,inner=j,group=group,endpoint=ep,family=conf['family'],configuration=conf['id'],n_train=len(u),n_predict=len(v)) as event:
                        pred,fit=fit_predict(conf,X,yu[ep],[r['org_id'] for r in u],Xv)
                        event.update(fit['timing'])
                    E.append((yv[ep]-pred)**2)
                errors[group,ep].append(np.stack(E))
    assert len(vmids)==len(set(vmids));weights=group_weights(vg)
    yt=provider(k,train,f'outer{k}/final_train',root)
    predrows=[dict(model_id=r['model_id'],org_id=r['org_id'],outer_fold=k,target='hellaswag') for r in test];log=[]
    for group,names in groups.items():
        X=design(train,names);Xt=design(test,names)
        for ep in endpoints:
            loss=np.concatenate(errors[group,ep],axis=1)@weights
            selected={f:select_configuration(configs,loss,f) for f in ['ridge','hgb']};best=select_configuration(configs,loss)
            predictions={};saved_paths={}
            for family,index in selected.items():
                conf=configs[index]
                with timer(root,'supervised_final_fit',outer=k,group=group,endpoint=ep,family=family,role='family',configuration=conf['id'],n_train=len(train),n_predict=len(test)) as event:
                    pred,fit=fit_predict(conf,X,yt[ep],[r['org_id'] for r in train],Xt)
                    event.update(fit['timing'])
                predictions[family]=pred
                p=root/'estimators'/f'outer{k}_{group}_{ep}_{family}.joblib';p.parent.mkdir(exist_ok=True)
                joblib.dump(fit,p);saved_paths[family]=dict(path=str(p.relative_to(root)),sha256=digest(p))
            bestfamily=configs[best]['family'];extra=selected[bestfamily]!=best
            if extra:
                conf=configs[best]
                with timer(root,'supervised_final_fit',outer=k,group=group,endpoint=ep,family=bestfamily,role='deployment_only',configuration=conf['id'],n_train=len(train),n_predict=len(test)) as event:
                    pred,fit=fit_predict(conf,X,yt[ep],[r['org_id'] for r in train],Xt);event.update(fit['timing'])
                p=root/'estimators'/f'outer{k}_{group}_{ep}_best.joblib';joblib.dump(fit,p)
                best_estimator=dict(path=str(p.relative_to(root)),sha256=digest(p));predictions['best']=pred
            else:
                best_estimator=saved_paths[bestfamily];predictions['best']=predictions[bestfamily]
            for family,pred in predictions.items():
                for r,p in zip(predrows,pred):r[f'{group}_{family}_{ep}']=float(p)
            log.append(dict(outer_fold=k,target='hellaswag',group=group,endpoint=ep,inner_losses=loss.tolist(),configs=configs,
              family_selected_indices=selected,best_index=best,best_family=bestfamily,estimators=saved_paths,best_estimator=best_estimator,deployment_extra_fit=extra,
              inner_validation_model_ids=vmids,train_model_ids=[r['model_id'] for r in train],test_model_ids=[r['model_id'] for r in test]))
    write_rows(dest/f'outer_{k}.csv',predrows);dump(dest/f'outer_{k}_selection.json',log)
    print(f'outer {k} sealed local predictions for {len(test)} models; no outer-test Y accessed',flush=True)

def build_recommendations(preds,source):
    bysource={r['model_id']:r for r in source};byorg={}
    for r in preds:byorg.setdefault(r['org_id'],[]).append(r)
    candidates=[];recommendations=[]
    for org,rr in sorted(byorg.items()):
        top=max(float(bysource[r['model_id']]['A']) for r in rr)
        cc=sorted([r for r in rr if float(bysource[r['model_id']]['A'])>=top-.02],key=lambda r:r['model_id'])
        if len(cc)<3:continue
        for r in cc:candidates.append(dict(org_id=org,model_id=r['model_id'],outer_fold=r['outer_fold'],target='hellaswag',source_A=float(bysource[r['model_id']]['A']),org_max_A=top,n_candidates=len(cc)))
        for group in ['B','C','L','F']:
            best=min(cc,key=lambda r:(float(r[f'{group}_best_Y_sel']),r['model_id']))
            recommendations.append(dict(org_id=org,target='hellaswag',outer_fold=best['outer_fold'],group=group,model_id=best['model_id'],predicted_R50=float(best[f'{group}_best_Y_sel']),n_candidates=len(cc)))
    return candidates,recommendations

def seal(root=ROOT):
    require_design(root)
    if (root/'PREDICTION_SEAL.json').exists():raise RuntimeError('seal exists; do not overwrite')
    preds=[];logs=[];pr=protocol(root)
    columns=[f'{g}_{f}_{e}' for g in pr['feature_groups'] for f in ['ridge','hgb','best'] for e in ['Y_sel','Y_cal']]
    for k in range(5):
        rr=rows(root/'predictions'/f'outer_{k}.csv');assert rr and all(int(r['outer_fold'])==k for r in rr)
        for r in rr:
            for c in columns:assert np.isfinite(float(r[c])) and 0<=float(r[c])<=1
        preds+=rr;logs+=readjson(root/'predictions'/f'outer_{k}_selection.json')
    assert len({r['model_id'] for r in preds})==len(preds)
    assert {r['model_id'] for r in preds}=={r['model_id'] for r in rows(root/'evaluation_cohort.csv')}
    preds.sort(key=lambda r:r['model_id']);write_rows(root/'predictions_oof.csv',preds);dump(root/'selection_log.json',logs)
    candidates,recs=build_recommendations(preds,rows(root/'source_features.csv'))
    write_rows(root/'candidates.csv',candidates,['org_id','model_id','outer_fold','target','source_A','org_max_A','n_candidates'])
    write_rows(root/'recommendations.csv',recs,['org_id','target','outer_fold','group','model_id','predicted_R50','n_candidates'])
    artifacts={n:digest(root/n) for n in ['predictions_oof.csv','selection_log.json','candidates.csv','recommendations.csv']}
    dump(root/'PREDICTION_SEAL.json',dict(artifacts=artifacts,n_models=len(preds),n_orgs=len({r['org_id'] for r in preds}),n_family_cells=16,n_best_cells=8,outer_folds=list(range(5)),n_case_orgs=len({r['org_id'] for r in candidates}),all_folds_cells_recommendations_sealed=True,A1_bundle_sha256=digest(root/'A1_OUTPUT_HASHES.json'),DESIGN_RELEASE_sha256=digest(root/'DESIGN_RELEASE.json'),FEATURES_READY_sha256=digest(root/'features/FEATURES_READY.json')))
    print('all 24 OOF cells, candidates and recommendations SHA sealed; final scoring still requires controller release',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['predict','outer','seal']);ap.add_argument('--outer',type=int);a=ap.parse_args()
    require_design()
    if a.stage=='outer':predict_outer(a.outer)
    elif a.stage=='seal':seal()
    else:
        for k in range(5):subprocess.run([sys.executable,__file__,'outer','--outer',str(k)],check=True,env=os.environ.copy())
        seal()
