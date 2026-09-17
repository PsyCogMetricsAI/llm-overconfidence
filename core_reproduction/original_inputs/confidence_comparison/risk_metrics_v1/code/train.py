"""E5 fixed-library nested training. Fresh process for each outer fold."""
from common import *
from legacy_primitives import group_weights
from modeling import fit_package,select_configuration,select_package,train_missing
from target_api import training_targets
import argparse,joblib,sys,subprocess
def data_for(k,name,root=ROOT):
    rr=rows(root/'features'/f'outer_{k}'/name)
    frozen=readjson(root/'cohort_contract.json')['fixed_rows'][f'features/outer_{k}/{name}']
    assert [r['model_id'] for r in rr]==frozen['model_ids']
    assert [r['org_id'] for r in rr]==frozen['org_ids']
    return rr
def labels(k,rr,context):
    out=training_targets(k,sorted({r['org_id'] for r in rr}),[r['model_id'] for r in rr],context)
    assert [r['model_id'] for r in out]==[r['model_id'] for r in rr]
    return {e:np.asarray([r[e] for r in out],float) for e in ['Y_sel','Y_cal']}
def outer(k,root=ROOT):
    require_design(root);sr=readjson(root/'SOURCE_RELEASE.json');assert sr['verdict']=='ACCEPT'
    assert sr['features_ready_sha256']==digest(root/'features/FEATURES_READY.json');verify_features(root)
    assert not (root/'predictions'/f'outer_{k}.csv').exists()
    pr=protocol(root);configs=pr['candidate_configurations'];groups=pr['feature_groups'];eps=pr['endpoints']
    train=data_for(k,'train.csv',root);test=data_for(k,'test.csv',root)
    assert len(set(r['org_id'] for r in train))>=20
    errors={(g,e):[] for g in groups for e in eps};vmids=[];vorgs=[];missing_contexts=[]
    for j in range(4):
        u=data_for(k,f'inner{j}_train.csv',root);v=data_for(k,f'inner{j}_valid.csv',root)
        assert not {r['org_id'] for r in u}&{r['org_id'] for r in v}
        yu=labels(k,u,f'outer{k}/inner{j}/train');yv=labels(k,v,f'outer{k}/inner{j}/validation')
        vmids.extend(r['model_id'] for r in v);vorgs.extend(r['org_id'] for r in v)
        missing_contexts.append(dict(context=f'outer{k}/inner{j}/train',all_missing=train_missing(u,pr['metric_ids'])))
        for group,names in groups.items():
            X=design(u,names);Xv=design(v,names);orgs=[r['org_id'] for r in u]
            for ep in eps:
                arr=[]
                for conf in configs:
                    with timer(root,'supervised_inner_tuning',outer=k,inner=j,group=group,endpoint=ep,family=conf['family'],configuration=conf['id'],n_train=len(u),n_predict=len(v)) as event:
                        pred,saved=fit_package(conf,X,yu[ep],orgs,Xv);event.update(saved['model']['timing'])
                    arr.append((yv[ep]-pred)**2)
                errors[group,ep].append(np.stack(arr))
        dump(root/'logs'/f'outer{k}_progress.json',dict(stage='inner',finished_inner_folds=j+1,total_inner_folds=4,nominal_fits=(j+1)*31*2*17,outer=k))
        print(f'outer{k}: inner fold {j+1}/4 complete; all31 packages and2 endpoints',flush=True)
    assert len(vmids)==len(set(vmids));weight=group_weights(vorgs);yt=labels(k,train,f'outer{k}/final_train')
    missing_contexts.append(dict(context=f'outer{k}/final_train',all_missing=train_missing(train,pr['metric_ids'])))
    evaluable={metric:not any(x['all_missing'][metric] for x in missing_contexts) for metric in pr['metric_ids']}
    joint_evaluable=not any(all(x['all_missing'][m] for m in pr['structure_ids']) for x in missing_contexts)
    predictions=[dict(model_id=r['model_id'],org_id=r['org_id'],outer_fold=k,target='hellaswag') for r in test]
    logs=[];best_losses={};package_logs={}
    for group,names in groups.items():
        X=design(train,names);Xt=design(test,names);orgs=[r['org_id'] for r in train]
        for ep in eps:
            loss=np.concatenate(errors[group,ep],axis=1)@weight
            selected={f:select_configuration(configs,loss,f) for f in ['ridge','hgb']};best=select_configuration(configs,loss)
            preds={};paths={}
            for fam,index in selected.items():
                conf=configs[index]
                with timer(root,'supervised_final_fit',outer=k,group=group,endpoint=ep,family=fam,role='family',configuration=conf['id'],n_train=len(train),n_predict=len(test)) as event:
                    pred,saved=fit_package(conf,X,yt[ep],orgs,Xt);event.update(saved['model']['timing'])
                p=root/'estimators'/f'outer{k}_{group}_{ep}_{fam}.joblib';joblib.dump(saved,p)
                preds[fam]=pred;paths[fam]=dict(path=str(p.relative_to(root)),sha256=digest(p))
            fam=configs[best]['family'];extra=selected[fam]!=best
            if extra:
                conf=configs[best]
                with timer(root,'supervised_final_fit',outer=k,group=group,endpoint=ep,family=fam,role='global_only',configuration=conf['id'],n_train=len(train),n_predict=len(test)) as event:
                    pred,saved=fit_package(conf,X,yt[ep],orgs,Xt);event.update(saved['model']['timing'])
                p=root/'estimators'/f'outer{k}_{group}_{ep}_best.joblib';joblib.dump(saved,p)
                bestpath=dict(path=str(p.relative_to(root)),sha256=digest(p));preds['best']=pred
            else:preds['best']=preds[fam];bestpath=paths[fam]
            for alg,pred in preds.items():
                for row,value in zip(predictions,pred):row[f'{group}_{alg}_{ep}']=float(value)
            log=dict(outer_fold=k,group=group,endpoint=ep,inner_losses=loss.tolist(),family_selected_indices=selected,best_index=best,best_configuration=configs[best]['id'],selected_actual_inner_loss=float(loss[best]),estimators=paths,best_estimator=bestpath,deployment_extra_fit=extra,train_model_ids=[r['model_id'] for r in train],inner_validation_model_ids=vmids,test_model_ids=[r['model_id'] for r in test])
            logs.append(log);package_logs[group,ep]=log;best_losses[group,ep]=float(loss[best])
        print(f'outer{k}: final package {group} complete',flush=True)
    proc_logs=[]
    for procedure,names in pr['procedures'].items():
        eligible={n:True if n in ['B0','Bd'] else evaluable[n[2:]] for n in names}
        for ep in eps:
            losses={n:best_losses[n,ep] for n in names};chosen=select_package(names,losses,eligible)
            for row in predictions:row[f'{procedure}_best_{ep}']=row[f'{chosen}_best_{ep}']
            proc_logs.append(dict(outer_fold=k,procedure=procedure,endpoint=ep,chosen_package=chosen,package_actual_inner_losses=losses,eligible_packages=eligible,best_estimator=package_logs[chosen,ep]['best_estimator'],no_additional_fit=True))
    write_rows(root/'predictions'/f'outer_{k}.csv',predictions)
    dump(root/'predictions'/f'outer_{k}_selection.json',dict(packages=logs,procedures=proc_logs,source_evaluable=evaluable,joint_evaluable=joint_evaluable,missing_contexts=missing_contexts,code_release_sha256=digest(root/'CODE_RELEASE.json'),source_release_sha256=digest(root/'SOURCE_RELEASE.json')))
    dump(root/'logs'/f'outer{k}_progress.json',dict(stage='COMPLETE',outer=k,n_models=len(test),prediction_columns=190))
    print(f'outer{k}:190 prediction columns saved; no outer-test Y read',flush=True)
def recommendation_rows(preds,root=ROOT):
    pr=protocol(root);methods=list(pr['feature_groups'])+list(pr['procedures'])
    source={r['model_id']:r for r in rows(root/'evaluation_cohort.csv')};byorg={}
    for r in preds:byorg.setdefault(r['org_id'],[]).append(r)
    cc=[];rec=[]
    for org,rr in sorted(byorg.items()):
        top=max(float(source[r['model_id']]['A']) for r in rr)
        candidates=sorted([r for r in rr if float(source[r['model_id']]['A'])>=top-.02],key=lambda r:r['model_id'])
        if len(candidates)<3:continue
        cc.extend(dict(org_id=org,model_id=r['model_id'],outer_fold=r['outer_fold'],source_A=float(source[r['model_id']]['A']),n_candidates=len(candidates)) for r in candidates)
        for method in methods:
            best=min(candidates,key=lambda r:(float(r[f'{method}_best_Y_sel']),r['model_id']))
            rec.append(dict(org_id=org,outer_fold=best['outer_fold'],method=method,model_id=best['model_id'],predicted_R50=float(best[f'{method}_best_Y_sel']),n_candidates=len(candidates)))
    return cc,rec
def seal(root=ROOT):
    require_design(root);pr=protocol(root);assert not (root/'PREDICTION_SEAL.json').exists()
    cols=[f'{g}_{a}_{e}' for g in pr['feature_groups'] for a in ['ridge','hgb','best'] for e in pr['endpoints']]+[f'{g}_best_{e}' for g in pr['procedures'] for e in pr['endpoints']]
    assert len(cols)==len(set(cols))==190
    preds=[];logs=[]
    for k in range(5):
        rr=rows(root/'predictions'/f'outer_{k}.csv');assert rr
        for r in rr:
            assert int(r['outer_fold'])==k
            assert all(np.isfinite(float(r[c])) and 0<=float(r[c])<=1 for c in cols)
        preds.extend(rr);logs.append(readjson(root/'predictions'/f'outer_{k}_selection.json'))
    expected={r['model_id']:r['org_id'] for r in rows(root/'evaluation_cohort.csv')}
    assert len(preds)==len(expected)==len({r['model_id'] for r in preds})
    assert {r['model_id']:r['org_id'] for r in preds}==expected
    preds.sort(key=lambda r:r['model_id']);write_rows(root/'predictions_oof.csv',preds);dump(root/'selection_log.json',logs)
    candidates,recs=recommendation_rows(preds,root)
    write_rows(root/'candidates.csv',candidates,['org_id','model_id','outer_fold','source_A','n_candidates'])
    write_rows(root/'recommendations.csv',recs,['org_id','outer_fold','method','model_id','predicted_R50','n_candidates'])
    names=['predictions_oof.csv','selection_log.json','candidates.csv','recommendations.csv']
    dump(root/'PREDICTION_SEAL.json',dict(artifacts={n:digest(root/n) for n in names},n_prediction_columns=190,n_models=len(preds),n_orgs=len(set(expected.values())),n_case_orgs=len({r['org_id'] for r in candidates}),CODE_RELEASE_sha256=digest(root/'CODE_RELEASE.json'),source_release_sha256=digest(root/'SOURCE_RELEASE.json'),E1_bundle_sha256=digest(root/'E1_BUNDLE.json')))
    print('All190 columns, selection logs and33-method recommendations sealed; awaiting V-SEAL',flush=True)
def run_all(root=ROOT):
    require_design(root);start=time.monotonic();pending=list(range(5));running={};complete=[]
    assert not any((root/'predictions'/f'outer_{k}.csv').exists() for k in pending),'Use explicit verified resume after a prior attempt'
    try:
        while pending or running:
            while pending and len(running)<3:
                k=pending.pop(0);stream=(root/'logs'/f'outer{k}.log').open('w')
                proc=subprocess.Popen([sys.executable,__file__,'outer','--outer',str(k)],stdout=stream,stderr=subprocess.STDOUT)
                running[k]=(proc,stream);print(f'E5 launched outer{k} pid={proc.pid}',flush=True)
            for k,(proc,stream) in list(running.items()):
                code=proc.poll()
                if code is not None:
                    stream.close();del running[k]
                    if code:raise RuntimeError(f'outer{k} failed with exit {code}; inspect preserved log')
                    complete.append(k);print(f'E5 completed outer{k}',flush=True)
            elapsed=time.monotonic()-start
            dump(root/'logs/E5_progress.json',dict(elapsed_seconds=elapsed,completed_outer_folds=complete,running={str(k):p.pid for k,(p,_) in running.items()},pending=pending))
            if elapsed>43200:raise RuntimeError('H-BUDGET: E5 wall time exceeds12 hours')
            if pending or running:time.sleep(1)
        seal(root)
    finally:
        for proc,stream in running.values():
            if proc.poll() is None:proc.terminate()
            stream.close()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['outer','seal','all']);ap.add_argument('--outer',type=int);a=ap.parse_args()
    with timer(ROOT,'E5_'+a.stage,outer=a.outer):
        if a.stage=='outer':outer(a.outer)
        elif a.stage=='all':run_all()
        else:seal()
