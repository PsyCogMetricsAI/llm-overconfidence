"""Whole A5 on constructed features/labels, instrumented cheap predictor only."""
from common import *
from legacy_primitives import fixed_folds
import run_mainline as runner
import tempfile,shutil

def run(root=ROOT):
    fixture=Path(tempfile.mkdtemp(prefix='pipeline_',dir=root/'_synthetic'))
    try:
        pr=protocol(root);dump(fixture/'protocol.json',pr);orgs=[f'synthetic_org_{g:02d}' for g in range(30)];outer=fixed_folds(orgs,5);split={'outer_org_fold':outer,'outer':{}}
        data=[]
        for i in range(330):
            g=orgs[i%30];data.append(dict(model_id=f'{g}/m{i}',org_id=g,valid_source_fit=True,valid_summaries=True,context='synthetic',reference_sha256='synthetic',**{n:.5 for n in pr['feature_groups']['F']}))
        for k in range(5):
            A={g for g in orgs if outer[g]!=k};inner=fixed_folds(A,4);split['outer'][str(k)]={'train_orgs':sorted(A),'test_orgs':sorted(set(orgs)-A),'inner_org_fold':inner}
            d=fixture/'features'/f'outer_{k}';write_rows(d/'train.csv',[r for r in data if r['org_id'] in A]);write_rows(d/'test.csv',[r for r in data if r['org_id'] not in A])
            for j in range(4):
                write_rows(d/f'inner{j}_train.csv',[r for r in data if r['org_id'] in A and inner[r['org_id']]!=j])
                write_rows(d/f'inner{j}_valid.csv',[r for r in data if r['org_id'] in A and inner[r['org_id']]==j])
        dump(fixture/'split_manifest.json',split);write_rows(fixture/'source_features.csv',data);write_rows(fixture/'evaluation_cohort.csv',data)
        dump(fixture/'A3_qualification.json',dict(sufficient=True));dump(fixture/'DESIGN_RELEASE.json',dict(synthetic_only=True));dump(fixture/'A1_OUTPUT_HASHES.json',dict(synthetic_only=True))
        paths=list((fixture/'features').glob('outer_*/*.csv'));dump(fixture/'features/FEATURES_READY.json',dict(files={str(p.relative_to(fixture)):digest(p) for p in paths}))
        calls=[];configs=pr['candidate_configurations']
        def provider(k,rr,context,rrroot):
            A=set(split['outer'][str(k)]['train_orgs']);assert {r['org_id'] for r in rr}<=A
            if '/inner' in context:
                j=int(context.split('/')[1][5:]);want_eq=context.endswith('/validation')
                assert all((split['outer'][str(k)]['inner_org_fold'][r['org_id']]==j)==want_eq for r in rr)
            calls.append(dict(outer=k,context=context,n=len(rr)))
            return {ep:np.full(len(rr),.5) for ep in ['Y_sel','Y_cal']}
        def fakefit(config,X,y,org,Xnew):
            i=next(j for j,c in enumerate(configs) if c['id']==config['id'])
            loss=0 if i==9 else (.9e-12 if i==0 else (1.8e-12 if i==8 else .01))
            value=.5+np.sqrt(loss);fit=dict(family='ridge',fit=dict(mean=[0]*X.shape[1],sd=[1]*X.shape[1],active=[True]*X.shape[1],beta=[0]*X.shape[1],intercept=value),timing=dict(fit_wall_seconds=0.,fit_cpu_seconds=0.,prediction_wall_seconds=0.,prediction_cpu_seconds=0.))
            return np.full(len(Xnew),value),fit
        orig_release,orig_fit=runner.require_design,runner.fit_predict
        runner.require_design=lambda r:None;runner.fit_predict=fakefit
        try:
            for k in range(5):runner.predict_outer(k,fixture,target_provider=provider)
            runner.seal(fixture)
        finally:runner.require_design,runner.fit_predict=orig_release,orig_fit
        events=event_list(fixture);logs=readjson(fixture/'selection_log.json');assert len(logs)==40 and all(r['deployment_extra_fit'] for r in logs)
        assert sum(e['stage']=='supervised_inner_tuning' for e in events)==2720
        assert sum(e['stage']=='supervised_final_fit' for e in events)==120
        assert sum(e.get('role')=='deployment_only' for e in events)==40 and len(calls)==45
        pp=rows(fixture/'predictions_oof.csv');assert len(pp)==330 and len([k for k in pp[0] if k.endswith('Y_sel') or k.endswith('Y_cal')])==24
        assert all(float(r['F_best_Y_sel'])!=float(r['F_ridge_Y_sel']) for r in pp)
        seal=readjson(fixture/'PREDICTION_SEAL.json');assert all(digest(fixture/n)==h for n,h in seal['artifacts'].items())
        report=dict(model_id=MODEL_ID,status='PASS',synthetic_models=330,orgs=30,outer_folds=5,inner_fits=2720,family_final_fits=80,additional_deployment_fits=40,white_list_requests=45,OOF_cells=24,candidate_orgs=seal['n_case_orgs'],target_reads=0,live_source_fits=0,fixture=str(fixture),fixture_seal_sha256=digest(fixture/'PREDICTION_SEAL.json'))
        dump(root/'pipeline_selftests.json',report);print(json.dumps(report),flush=True)
    except Exception:
        dump(root/'pipeline_selftests.json',dict(status='FAIL',fixture=str(fixture)));raise
if __name__=='__main__':
    (ROOT/'_synthetic').mkdir(exist_ok=True);run()
