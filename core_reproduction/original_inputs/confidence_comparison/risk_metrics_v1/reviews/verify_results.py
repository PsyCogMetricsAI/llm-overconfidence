"""Independent V-RESULT recomputation; no production statistical imports.

Run only after SCORING_RELEASE plus completed E6/E7 artifacts. Target access uses
the gated administrator process and is recorded as a post-seal aggregate read.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import csv,hashlib,json,subprocess,time
import numpy as np
W=Path(__file__).resolve().parents[1]
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
errors={}
def eq(a,b,label,tol=1e-8):
    if a is None or b is None:
        assert a is None and b is None,(label,a,b)
        return
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape,(label,a.shape,b.shape)
    assert np.array_equal(np.isfinite(a),np.isfinite(b)),(label,'finite pattern')
    use=np.isfinite(a);delta=float(np.max(abs(a[use]-b[use]))) if use.any() else 0.
    errors[label.split(':')[0]]=max(errors.get(label.split(':')[0],0.),delta)
    assert delta<=tol,(label,delta)
def bootstrap_weights(n,seed=20260914):
    ix=np.random.default_rng(seed).integers(0,n,size=(2000,n))
    count=np.array([np.bincount(row,minlength=n) for row in ix],float)/n
    return ix,count
def percentile(boot):return np.quantile(boot,[.025,.975],axis=0,method='linear')

def run():
    start=time.monotonic();pr=js(W/'protocol.json');seal=js(W/'PREDICTION_SEAL.json');release=js(W/'SCORING_RELEASE.json')
    assert release['verdict']=='ACCEPT' and release['seal_sha256']==h(W/'PREDICTION_SEAL.json')
    assert seal['CODE_RELEASE_sha256']==h(W/'CODE_RELEASE.json') and seal['source_release_sha256']==h(W/'SOURCE_RELEASE.json')
    for name,sha in seal['artifacts'].items():assert h(W/name)==sha,name
    complete=js(W/'results/E6_COMPLETE.json')
    for name,sha in complete['files'].items():assert h(W/name)==sha,name
    warm=js(W/'cost/warm_summary.json');cost=js(W/'results/cost_report.json')
    assert cost['warm_report_sha256']==h(W/'cost/warm_summary.json')
    # Administrator performs the existing seal/source/code checks and logs access.
    proc=subprocess.run([pr['legacy_python'],str(W/'code/target_admin.py'),'score'],
        input=json.dumps({'seal_sha256':h(W/'PREDICTION_SEAL.json')}),text=True,capture_output=True,check=True)
    target=json.loads(proc.stdout);by={r['model_id']:r for r in target};oo=rows(W/'predictions_oof.csv')
    assert set(by)=={r['model_id'] for r in oo} and len(oo)==len(by)==6666
    assert all(by[r['model_id']]['org_id']==r['org_id'] for r in oo)
    orgs=sorted({r['org_id'] for r in oo});gi={g:i for i,g in enumerate(orgs)};groups=np.array([gi[r['org_id']] for r in oo]);n=np.bincount(groups)
    gf=np.array([int(next(r['outer_fold'] for r in oo if r['org_id']==g)) for g in orgs]);assert len(orgs)==1324
    ix,bw=bootstrap_weights(len(orgs),pr['seed']);methods=list(pr['feature_groups'])+list(pr['procedures']);logs=js(W/'selection_log.json')
    matrix={(r['endpoint'],r['group'],r['algorithm']):r for r in rows(W/'results/effect_matrix.csv')};assert len(matrix)==190
    primary=js(W/'results/PRIMARY_RESULTS.json');secondary=js(W/'results/SECONDARY_RESULTS.json');statuses=[]
    for ep in pr['endpoints']:
        keys=[(g,a) for g in pr['feature_groups'] for a in ['ridge','hgb','best']]+[(g,'best') for g in pr['procedures']]
        p=np.array([[float(r[f'{g}_{a}_{ep}']) for g,a in keys] for r in oo]);y=np.array([by[r['model_id']][ep] for r in oo]);loss=(p-y[:,None])**2
        gl=np.zeros((len(orgs),len(keys)));np.add.at(gl,groups,loss);gl/=n[:,None];total=gl.mean(0);boot=bw@gl;ci=percentile(boot)
        for j,(g,a) in enumerate(keys):
            stored=matrix[ep,g,a];eq([total[j],ci[0,j],ci[1,j]],[float(stored[k]) for k in ['MSE','CI95_low','CI95_high']],'effect_matrix:'+ep+g+a)
        ciidx={key:j for j,key in enumerate(keys)};d=[];base=[];fold=[];eligible=[]
        for contrast in pr['contrasts']:
            b,a=ciidx[contrast['baseline'],'best'],ciidx[contrast['method'],'best'];difference=gl[:,b]-gl[:,a]
            d.append(difference);base.append(total[b]);fold.append([difference[gf==f].mean() for f in range(5)])
            metric=contrast['metric'];eligible.append(all(l['source_evaluable'][metric] for l in logs) if metric else all(l['joint_evaluable'] for l in logs) if contrast['method']=='J' else True)
        d=np.array(d).T;base=np.array(base);fold=np.array(fold).T;delta=d.mean(0);bootd=bw@d
        if ep=='Y_sel':
            assert primary['n_contrasts']==len(pr['contrasts'])==31 and primary['n_models']==len(oo) and primary['n_orgs']==len(orgs)
            se=np.std(d,axis=0,ddof=1)/np.sqrt(len(orgs));zero=(d==0).all(0);calc=(se>1e-12)&(base>1e-12)
            t=np.max(abs((bootd[:,calc]-delta[calc])/se[calc]),axis=1) if calc.any() else np.zeros(2000)
            critical=float(np.quantile(t,.95,method='linear')) if calc.any() else None
            eq(critical,primary['critical_value'],'critical');assert js(W/'results/bootstrap_org_order.json')==orgs
            assert np.array_equal(np.load(W/'results/bootstrap_indices.npy'),ix)
            eq(np.load(W/'results/paired_group_differences.npy'),d,'group_differences');eq(np.load(W/'results/bootstrap_delta.npy'),bootd,'bootstrap_delta');eq(np.load(W/'results/bootstrap_max_statistics.npy'),t,'max_statistics')
            for j,(co,stored) in enumerate(zip(pr['contrasts'],primary['contrasts'])):
                assert all(stored[k]==v for k,v in co.items());reason='';rel=None;bounds=[None,None]
                if base[j]<=1e-12:status='INCONCLUSIVE';reason='BASELINE_LOSS_LE1E12'
                elif not calc[j] and not zero[j]:status='INCONCLUSIVE';reason='DEGENERATE_VARIANCE'
                else:
                    rel=delta[j]/base[j];bounds=[0.,0.] if zero[j] else [delta[j]-critical*se[j],delta[j]+critical*se[j]]
                    if not eligible[j]:status='INCONCLUSIVE';reason='ALL_TRAIN_MISSING'
                    elif rel>=.05 and bounds[0]>0 and (fold[:,j]>0).sum()>=4:status='EXPLORATORY_CANDIDATE'
                    else:status='NO_PREDEFINED_INCREMENT'
                eq(delta[j],stored['delta'],'primary_delta:'+co['id']);eq(base[j],stored['baseline_loss'],'primary_baseline:'+co['id']);eq(se[j],stored['standard_error'],'primary_se:'+co['id']);eq(rel,stored['relative_improvement'],'primary_relative:'+co['id'])
                for bound,actual in zip(bounds,stored['simultaneous_CI']):eq(bound,actual,'simultaneous_CI:'+co['id'])
                eq(fold[:,j],stored['outer_fold_delta'],'fold_delta:'+co['id']);assert int((fold[:,j]>0).sum())==stored['positive_outer_folds']
                assert (status,reason)==(stored['status'],stored['reason']),(co['id'],status,stored['status']);statuses.append({'id':co['id'],'status':status})
        else:
            bounds=percentile(bootd);assert len(secondary['contrasts'])==31
            for j,(co,stored) in enumerate(zip(pr['contrasts'],secondary['contrasts'])):
                assert all(stored[k]==v for k,v in co.items());eq(delta[j],stored['delta'],'secondary_delta:'+co['id']);eq(delta[j]/base[j] if base[j]>1e-12 else None,stored['relative_improvement'],'secondary_relative:'+co['id']);eq(bounds[:,j],stored['CI95'],'secondary_CI:'+co['id']);eq(fold[:,j],stored['outer_fold_delta'],'secondary_fold:'+co['id'])
    decision=js(W/'results/selection_results.json');cc=rows(W/'candidates.csv');recs=rows(W/'recommendations.csv');caseorg=sorted({r['org_id'] for r in cc});byrec={(r['org_id'],r['method']):r['model_id'] for r in recs}
    assert len(byrec)==len(recs)==33*len(caseorg)
    if caseorg:
        caseix,casebw=bootstrap_weights(len(caseorg),pr['seed']);assert np.array_equal(np.load(W/'results/case_bootstrap_indices.npy'),caseix)
        risks=np.array([[by[byrec[g,m]]['Y_sel'] for m in methods] for g in caseorg]);oracle=np.array([min(by[r['model_id']]['Y_sel'] for r in cc if r['org_id']==g) for g in caseorg]);regret=risks-oracle[:,None]
        assert decision['case_orgs']==caseorg and decision['n_case_orgs']==len(caseorg)
        folds_case=len({r['outer_fold'] for r in cc});assert decision['n_outer_folds']==folds_case
        assert decision['status']==('DESCRIPTIVE_ONLY' if len(caseorg)>=30 and folds_case>=3 else 'INSUFFICIENT_CASES');eq(oracle.mean(),decision['U_oracle'],'decision_oracle')
        risk_ci=percentile(casebw@risks);regret_ci=percentile(casebw@regret);methodidx={m:j for j,m in enumerate(methods)}
        for j,m in enumerate(methods):
            stored=decision['groups'][m];eq(risks[:,j].mean(),stored['U'],'decision_U:'+m);eq(risk_ci[:,j],stored['U_CI95'],'decision_U_CI:'+m);eq(regret[:,j].mean(),stored['regret'],'decision_regret:'+m);eq(regret_ci[:,j],stored['regret_CI95'],'decision_regret_CI:'+m)
        for co,stored in zip(pr['contrasts'],decision['contrasts']):
            b,a=methodidx[co['baseline']],methodidx[co['method']];diff=risks[:,b]-risks[:,a]
            assert all(stored[k]==v for k,v in co.items());eq(diff.mean(),stored['D'],'decision_D:'+co['id']);eq(percentile(casebw@diff),stored['D_CI95'],'decision_D_CI:'+co['id']);eq(regret[:,b].mean(),stored['baseline_headroom'],'decision_headroom:'+co['id']);assert stored['status']=='DESCRIPTIVE_ONLY'
    else:assert decision['status']=='INSUFFICIENT_CASES' and decision['n_case_orgs']==0
    # The cost report is a snapshot taken before independent verification. Read
    # its bound event files only; this audit's administrator event is separate.
    events=[]
    for name,sha in cost['cost_events_sha256'].items():
        assert h(W/name)==sha,name
        events.extend(json.loads(line) for line in (W/name).read_text().splitlines())
    stages={}
    for event in events:
        bucket=stages.setdefault(event['stage'],[0,0.,0.,0]);bucket[0]+=1;bucket[1]+=event['wall_seconds'];bucket[2]+=event['cpu_seconds'];bucket[3]=max(bucket[3],event['peak_RSS_bytes'])
    assert set(stages)==set(cost['stages'])
    for stage,expected in stages.items():eq(expected,[cost['stages'][stage][k] for k in ['n_events','sum_wall_seconds','sum_cpu_seconds','process_peak_RSS_bytes']],'cost_stages:'+stage)
    inner=[e for e in events if e['stage']=='supervised_inner_tuning'];final=[e for e in events if e['stage']=='supervised_final_fit'];ui={tuple(e[k] for k in ['outer','inner','group','endpoint','configuration']) for e in inner};uf={tuple(e[k] for k in ['outer','group','endpoint','family','role','configuration']) for e in final}
    assert len(ui)==cost['unique_supervised_inner_fits']==21080 and 620<=len(uf)==cost['unique_supervised_final_fits']<=930
    assert len(inner)==cost['actual_supervised_inner_fits'] and len(final)==cost['actual_supervised_final_fits'];assert len(inner)+len(final)-len(ui)-len(uf)==cost['repeated_fit_attempts']
    assert cost['extra_LLM_calls']==0 and cost['complete_cold_end_to_end_seconds'] is None and cost['complete_cold_status'].startswith('UNMEASURED')
    for name,sha in warm['files'].items():assert h(W/name)==sha,name
    schedule=js(W/'cost/warm_schedule.json');selected=sorted(by,key=lambda mid:(hashlib.sha256(('20260914|'+mid).encode()).hexdigest(),mid))[:100]
    assert schedule['selected_model_ids']==selected and schedule['methods']==methods and schedule['feature_cache_across_methods'] is False
    measurements=[json.loads(line) for line in (W/'cost/warm_measurements.jsonl').read_text().splitlines()];assert len(measurements)==warm['n_measurements']==9900
    rng=np.random.default_rng(20260914);generated=[]
    for repetition in range(3):
        for mid in selected:
            for pos,method in enumerate(rng.permutation(methods)):generated.append({'repetition':repetition,'model_id':mid,'order_within_model':pos,'method':str(method)})
    assert schedule['rows']==generated
    ooindex={r['model_id']:r for r in oo};wm={};maxpredictionerror=0.
    fields=['total_wall_seconds','total_cpu_seconds','io_wall_seconds','io_cpu_seconds','extract_project_wall_seconds','extract_project_cpu_seconds','extract_project_predict_wall_seconds','extract_project_predict_cpu_seconds']
    for expected,measured in zip(generated,measurements):
        assert all(measured[k]==v for k,v in expected.items()) and measured['extra_LLM_calls']==0
        wm.setdefault((measured['model_id'],measured['method']),[]).append(measured)
        assert all(np.isfinite(measured[k]) and measured[k]>=0 for k in fields)
        for ep in pr['endpoints']:
            expected_prediction=float(ooindex[measured['model_id']][measured['method']+'_best_'+ep]);actual=measured[ep+'_prediction'];error=abs(actual-expected_prediction);maxpredictionerror=max(maxpredictionerror,error)
            eq(error,measured[ep+'_oof_abs_error'],'warm_recorded_error',1e-10);assert error<=1e-10
    eq(maxpredictionerror,warm['max_oof_prediction_abs_error'],'warm_max_error',1e-10);assert warm['target_labels_read'] is False
    permodel={(r['model_id'],r['method']):r for r in rows(W/'cost/warm_per_model.csv')};assert set(permodel)==set(wm)
    expected_summary={}
    for key,measured in wm.items():
        assert len(measured)==3
        for field in fields:
            quantile=np.quantile([r[field] for r in measured],[.25,.5,.75],method='linear');eq(quantile,[float(permodel[key][field+'_'+q]) for q in ['q25','median','q75']],'warm_permodel:'+field)
            expected_summary.setdefault((key[1],field),[]).append(quantile[1])
    for (method,field),vals in expected_summary.items():eq(np.quantile(vals,[.25,.5,.75],method='linear'),[warm['methods'][method][field][q] for q in ['q25','median','q75']],'warm_summary:'+method+field)
    result={'verdict':'ACCEPT','node':'V-RESULT','round':1,'reviewer':'independent statistical checker','authored_or_modified_E6_E7':False,'production_statistical_modules_imported':False,'target_access':'gated administrator after SCORING_RELEASE; access audit log preserved','absolute_tolerance':1e-8,'max_errors':errors,'primary_status_disagreements':0,'n_primary_contrasts':31,'n_secondary_contrasts':31,'n_effect_rows':190,'n_case_orgs':len(caseorg),'n_warm_measurements':len(measurements),'primary_statuses':statuses,'plan_sha256':pr['plan_sha256'],'E1_bundle_sha256':h(W/'E1_BUNDLE.json'),'code_release_sha256':h(W/'CODE_RELEASE.json'),'seal_sha256':h(W/'PREDICTION_SEAL.json'),'scoring_release_sha256':h(W/'SCORING_RELEASE.json'),'E6_complete_sha256':h(W/'results/E6_COMPLETE.json'),'cost_report_sha256':h(W/'results/cost_report.json'),'warm_summary_sha256':h(W/'cost/warm_summary.json'),'audit_script_sha256':h(__file__),'wall_seconds':time.monotonic()-start,'scope':'Fixed OOF exploratory results and production cost snapshot; not independent new-data confirmation. This verifier administrator read has its own later cost event.'}
    (W/'reviews/RESULT_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':run()
