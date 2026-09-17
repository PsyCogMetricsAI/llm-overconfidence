"""E6 entire frozen matrix; aggregate labels are available only after V-SEAL."""
from common import *
from evaluation_math import group_losses,simultaneous,descriptive_ci
from target_api import scoring_targets
def run(root=ROOT):
    require_design(root);assert not (root/'results/E6_COMPLETE.json').exists(),'E6 already complete'
    identity=dict(seal_sha256=digest(root/'PREDICTION_SEAL.json'),code_release_sha256=digest(root/'CODE_RELEASE.json'))
    attempt=root/'results/E6_ATTEMPT.json'
    if attempt.exists():
        assert readjson(attempt)['identity']==identity,'Partial E6 provenance changed; invalidate/review before resume'
        import shutil
        saved=root/'results'/f'partial_E6_{time.time_ns()}';saved.mkdir()
        for p in (root/'results').iterdir():
            if p.is_file() and (p.name.startswith(('PRIMARY_','SECONDARY_','bootstrap_','paired_group_','E6_')) or p.name in ['effect_matrix.csv','selection_results.json','case_bootstrap_indices.npy']):shutil.copy2(p,saved/p.name)
    dump(attempt,dict(identity=identity,status='RUNNING',pid=os.getpid()))
    pr=protocol(root);rr=rows(root/'predictions_oof.csv');target=scoring_targets(digest(root/'PREDICTION_SEAL.json'))
    by={r['model_id']:r for r in target};assert set(by)=={r['model_id'] for r in rr}
    orgs=np.array([r['org_id'] for r in rr]);folds=np.array([int(r['outer_fold']) for r in rr]);logs=readjson(root/'selection_log.json')
    methods=list(pr['feature_groups'])+list(pr['procedures']);all_effects=[];primary=[]
    with timer(root,'full_fixed_OOF_evaluation',n_contrasts=31,n_models=len(rr)):
        for ep in pr['endpoints']:
            cols=[(g,a,f'{g}_{a}_{ep}') for g in pr['feature_groups'] for a in ['ridge','hgb','best']]+[(g,'best',f'{g}_best_{ep}') for g in pr['procedures']]
            Y=np.array([by[r['model_id']][ep] for r in rr]);P=np.array([[float(r[c]) for _,_,c in cols] for r in rr]);go,gl=group_losses(Y,P,orgs);L=gl.mean(0);ciix=np.random.default_rng(pr['seed']).integers(0,len(go),size=(2000,len(go)))
            loss_ci=descriptive_ci(gl,ciix)
            for j,(g,a,c) in enumerate(cols):all_effects.append(dict(endpoint=ep,group=g,algorithm=a,MSE=float(L[j]),CI95_low=float(loss_ci[0,j]),CI95_high=float(loss_ci[1,j]),interval_scope='descriptive fixed OOF percentile; unadjusted'))
            colindex={c:j for j,(_,_,c) in enumerate(cols)};d=[];bases=[];fd=[];evaluable=[]
            for co in pr['contrasts']:
                ib=colindex[f'{co["baseline"]}_best_{ep}'];ia=colindex[f'{co["method"]}_best_{ep}'];dg=gl[:,ib]-gl[:,ia]
                groupfold=np.array([int(np.unique(folds[orgs==g])[0]) for g in go]);assert all(len(np.unique(folds[orgs==g]))==1 for g in go)
                d.append(dg);bases.append(L[ib]);fd.append([float(dg[groupfold==f].mean()) for f in range(5)])
                if co['metric']:ok=all(l['source_evaluable'][co['metric']] for l in logs)
                elif co['method']=='J':ok=all(l['joint_evaluable'] for l in logs)
                else:ok=True
                evaluable.append(ok)
            d=np.array(d).T;fd=np.array(fd).T
            if ep=='Y_sel':
                sim=simultaneous(d,bases,fd,evaluable);np.save(root/'results/bootstrap_indices.npy',sim['bootstrap_indices']);np.save(root/'results/bootstrap_delta.npy',sim['bootstrap_delta']);np.save(root/'results/bootstrap_max_statistics.npy',sim['max_statistics'])
                primary=[dict(**co,**res) for co,res in zip(pr['contrasts'],sim['results'])]
                dump(root/'results/PRIMARY_RESULTS.json',dict(status='COMPLETE_PENDING_V_RESULT',plan_sha256=pr['plan_sha256'],endpoint=ep,n_models=len(rr),n_orgs=len(go),n_contrasts=31,critical_value=sim['critical_value'],contrasts=primary,scope='exploratory fixedOOF simultaneous intervals; no historical adaptivity or refit correction'))
                dump(root/'results/bootstrap_org_order.json',go.tolist());np.save(root/'results/paired_group_differences.npy',d)
            else:
                ci=descriptive_ci(d,ciix)
                dump(root/'results/SECONDARY_RESULTS.json',dict(endpoint=ep,scope='descriptive only; unadjusted95 percentile intervals',contrasts=[dict(**co,delta=float(d[:,j].mean()),relative_improvement=float(d[:,j].mean()/bases[j]) if bases[j]>1e-12 else None,CI95=ci[:,j].tolist(),outer_fold_delta=fd[:,j].tolist()) for j,co in enumerate(pr['contrasts'])]))
        write_rows(root/'results/effect_matrix.csv',all_effects)
        _selection(by,methods,pr,root)
    names=['results/PRIMARY_RESULTS.json','results/SECONDARY_RESULTS.json','results/effect_matrix.csv','results/selection_results.json']
    dump(root/'results/E6_COMPLETE.json',dict(status='COMPLETE_PENDING_V_RESULT',identity=identity,files={n:digest(root/n) for n in names}))
    print(json.dumps(dict(status='E6_COMPLETE_PENDING_V_RESULT',supporting_ids=[r['id'] for r in primary if r['status']=='EXPLORATORY_CANDIDATE']),ensure_ascii=False),flush=True)
def _selection(by,methods,pr,root):
    cc=rows(root/'candidates.csv');recs=rows(root/'recommendations.csv');groups=sorted({r['org_id'] for r in cc});allfolds={int(r['outer_fold']) for r in cc}
    values=np.array([[by[next(r['model_id'] for r in recs if r['org_id']==g and r['method']==m)]['Y_sel'] for m in methods] for g in groups])
    oracle=np.array([min(by[r['model_id']]['Y_sel'] for r in cc if r['org_id']==g) for g in groups])
    if not groups:
        dump(root/'results/selection_results.json',dict(status='INSUFFICIENT_CASES',n_case_orgs=0,groups={},contrasts=[]));return
    ix=np.random.default_rng(pr['seed']).integers(0,len(groups),size=(2000,len(groups)));np.save(root/'results/case_bootstrap_indices.npy',ix)
    u=values.mean(0);regret=values-oracle[:,None];uci=descriptive_ci(values,ix);rci=descriptive_ci(regret,ix);idx={m:i for i,m in enumerate(methods)}
    comparisons=[]
    for co in pr['contrasts']:
        i=idx[co['baseline']];j=idx[co['method']];D=values[:,i]-values[:,j];ci=descriptive_ci(D,ix)
        comparisons.append(dict(**co,D=float(D.mean()),D_CI95=ci.tolist(),baseline_headroom=float(regret[:,i].mean()),status='DESCRIPTIVE_ONLY'))
    dump(root/'results/selection_results.json',dict(status='DESCRIPTIVE_ONLY' if len(groups)>=30 and len(allfolds)>=3 else 'INSUFFICIENT_CASES',n_case_orgs=len(groups),n_outer_folds=len(allfolds),case_orgs=groups,U_oracle=float(oracle.mean()),groups={m:dict(U=float(u[j]),U_CI95=uci[:,j].tolist(),regret=float(regret[:,j].mean()),regret_CI95=rci[:,j].tolist()) for j,m in enumerate(methods)},contrasts=comparisons,scope='unadjusted descriptive paired case-org intervals; no binary decision-success threshold'))
if __name__=='__main__':run()
