"""A2 constructed-data checks, no live source fit or target read."""
from common import *
from legacy_primitives import *
from source_stage import summaries,validate_pilot
from predictors import *
from run_mainline import build_recommendations
from score_comparison import contrast,decision_results
import sklearn

def run(root=ROOT):
    checks={};rng=np.random.default_rng(20260914)
    stats,(ell,c,y,v)=summaries(np.array([[0.,0.,np.nan],[0.,0.,0.],[0.,-.1,-.2]]),np.array([[1.,1.,np.nan],[1.,1.,1.],[1.,1.,1.]]),np.array([0,1,0]),np.array(['c','a','b']),np.array([2,3,3]))
    assert y.tolist()==[True,False,True] and ell[2]<0 and v.all();assert abs(stats['A']-2/3)<1e-12
    uniform,_=summaries([[0.,0.,0.]],[[1.,1.,1.]],[0],['a'],[3]);assert abs(uniform['Brier']-2/3)<1e-12 and abs(uniform['H']-1)<1e-12 and uniform['Margin']==0
    for h in [[[1,0]],[[1,np.inf]],[[1,np.nan]]]:assert not validity([[0,0]],h,[0],[2])[0][0]
    assert abs(uniform['REL10']-4/9)<1e-12 and abs(uniform['ECE10']-2/3)<1e-12
    checks['observation_metrics']='PASS: real options, padding, tie, ell/y distinction, Brier K, entropy, REL/ECE'
    X=rng.normal(size=(50,5));X[:,4]=3;y=.2+X[:,0]*.1;org=np.array(['a']*40+['b']*10)
    pred,fit=ridge_fit_predict(X,y,org,X,.1);w=group_weights(org);assert abs(w[:40].sum()-.5)<1e-12
    mu=w@X;sd=np.sqrt(w@((X-mu)**2));active=sd>1e-12;Z=np.where(active,(X-mu)/np.where(active,sd,1),0)
    D=np.c_[np.ones(len(X)),Z];pen=np.diag([0]+[.1]*5);coef=np.linalg.solve(D.T@(w[:,None]*D)+pen,D.T@(w*y));expected=np.clip(D@coef,0,1)
    assert np.max(abs(pred-expected))<1e-12 and not fit['active'][4]
    configs=protocol(root)['candidate_configurations'];loss=np.ones(17);assert configs[select_configuration(configs,loss)]['lambda']==1e4
    hi=select_configuration(configs,loss,'hgb');assert configs[hi]['max_leaf_nodes']==7 and configs[hi]['l2_regularization']==10 and configs[hi]['learning_rate']==.05
    nontransitive=np.ones(17);nontransitive[9]=0;nontransitive[0]=.9e-12;nontransitive[8]=1.8e-12
    assert select_configuration(configs,nontransitive)==0 and select_configuration(configs,nontransitive,'ridge')==8
    checks['nontransitive_tie']='PASS: global ridge0 differs from family ridge8; deploy requires separate selected fit, no new search'
    p,m=fit_predict(configs[9],X,y,org,X);assert sklearn.__version__=='1.6.1' and not m['params']['early_stopping'] and m['params']['categorical_features'] is None
    assert np.max(abs(p-predict_saved(m,X)))==0
    checks['predictors']='PASS: institution weights, train scaling, unpenalized intercept, HGB version/config, ties'
    b=np.linspace(-2,2,130);b=(b-b.mean())/b.std();a=rng.normal(size=60);s=rng.uniform(.5,2,60);e=a[:,None]-s[:,None]*b+rng.normal(0,.01,(60,130));mask=np.ones_like(e,bool)
    bf,d=als_reference(e,mask);assert np.corrcoef(bf,b)[0,1]>.999 and abs(d['mean_b'])<1e-12 and abs(d['var_b']-1)<1e-12
    pp=project_source(e[0],mask[0],bf);assert pp['valid_source_fit'] and abs(pp['normal_intercept'])<1e-10 and abs(pp['normal_slope'])<1e-10
    assert not project_source(e[0],np.arange(130)<99,bf)['valid_source_fit']
    checks['ALS_OLS']='PASS: constructed reference normal equations/gauge, population sigma, >=100 projection rule'
    identity={'reference_sha256':'synthetic'}
    def pilot_record(seconds):return dict(identity=identity,first_required_fit_event=dict(stage='reference_ALS',reference_sha256='synthetic',wall_seconds=seconds),conservative_source_seconds=105*seconds+300,scope_gate=105*seconds+300>43200)
    assert validate_pilot(pilot_record(1),identity)['scope_gate']==False
    for rec,ident in [(pilot_record(1000),identity),(pilot_record(1),{'reference_sha256':'changed'})]:
        try:validate_pilot(rec,ident)
        except RuntimeError:pass
        else:raise AssertionError('source pilot gate bypass')
    # Rechecking the very same stored over-limit record must continue to fail.
    for repeat in range(2):
        try:validate_pilot(pilot_record(1000),identity)
        except RuntimeError:pass
        else:raise AssertionError('cached pilot bypass')
    checks['pilot_resume_gate']='PASS: positive original timing/identity validation; persisted over-limit blocks every resume'
    splits=readjson(root/'split_manifest.json');assert len(splits['projection_contexts'])==125
    for ctx in splits['projection_contexts']:assert not set(ctx['scored_orgs'])&set(ctx['reference_orgs'])
    assert len({r['reference_set_sha256'] for r in splits['projection_contexts']})==55
    checks['split_enumeration']='PASS:125 disjoint contexts,55 reference sets'
    # Source-only candidate selection cannot depend on actual target risks.
    preds=[dict(model_id=f'g/m{i}',org_id='g',outer_fold=0,**{f'{g}_best_Y_sel':.2 for g in ['B','C','L','F']}) for i in range(4)]
    source=[dict(model_id=f'g/m{i}',A=.5 if i<3 else .47) for i in range(4)]
    cc,rr=build_recommendations(preds,source);assert len(cc)==3 and all(r['model_id']=='g/m0' for r in rr)
    orgs=[f'g{i}' for i in range(30)];folds={g:i%5 for i,g in enumerate(orgs)};draws=np.random.default_rng(20260914).integers(0,30,(2000,30))
    res=contrast(np.ones(30),np.ones(30)*.95,draws,orgs,folds);assert res['condition_status']=='FOLLOWUP_PREDICTIVE_SUPPORT'
    res=contrast(np.ones(30),np.ones(30)*.951,draws,orgs,folds);assert res['condition_status']=='NO_PREDEFINED_PREDICTIVE_SUPPORT'
    res=contrast(np.ones(30)*1e-12,np.ones(30)*.5e-12,draws,orgs,folds);assert res['condition_status']=='INCONCLUSIVE'
    ds,_=decision_results([],[],{},'FOLLOWUP_PREDICTIVE_SUPPORT');assert ds['status']=='INSUFFICIENT_CASES' and ds['claim']=='PREDICTION_ONLY'
    checks['decision_inference']='PASS: source-only candidate threshold/tie; primary5% boundary; denominator boundary; insufficient cases'
    dump(root/'selftests.json',dict(model_id=MODEL_ID,status='PASS',checks=checks,production_source_fits=0,target_reads=0,sklearn=sklearn.__version__))
    print(json.dumps(checks),flush=True)
if __name__=='__main__':run()
