#!/usr/bin/env python3
"""Synthetic-only producer checks; not independent V-DESIGN or V-CV."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time,json
import numpy as np
from cv_core import *
from score_mainline import evaluate,decide
ROOT=Path(__file__).resolve().parents[1]

def main():
    start=time.monotonic();rng=np.random.default_rng(SEED);M,I=80,240
    orgs=np.array([f'org{m//2:03}' for m in range(M)]);ids=np.array([f'{orgs[m]}/model{m}' for m in range(M)]);items=np.array([str(i) for i in range(I)]);fh=fixed_folds(items,2);half=np.array([fh[i] for i in items]);b=rng.normal(size=I);b=(b-b.mean())/b.std();a=rng.normal(0,.2,M);s=rng.uniform(.3,2,M);E=a[:,None]-s[:,None]*b+rng.normal(0,.025,(M,I));valid=np.ones((M,I),bool)
    for m in range(M):valid[m,rng.choice(np.where(half==0)[0],5,replace=False)]=False
    data={'model_ids':ids,'org_ids':orgs,'item_ids':items,'source_half':half,'ell':E,'valid':valid};base={mid:dict(zip(BASE,rng.uniform(.1,.8,5))) for mid in ids}
    bf,diagnostics=als_reference(E[:,half==0],valid[:,half==0]);assert abs(np.mean(bf))<1e-12 and abs(np.var(bf)-1)<1e-12;assert diagnostics['normal_intercept_max']<1e-10 and diagnostics['normal_slope_max']<1e-10;assert np.corrcoef(bf,b[half==0])[0,1]>.999
    # Direct per-model independent OLS check of projection function.
    rec=project_source(E[0,half==0],valid[0,half==0],bf);use=valid[0,half==0];coef=np.linalg.lstsq(np.column_stack((np.ones(use.sum()),bf[use])),E[0,half==0][use],rcond=None)[0];assert abs(rec['s']+coef[1])<1e-12 and abs(rec['a']-coef[0])<1e-12
    fe=SourceFeatures(data,base);outer=fixed_folds(orgs,5);requests=0
    for k in range(5):
        A={g for g in set(orgs) if outer[g]!=k};H=set(orgs)-A;inner=fixed_folds(A,4)
        fe.crossfit(A,f'outer{k}/train');fe.project(H,A,f'outer{k}/test');requests+=5
        for j in range(4):
            U={g for g in A if inner[g]!=j};V=A-U;fe.crossfit(U,f'outer{k}/inner{j}/train');fe.project(V,U,f'outer{k}/inner{j}/valid');requests+=5
    assert requests==125 and len(fe.cache)<=105
    assert all(not(set(r['reference_orgs'])&set(r['score_orgs'])) for r in fe.projection_log)
    try:fe.project({'org000'},{'org000','org001'},'intentional-leak')
    except AssertionError:pass
    else:raise AssertionError('reference leak guard failed')
    X=rng.normal(size=(80,8));Y=np.clip(.4+X[:,0]*.08+rng.normal(0,.03,80),0,1);Xn=rng.normal(size=(12,8))
    p,f=ridge_fit_predict(X,Y,orgs,Xn,.1);p2,f2=ridge_fit_predict(X,Y,orgs,Xn*1e3,.1);assert f==f2
    # Weighted ridge compared with direct augmented weighted least squares (intercept unpenalized).
    v=group_weights(orgs);mu=v@X;sd=np.sqrt(v@((X-mu)**2));Z=(X-mu)/sd;D=np.column_stack((np.ones(80),Z));Q=D.T@(v[:,None]*D)+np.diag([0]+[.1]*8);bb=np.linalg.solve(Q,D.T@(v*Y));expected=np.clip(np.column_stack((np.ones(12),(Xn-mu)/sd))@bb,0,1);assert np.max(abs(expected-p))<1e-12
    dup=np.concatenate((np.arange(80),np.where(orgs=='org000')[0]));pd,_=ridge_fit_predict(X[dup],Y[dup],orgs[dup],Xn,.1);assert np.max(abs(pd-p))<1e-12
    assert select_lambda(np.ones(9))==8
    og=np.array([f'g{i//2}' for i in range(100)]);of=fixed_folds(og,5);ff=np.array([of[g] for g in og]);idx=rng.integers(0,50,(2000,50));yes=evaluate(np.full(100,.3),np.full(100,.6),np.full(100,.4),og,ff,idx);no=evaluate(np.full(100,.3),np.full(100,.6),np.full(100,.8),og,ff,idx);zero=evaluate(np.full(100,.3),np.full(100,.3),np.full(100,.4),og,ff,idx)
    assert yes['supported'] and not no['supported'] and zero['inconclusive_reason']=='BASELINE_LOSS_LE1E12';assert decide({'Y_cal':yes,'Y_sel':yes})=='JOINT_SUPPORT';assert decide({'Y_cal':yes,'Y_sel':no})=='CAL_ONLY';assert decide({'Y_cal':yes,'Y_sel':zero})=='INCONCLUSIVE'
    out={'producer_checks':'PASS','not_independent_verification':True,'real_target_reads':0,'real_source_fits':0,'synthetic_shape':[M,I],'ALS_gauge_and_normal_equations':diagnostics,'finite_nesting_reference_requests':requests,'finite_nesting_distinct_references':len(fe.cache),'reference_disjoint_checks':len(fe.projection_log),'ridge_direct_augmented_max_abs_diff':float(np.max(abs(expected-p))),'test_features_do_not_change_training_transform':True,'org_replication_weight_invariance':True,'lambda_tie_largest':True,'score_enums_and_zero_baseline':True,'wall_seconds':time.monotonic()-start}
    (ROOT/'synthetic_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
