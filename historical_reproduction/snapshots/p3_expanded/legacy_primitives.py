"""Exact source mathematical primitives copied from frozen CV; no path-owning classes or target readers."""

import hashlib,json

from pathlib import Path

import numpy as np

from scipy.special import logsumexp

SEED=20260914

def fixed_folds(ids,k):
    keys=sorted(set(map(str,ids)),key=lambda s:(hashlib.sha256(f'{SEED}|{s}'.encode()).hexdigest(),s))
    return {key:i%k for i,key in enumerate(keys)}

def set_sha(orgs):return hashlib.sha256(('\n'.join(sorted(set(map(str,orgs))))+'\n').encode()).hexdigest()

def group_weights(orgs):
    orgs=np.asarray(orgs);u,n=np.unique(orgs,return_counts=True);counts=dict(zip(u,n));return np.array([1/(len(u)*counts[o]) for o in orgs])

def group_mse(y,p,orgs):
    return float(group_weights(orgs)@((np.asarray(y)-np.asarray(p))**2))

def ridge_fit_predict(X,Y,orgs,Xnew,lam):
    X=np.asarray(X,float);Y=np.asarray(Y,float);Xnew=np.asarray(Xnew,float)
    if not np.isfinite(X).all() or not np.isfinite(Y).all() or not np.isfinite(Xnew).all():raise ValueError('nonfinite ridge input')
    v=group_weights(orgs);mu=v@X;sd=np.sqrt(v@((X-mu)**2));active=sd>1e-12
    scale=np.where(active,sd,1.);Z=np.where(active,(X-mu)/scale,0.);Zn=np.where(active,(Xnew-mu)/scale,0.)
    ym=v@Y;beta=np.linalg.solve(Z.T@(v[:,None]*Z)+lam*np.eye(X.shape[1]),Z.T@(v*(Y-ym)))
    return np.clip(ym+Zn@beta,0,1),dict(mean=mu.tolist(),sd=sd.tolist(),active=active.tolist(),beta=beta.tolist(),intercept=float(ym),lambda_value=float(lam))

def als_reference(ell,valid,tol=1e-6,max_iter=500,min_models_per_item=20):
    """Masked ALS with gauge-preserving parameter transformation and fixed sign."""
    ell=np.asarray(ell,float);valid=np.asarray(valid,bool)&np.isfinite(ell)
    eligible=valid.sum(0)>=min_models_per_item
    if eligible.sum()<100:raise ValueError('INSUFFICIENT_REFERENCE_ITEMS')
    e=ell[:,eligible];m=valid[:,eligible];n=m.sum(1)
    if np.any(n<2):raise ValueError('REFERENCE_MODEL_WITH_LT2_QUALIFIED_ITEMS')
    e=np.where(m,e,0.);init=-e.sum(0)/m.sum(0);sd=init.std()
    if sd<=1e-12:raise ValueError('DEGENERATE_REFERENCE_INIT')
    init=(init-init.mean())/sd;b=init.copy();hist=[]
    def update(b):
        mb=(m*b).sum(1)/n;me=e.sum(1)/n;var=(m*b*b).sum(1)/n-mb*mb
        if np.any(var<=1e-12):raise ValueError('DEGENERATE_REFERENCE_B_VARIANCE')
        s=-((e*b).sum(1)/n-me*mb)/var;a=me+s*mb
        return a,s
    for it in range(1,max_iter+1):
        prev=b.copy();a,s=update(b);den=(m*s[:,None]**2).sum(0)
        if np.any(den<=1e-20):raise ValueError('DEGENERATE_REFERENCE_S_DENOMINATOR')
        raw=-((e-a[:,None]*m)*s[:,None]).sum(0)/den
        mu=raw.mean();scale=raw.std()
        if scale<=1e-12:raise ValueError('DEGENERATE_REFERENCE_GAUGE')
        # a-s*raw == (a-s*mu)-(s*scale)*standard(raw).
        b=(raw-mu)/scale;a=a-s*mu;s=s*scale
        if np.dot(b,init)<0:b=-b;s=-s
        obj=float(np.sum(np.where(m,e-a[:,None]+s[:,None]*b,0.)**2));hist.append(obj)
        delta=float(np.max(abs(b-prev)))
        if delta<tol:break
    if delta>=tol:raise ValueError(f'ALS_NONCONVERGENCE iterations={it} delta={delta}')
    a,s=update(b);res=np.where(m,e-a[:,None]+s[:,None]*b,0.)
    fullb=np.full(ell.shape[1],np.nan);fullb[eligible]=b
    return fullb,dict(iterations=it,delta=delta,objective=float(np.sum(res**2)),objective_history=hist,n_reference_models=len(e),n_qualified_items=int(eligible.sum()),mean_b=float(b.mean()),var_b=float(b.var()),sign_dot_init=float(b@init),normal_intercept_max=float(np.max(abs(res.sum(1)))),normal_slope_max=float(np.max(abs((res*b).sum(1)))))

def project_source(ell,valid,b):
    use=np.asarray(valid,bool)&np.isfinite(ell)&np.isfinite(b)
    n=int(use.sum());out={'n_valid_F':n,'valid_source_fit':False}
    if n<100:out['exclusion_reason']='F_QUALIFIED_LT100';return out
    x=np.asarray(b)[use];y=np.asarray(ell)[use];vx=float(np.var(x));vy=float(np.var(y))
    if vx<=1e-12:out['exclusion_reason']='B_VARIANCE_LE1E12';return out
    s=-float(np.mean((x-x.mean())*(y-y.mean())))/vx;a=float(y.mean()+s*x.mean());res=y-a+s*x;sigma=float(np.sqrt(np.mean(res**2)))
    out.update(a=a,s=s,sigma=sigma,var_ell=vy,normal_intercept=float(np.mean(res)),normal_slope=float(np.mean(res*(x-x.mean()))))
    reasons=[]
    if not np.isfinite(s) or s<=1e-8:reasons.append('S_LE1E8')
    if not np.isfinite(sigma) or sigma<=1e-8:reasons.append('SIGMA_LE1E8')
    if vy<=1e-6:reasons.append('ELL_VARIANCE_LE1E6')
    if reasons:out['exclusion_reason']=';'.join(reasons);return out
    theta=a/s
    if not np.isfinite(theta):out['exclusion_reason']='THETA_NONFINITE';return out
    out.update(theta=theta,log_s=float(np.log(s)),log_sigma=float(np.log(sigma)),valid_source_fit=True,exclusion_reason='')
    return out

def validity(loglik,char_lens,gold,n_options=None):
    ll=np.asarray(loglik,dtype=np.float64); h=np.asarray(char_lens,dtype=np.float64)
    rawg=np.asarray(gold)
    if ll.ndim!=2 or h.shape!=ll.shape or rawg.shape!=(ll.shape[0],):raise ValueError('shape_mismatch')
    n,k=ll.shape
    real=np.isfinite(h)&(h>0)
    nr=real.sum(1)
    # Cache padding is NaN in both h and L. Inf/zero/negative h is invalid,
    # not a missing option silently dropped from a probability denominator.
    bad_h=np.any((~np.isnan(h))&(~real),axis=1)
    padding_bad=np.any(np.isnan(h)&(~np.isnan(ll)),axis=1)
    complete=np.all((~real)|np.isfinite(ll),axis=1)
    gfinite=np.isfinite(rawg)
    gint=gfinite&(rawg==np.floor(np.where(gfinite,rawg,0)))
    g=np.where(gint,rawg,0).astype(np.int64)
    legal=gint&(g>=0)&(g<k)
    ix=np.arange(n);safe=np.clip(g,0,max(k-1,0))
    legal &= real[ix,safe]
    mismatch=np.zeros(n,dtype=bool)
    if n_options is not None:
        no=np.asarray(n_options)
        if no.shape!=(n,):raise ValueError('n_options_shape')
        mismatch=(~np.isfinite(no))|(no!=nr)
    reason={'bad_char_length':bad_h,'padding_has_loglik':padding_bad,'incomplete_real_loglik':~complete,
            'invalid_gold':~legal,'fewer_than_two_options':nr<2,'n_options_disagrees':mismatch}
    valid=np.ones(n,dtype=bool)
    for a in reason.values():valid &= ~a
    return valid,real,g,{a:int(b.sum()) for a,b in reason.items()},nr

def observations(loglik,char_lens,gold,n_options=None):
    ll=np.asarray(loglik,dtype=np.float64);h=np.asarray(char_lens,dtype=np.float64)
    valid,real,g,reasons,nr=validity(ll,h,gold,n_options)
    n,k=ll.shape
    ell=np.full(n,np.nan);c=np.full(n,np.nan);y=np.zeros(n,dtype=bool)
    r=np.flatnonzero(valid)
    err=0.0
    if len(r):
        u=np.full((len(r),k),-np.inf)
        np.divide(ll[r],h[r],out=u,where=real[r])
        pred=np.argmax(u,axis=1) # first index resolves ties
        p=np.exp(u-logsumexp(u,axis=1,keepdims=True))
        err=float(np.max(np.abs(p.sum(1)-1)))
        c[r]=p[np.arange(len(r)),pred];y[r]=pred==g[r]
        gold_u=u[np.arange(len(r)),g[r]].copy()
        u[np.arange(len(r)),g[r]]=-np.inf
        ell[r]=gold_u-logsumexp(u,axis=1)
        if not np.all(np.isfinite(ell[r])):raise ValueError('finite_inputs_nonfinite_logodds')
        if err>1e-12:raise AssertionError('probability_sum_tolerance')
    return ell,c,y,valid,reasons,nr,err
