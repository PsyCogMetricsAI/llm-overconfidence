"""Confidence validity v2: source-only features and grouped nested prediction primitives.
No target-file reader lives here. Training targets must come from the sealed API.
"""
import hashlib,json
from pathlib import Path
import numpy as np

SEED=20260914
LAMBDAS=10.**np.arange(-4,5)
BASE=['A','C','O','REL10','R50'];FULL=BASE+['theta','log_s','log_sigma']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
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
def select_lambda(losses):
    losses=np.asarray(losses);minimum=float(losses.min());return int(np.where(losses<=minimum+1e-12)[0][-1])

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

class SourceFeatures:
    def __init__(self,data,base,root=None,source_sha='synthetic'):
        self.data=data;self.base=base;self.root=Path(root) if root else None;self.source_sha=source_sha
        self.model_ids=np.asarray(data['model_ids']).astype(str);self.orgs=np.asarray(data['org_ids']).astype(str)
        self.f=np.asarray(data['source_half'])==0;self.ell=np.asarray(data['ell'])[:,self.f];self.valid=np.asarray(data['valid'])[:,self.f]
        self.cache={};self.reference_log={};self.projection_log=[]
        self.algorithm_sha=digest(__file__)
        plan=Path(__file__).resolve().parents[1]/'contracts/PROTOCOL.md'
        self.plan_sha=digest(plan) if plan.exists() else 'synthetic'
        if self.root:self.root.mkdir(parents=True,exist_ok=True)
    def reference(self,orgs):
        orgs=tuple(sorted(set(map(str,orgs))));key=set_sha(orgs)
        if key in self.cache:
            assert self.reference_log[key]['reference_orgs']==list(orgs)
            return key,self.cache[key]
        stem=f'reference_{key}_{self.algorithm_sha[:12]}_{self.plan_sha[:12]}'
        path=self.root/f'{stem}.npz' if self.root else None;meta=self.root/f'{stem}.json' if self.root else None
        if path and path.exists() and meta.exists():
            log=json.loads(meta.read_text());assert log['source_sha256']==self.source_sha and log['reference_orgs']==list(orgs) and log['algorithm_sha256']==self.algorithm_sha and log['plan_sha256']==self.plan_sha;b=np.load(path)['b']
        else:
            take=np.isin(self.orgs,orgs);assert take.any()
            try:b,diag=als_reference(self.ell[take],self.valid[take])
            except ValueError as exc:
                failure=dict(reference_orgs=list(orgs),reference_sha256=key,source_sha256=self.source_sha,algorithm_sha256=self.algorithm_sha,plan_sha256=self.plan_sha,failure=str(exc),reference_model_ids=self.model_ids[take].tolist())
                if self.root:(self.root/f'{stem}.failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                raise
            log=dict(reference_orgs=list(orgs),reference_sha256=key,source_sha256=self.source_sha,algorithm_sha256=self.algorithm_sha,plan_sha256=self.plan_sha,diagnostics=diag,reference_model_ids=self.model_ids[take].tolist())
            if path:np.savez(path,b=b);meta.write_text(json.dumps(log,indent=2)+'\n')
        self.cache[key]=b;self.reference_log[key]=log;return key,b
    def project(self,score_orgs,ref_orgs,context):
        score=set(map(str,score_orgs));ref=set(map(str,ref_orgs));assert not(score&ref),'reference/score org leakage'
        key,b=self.reference(ref);take=np.where(np.isin(self.orgs,list(score)))[0];rows=[]
        for i in take:
            rec=project_source(self.ell[i],self.valid[i],b);rec.update(model_id=self.model_ids[i],org_id=self.orgs[i],reference_sha256=key,context=context)
            rec.update(self.base[self.model_ids[i]]);rows.append(rec)
        self.projection_log.append(dict(context=context,score_orgs=sorted(score),reference_orgs=sorted(ref),reference_sha256=key,n_models=len(rows),n_qualified=sum(x['valid_source_fit'] for x in rows)))
        return rows
    def crossfit(self,orgs,context):
        orgs=set(map(str,orgs));folds=fixed_folds(orgs,4);rows=[]
        for k in range(4):
            score={g for g in orgs if folds[g]==k};ref=orgs-score
            if score:rows+=self.project(score,ref,f'{context}/aux4_{k}')
        return rows

def qualified(rows):return [r for r in rows if r['valid_source_fit']]
def design(rows,names):return np.array([[r[k] for k in names] for r in rows],float)
