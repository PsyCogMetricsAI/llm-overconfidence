"""Pure fixed-OOF grouped evaluation and simultaneous intervals."""
from common import *
def group_losses(y,p,orgs):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);orgs=np.asarray(orgs).astype(str)
    if p.ndim==1:p=p[:,None]
    assert len(y)==len(p)==len(orgs) and np.isfinite(y).all() and np.isfinite(p).all()
    order=np.unique(orgs)
    return order,np.stack([np.mean((y[orgs==g,None]-p[orgs==g])**2,axis=0) for g in order])
def simultaneous(d,base_losses,fold_delta,source_evaluable=None,n_boot=2000,seed=20260914):
    d=np.asarray(d,dtype=float);base=np.asarray(base_losses,dtype=float);fold_delta=np.asarray(fold_delta,dtype=float)
    G,K=d.shape;assert G>=2 and base.shape==(K,) and fold_delta.shape==(5,K) and np.isfinite(d).all()
    source_evaluable=np.ones(K,bool) if source_evaluable is None else np.asarray(source_evaluable,bool)
    delta=d.mean(0);se=np.sqrt(np.sum((d-delta)**2,axis=0)/(G*(G-1)))
    zero=np.all(d==0,axis=0);calc=(se>1e-12)&(base>1e-12)
    ix=np.random.default_rng(seed).integers(0,G,size=(n_boot,G));boot=np.empty((n_boot,K))
    for b in range(n_boot):boot[b]=d[ix[b]].mean(0)
    if calc.any():
        t=np.max(np.abs((boot[:,calc]-delta[calc])/se[calc]),axis=1);crit=float(np.quantile(t,.95,method='linear'))
    else:t=np.zeros(n_boot);crit=None
    result=[]
    for j in range(K):
        lo=hi=I=None;reason=''
        if base[j]<=1e-12:status='INCONCLUSIVE';reason='BASELINE_LOSS_LE1E12'
        elif not calc[j] and not zero[j]:status='INCONCLUSIVE';reason='DEGENERATE_VARIANCE'
        else:
            I=float(delta[j]/base[j]);lo=hi=0. if zero[j] else None
            if not zero[j]:lo=float(delta[j]-crit*se[j]);hi=float(delta[j]+crit*se[j])
            if not source_evaluable[j]:status='INCONCLUSIVE';reason='ALL_TRAIN_MISSING'
            elif I>=.05 and lo>0 and int((fold_delta[:,j]>0).sum())>=4:status='EXPLORATORY_CANDIDATE'
            else:status='NO_PREDEFINED_INCREMENT'
        result.append(dict(delta=float(delta[j]),baseline_loss=float(base[j]),relative_improvement=I,standard_error=float(se[j]),simultaneous_CI=[lo,hi],positive_outer_folds=int((fold_delta[:,j]>0).sum()),outer_fold_delta=fold_delta[:,j].tolist(),status=status,reason=reason))
    return dict(results=result,critical_value=crit,bootstrap_indices=ix,bootstrap_delta=boot,max_statistics=t)
def descriptive_ci(values,indices):
    values=np.asarray(values,float)
    b=np.asarray([values[ix].mean(0) for ix in indices])
    return np.quantile(b,[.025,.975],axis=0,method='linear')
