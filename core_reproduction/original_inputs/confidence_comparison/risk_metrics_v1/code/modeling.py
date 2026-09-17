"""Training-only imputation and frozen package-level selection; no target reader."""
from common import *
from predictors import fit_predict,predict_saved,select_configuration
def fit_imputer(X):
    X=np.asarray(X,dtype=np.float64);assert X.ndim==2 and len(X)>0
    med=np.zeros(X.shape[1]);empty=np.zeros(X.shape[1],dtype=bool)
    for j in range(X.shape[1]):
        finite=X[np.isfinite(X[:,j]),j];empty[j]=len(finite)==0
        if len(finite):med[j]=np.median(finite)
    return {'median':med,'all_missing':empty}
def impute(X,imputer):
    X=np.asarray(X,dtype=np.float64);med=np.asarray(imputer['median']);empty=np.asarray(imputer['all_missing'],bool)
    assert X.ndim==2 and X.shape[1]==len(med)
    out=np.where(np.isfinite(X),X,med[None,:]);out[:,empty]=0.
    assert np.isfinite(out).all();return out
def fit_package(config,X,y,orgs,Xnew):
    imp=fit_imputer(X);pred,saved=fit_predict(config,impute(X,imp),y,orgs,impute(Xnew,imp))
    return pred,dict(imputer=imp,model=saved)
def predict_package(saved,X):return predict_saved(saved['model'],impute(X,saved['imputer']))
def select_package(names,actual_losses,eligible=None):
    eligible=eligible or {n:True for n in names}
    available=[n for n in names if eligible.get(n,False)]
    assert available and names[0] in available
    minimum=min(float(actual_losses[n]) for n in available)
    return next(n for n in names if n in available and actual_losses[n]<=minimum+1e-12)
def train_missing(data,ids):
    return {j:not np.isfinite(np.asarray([float(r[j]) for r in data])).any() for j in ids}
