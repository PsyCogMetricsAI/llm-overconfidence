"""Two fixed families and inner-only selection."""
from common import *
from legacy_primitives import ridge_fit_predict,group_weights
from sklearn.ensemble import HistGradientBoostingRegressor
def fit_predict(config,X,y,orgs,Xnew):
    assert np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(Xnew).all()
    start=time.perf_counter();cpu=time.process_time()
    if config['family']=='ridge':
        _,fit=ridge_fit_predict(X,y,orgs,np.empty((0,X.shape[1])),config['lambda'])
        saved=dict(family='ridge',fit=fit)
        ft=time.perf_counter()-start;fc=time.process_time()-cpu;start=time.perf_counter();cpu=time.process_time()
        pred=predict_saved(saved,Xnew)
        saved['timing']=dict(fit_wall_seconds=ft,fit_cpu_seconds=fc,prediction_wall_seconds=time.perf_counter()-start,prediction_cpu_seconds=time.process_time()-cpu)
        return pred,saved
    model=HistGradientBoostingRegressor(loss='squared_error',max_iter=200,min_samples_leaf=20,
      max_bins=255,early_stopping=False,random_state=20260914,categorical_features=None,
      max_depth=None,warm_start=False,learning_rate=config['learning_rate'],
      max_leaf_nodes=config['max_leaf_nodes'],l2_regularization=config['l2_regularization'])
    model.fit(X,y,sample_weight=len(y)*group_weights(orgs))
    ft=time.perf_counter()-start;fc=time.process_time()-cpu;start=time.perf_counter();cpu=time.process_time()
    pred=np.clip(model.predict(Xnew),0,1)
    return pred,dict(family='hgb',estimator=model,params=model.get_params(),timing=dict(fit_wall_seconds=ft,fit_cpu_seconds=fc,prediction_wall_seconds=time.perf_counter()-start,prediction_cpu_seconds=time.process_time()-cpu))
def predict_saved(saved,X):
    if saved['family']=='hgb':return np.clip(saved['estimator'].predict(X),0,1)
    f=saved['fit'];mu=np.asarray(f['mean']);sd=np.asarray(f['sd']);active=np.asarray(f['active'])
    Z=np.where(active,(X-mu)/np.where(active,sd,1),0.)
    return np.clip(f['intercept']+Z@np.asarray(f['beta']),0,1)
def select_configuration(configs,losses,family=None):
    ids=[i for i,c in enumerate(configs) if family is None or c['family']==family]
    minimum=min(float(losses[i]) for i in ids)
    tied=[i for i in ids if losses[i]<=minimum+1e-12]
    def order(i):
        c=configs[i]
        return (0,-c['lambda'],0,0) if c['family']=='ridge' else (1,c['max_leaf_nodes'],-c['l2_regularization'],c['learning_rate'])
    return min(tied,key=order)
