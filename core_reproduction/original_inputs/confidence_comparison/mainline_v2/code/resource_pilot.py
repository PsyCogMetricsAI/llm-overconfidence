"""A2 maximum training dimensions; only constructed features and labels."""
from common import *
from predictors import fit_predict
from threadpoolctl import threadpool_info
import platform
def run(root=ROOT):
    manifest=rows(root/'cohort_manifest.csv');n=max(sum(int(r['outer_fold'])!=k for r in manifest) for k in range(5));p=12
    rng=np.random.default_rng(20260914);X=rng.normal(size=(n,p));y=np.clip(.3+.07*X[:,0]+.02*rng.normal(size=n),0,1);org=np.array([f'o{i%1065}' for i in range(n)])
    measure=[]
    for conf in protocol(root)['candidate_configurations']:
        start=time.perf_counter();cpu=time.process_time();_,saved=fit_predict(conf,X,y,org,X[:max(1,n//4)])
        measure.append(dict(configuration=conf['id'],family=conf['family'],wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,**saved['timing']))
    hgb=max(r['wall_seconds'] for r in measure if r['family']=='hgb');ridge=max(r['wall_seconds'] for r in measure if r['family']=='ridge')
    # Full dimensional rows and worst observed configuration, doubled for pilot uncertainty.
    supervised=2*(5*4*2*4*(8*hgb+9*ridge)+120*max(hgb,ridge))+600
    source_planning=105*2.238733771024272+300
    estimate=dict(model_id=MODEL_ID,phase='A2_CONSTRUCTED_DATA_ONLY',n_train=n,n_features=p,config_measurements=measure,conservative_A5_seconds=supervised,planning_A3_seconds=source_planning,A3_evidence='old CV pilot; not charged as new cold measurement; fresh max-reference first required fit after V-D',warm_estimate_seconds=1200,threshold_seconds=43200,scope_gate=supervised>43200,threadpools=threadpool_info(),hardware=platform.platform(),true_source_fits=0,target_reads=0)
    assert all(r['num_threads']==1 for r in estimate['threadpools'])
    dump(root/'resource_estimate.json',estimate);print(json.dumps({k:v for k,v in estimate.items() if k!='config_measurements'}),flush=True)
    if estimate['scope_gate']:raise RuntimeError('G-SCOPE predicted A5 >12h')
if __name__=='__main__':run()
