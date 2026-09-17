"""E2 timing on constructed supervised data plus already permitted source pilots."""
from common import *
from modeling import fit_package
from threadpoolctl import threadpool_info
import platform
def run(root=ROOT):
    require_data(root);pr=protocol(root);cc=readjson(root/'cohort_contract.json')
    n=max(len(v['model_ids']) for v in cc['fixed_rows'].values());rng=np.random.default_rng(pr['seed'])
    X=rng.normal(size=(n,49));Xn=rng.normal(size=(max(100,n//4),49));y=np.clip(.3+.1*X[:,0]+rng.normal(0,.03,n),0,1);orgs=np.array([f'synthetic{i//4}' for i in range(n)])
    reports=[]
    for c in pr['candidate_configurations']:
        with timer(root,'synthetic_supervised_pilot',configuration=c['id'],n_train=n,n_features=49) as e:
            _,saved=fit_package(c,X,y,orgs,Xn);e.update(saved['model']['timing'])
        reports.append({**e,'family':c['family']})
    # Twice measured full49-feature configuration batch for each nominal inner
    # context, plus worst measured final-fit configuration. This is an estimate.
    E5=2*sum(r['wall_seconds'] for r in reports)*(31*2*5*4)+2*max(r['wall_seconds'] for r in reports)*930
    ab=readjson(root/'resource_pilot_ab.json');st=readjson(root/'source_structure_pilot.json')
    # Warm measured source components per model; all33 share M so full20 extraction
    # is included. Prediction allowance derives from worst pilot prediction time/row.
    ab_per=max(r['raw_read_and_compute_wall_seconds'] if 'raw_read_and_compute_wall_seconds' in r else r['wall_seconds'] for r in ab['models'])
    st_per=st['projection_sample_wall_seconds']/st['projection_sample_n']
    pred_per=max(r['prediction_wall_seconds'] for r in reports)/len(Xn)
    warm=3*(ab_per+st_per+2*pred_per)*(100*3*33)+120
    stages={'E3AB':ab['estimated_full_seconds'],'E3RCD':st['conservative_structure_seconds'],'E5':E5,'E7_warm':warm}
    dump(root/'resource_estimate.json',dict(status='E2_PILOT_COMPLETE',n_train=n,n_features=49,config_measurements=reports,stages_estimated_seconds=stages,over_budget={k:v>43200 for k,v in stages.items()},supervised_rule='2*sum17 measured times*1240 +2*max17 measured time*930',warm_rule='3*(maxAB permodel+structure permodel+twoendpoint prediction perrow)*9900+120',source_pilot_sha256={'AB':digest(root/'resource_pilot_ab.json'),'structure':digest(root/'source_structure_pilot.json')},environment_sha256=digest(root/'environment.lock'),threadpools=threadpool_info(),hardware=platform.platform(),target_reads=0,supervised_data='constructed',node_limit_seconds=43200))
    print(json.dumps(dict(stages_estimated_seconds=stages,over_budget={k:v>43200 for k,v in stages.items()})),flush=True)
if __name__=='__main__':run()
