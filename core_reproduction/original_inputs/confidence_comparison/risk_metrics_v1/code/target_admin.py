"""Independent target administrator subprocess. Never imported by source workers."""
from common import *
import argparse,sys,re
def allowed_request(req,root=ROOT):
    if req.get('target')!='hellaswag':raise PermissionError('unknown target')
    k=req.get('outer_fold');context=req.get('context')
    if type(k)!=int or k not in range(5):raise PermissionError('invalid fold')
    a=re.fullmatch(r'outer([0-4])/final_train',context or '')
    b=re.fullmatch(r'outer([0-4])/inner([0-3])/(train|validation)',context or '')
    if a and int(a[1])==k:filename='train.csv'
    elif b and int(b[1])==k:filename=f'inner{b[2]}_{"train" if b[3]=="train" else "valid"}.csv'
    else:raise PermissionError('invalid context')
    fixed=readjson(root/'cohort_contract.json')['fixed_rows'][f'features/outer_{k}/{filename}']
    mids=req.get('model_ids');orgs=req.get('allowed_orgs')
    if not isinstance(mids,list) or not isinstance(orgs,list) or len(set(mids))!=len(mids) or len(set(orgs))!=len(orgs):raise PermissionError('duplicate or invalid request')
    if set(mids)!=set(fixed['model_ids']) or set(orgs)!=set(fixed['org_ids']):raise PermissionError('not exact frozen qualified context')
    split=readjson(root/'split_manifest.json');ident=dict(zip(fixed['model_ids'],fixed['org_ids']))
    if any(int(split['outer_org_fold'][g])==k for g in ident.values()):raise PermissionError('outer test leakage')
    if b:
        j=int(b[2]);folds=split['outer'][str(k)]['inner_org_fold']
        if any((int(folds[g])==j)!=(b[3]=='validation') for g in ident.values()):raise PermissionError('inner fold leakage')
    return mids,ident
def ready(root=ROOT):
    require_design(root);r=readjson(root/'targets/TARGETS_READY.json')
    assert r['CODE_RELEASE_sha256']==digest(root/'CODE_RELEASE.json') and r['targets_sha256']==digest(root/'targets/targets.csv')
    return r
def subset(mids,ident,root=ROOT):
    # Only this independent subprocess reads the all-target artifact.
    by={}
    for r in rows(root/'targets/targets.csv'):
        if r['model_id'] in ident:
            m=r['model_id'];assert m not in by and r['org_id']==ident[m]
            by[m]=dict(model_id=m,org_id=r['org_id'],Y_sel=float(r['Y_sel']),Y_cal=float(r['Y_cal']))
            assert all(np.isfinite(by[m][e]) and 0<=by[m][e]<=1 for e in ['Y_sel','Y_cal'])
    assert set(by)==set(mids);return [by[m] for m in mids]
def audit(x,root=ROOT):
    with (root/'targets'/f'access_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(dict(time_ns=time.time_ns(),**x))+'\n')
def prepare(root=ROOT):
    require_design(root)
    assert not (root/'targets/TARGETS_READY.json').exists()
    expected={r['path']:r['sha256'] for r in readjson(root/'input_manifest.json')['files']}
    old=Path(protocol(root)['legacy_root']);p=old/'sealed/targets.csv';r=readjson(old/'sealed/TARGETS_READY.json')
    assert digest(p)==expected[str(p)]==r['targets_sha256']
    # Same snapshot and floor-bin scoring contract as already audited CBV2.
    cf=readjson(old/'feature_contract.json');assert cf['statistics']['bins'].endswith('bin=min(floor(10*c),9)')
    rr=rows(p);ident={x['model_id']:x['org_id'] for x in rows(root/'cohort_manifest.csv')}
    assert len(rr)==len(ident) and {x['model_id']:x['org_id'] for x in rr}==ident
    for x in rr:
        assert int(x['n_valid'])>=200
        assert all(np.isfinite(float(x[e])) and 0<=float(x[e])<=1 for e in ['Y_sel','Y_cal'])
    (root/'targets/targets.csv').write_bytes(p.read_bytes())
    dump(root/'targets/TARGETS_READY.json',dict(status='E4_COMPLETE',CODE_RELEASE_sha256=digest(root/'CODE_RELEASE.json'),targets_sha256=digest(root/'targets/targets.csv'),legacy_targets_path=str(p),legacy_targets_sha256=digest(p),n_models=len(rr),n_orgs=len(set(ident.values())),score_contract='same exact audited snapshot/char-normalization/floor bins/R50 ties; reuse sealed CBV2 labels',new_effects_computed=False))
    print(json.dumps(dict(status='E4_COMPLETE',n_models=len(rr),effects_revealed=False)))
def query(req,root=ROOT):
    ready(root);mids,ident=allowed_request(req,root)
    release=readjson(root/'SOURCE_RELEASE.json');assert release['verdict']=='ACCEPT'
    out=subset(mids,ident,root)
    audit(dict(kind='training',outer_fold=req['outer_fold'],context=req['context'],n_models=len(mids),models_sha256=sethash(mids),orgs_sha256=sethash(ident.values())),root)
    return out
def score(req,root=ROOT):
    ready(root);release=readjson(root/'SCORING_RELEASE.json');seal=readjson(root/'PREDICTION_SEAL.json')
    assert release['verdict']=='ACCEPT' and release['seal_sha256']==digest(root/'PREDICTION_SEAL.json')==req['seal_sha256']
    for n,h in seal['artifacts'].items():assert digest(root/n)==h,n
    assert seal['n_prediction_columns']==190 and seal['source_release_sha256']==digest(root/'SOURCE_RELEASE.json')
    assert seal['CODE_RELEASE_sha256']==digest(root/'CODE_RELEASE.json') and seal['E1_bundle_sha256']==digest(root/'E1_BUNDLE.json')
    rr=rows(root/'evaluation_cohort.csv');ident={r['model_id']:r['org_id'] for r in rr}
    out=subset(sorted(ident),ident,root);audit(dict(kind='aggregate_after_V_SEAL',seal_sha256=req['seal_sha256'],n_models=len(out)),root)
    return out
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','query','score']);args=a.parse_args()
    with timer(ROOT,'target_admin_'+args.stage):
        if args.stage=='prepare':prepare()
        else:print(json.dumps((query if args.stage=='query' else score)(json.load(sys.stdin)),allow_nan=False))
