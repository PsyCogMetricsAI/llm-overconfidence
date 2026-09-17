"""Synthetic authorization tests; never accesses real outcomes."""
import tempfile,json
from pathlib import Path
from common import dump,write_rows,digest
from score_targets import allowed_request,validate_seal,selftest,ARTIFACTS,checked_dependency

def expect_reject(fn):
    try:fn()
    except (PermissionError,KeyError,ValueError,FileNotFoundError):return
    raise AssertionError('unauthorized request accepted')

def run():
    results=[]
    with tempfile.TemporaryDirectory(prefix='target_provider_') as td:
        r=Path(td)
        dependency=r/'frozen_dependency.py';dependency.write_text('original')
        manifest={str(dependency):digest(dependency)};checked_dependency(dependency,manifest)
        expect_reject(lambda:checked_dependency(r/'unknown.py',manifest))
        dependency.write_text('changed');expect_reject(lambda:checked_dependency(dependency,manifest))
        results.append('unfrozen_or_changed_original_dependency_reject')
        dump(r/'split_manifest.json',{'outer_org_fold':{'a':1,'b':2,'h':0},'outer':{'0':{'train_orgs':['a','b'],'test_orgs':['h'],'inner_org_fold':{'a':0,'b':1}}}})
        write_rows(r/'cohort_manifest.csv',[{'model_id':m,'org_id':g} for m,g in [('a/1','a'),('a/2','a'),('b/1','b'),('h/1','h')]])
        data=[dict(model_id=m,org_id=g,valid_source_fit='true',valid_summaries='true') for m,g in [('a/1','a'),('a/2','a'),('b/1','b')]]
        files={}
        for n,x in [('train',data),('inner0_train',data[2:]),('inner0_valid',data[:2])]:
            name=f'features/outer_0/{n}.csv';write_rows(r/name,x);files[name]=digest(r/name)
        dump(r/'features/FEATURES_READY.json',{'files':files})
        base={'outer_fold':0,'allowed_orgs':['a','b'],'model_ids':['a/1','a/2','b/1'],'context':'outer0/final_train','target':'hellaswag'}
        assert allowed_request(base,r)[0]==base['model_ids'];results.append('exact_final_context_accept')
        for edit in [{'model_ids':['a/1','b/1']},{'model_ids':['a/1','a/1','b/1']},{'allowed_orgs':['a','b','h']},{'context':'outer0/test'},{'context':'outer1/final_train'},{'target':'mmlu'},{'model_ids':['a/1','a/2','b/1','h/1']},{'allowed_orgs':['a']}]:expect_reject(lambda e=edit:allowed_request(dict(base,**e),r))
        results.append('missing_extra_duplicate_outer_unknown_requests_reject')
        inner=dict(base,context='outer0/inner0/train',model_ids=['b/1'],allowed_orgs=['b']);allowed_request(inner,r)
        valid=dict(base,context='outer0/inner0/validation',model_ids=['a/1','a/2'],allowed_orgs=['a']);allowed_request(valid,r)
        expect_reject(lambda:allowed_request(dict(inner,model_ids=['a/1','a/2'],allowed_orgs=['a']),r));results.append('inner_train_validation_exact_disjoint')
        p=r/'features/outer_0/train.csv';p.write_text(p.read_text()+'\n');expect_reject(lambda:allowed_request(base,r));results.append('changed_feature_hash_reject')
        for n in ARTIFACTS:(r/n).write_text('synthetic')
        dump(r/'PREDICTION_SEAL.json',{'artifacts':{n:digest(r/n) for n in ARTIFACTS}})
        req=dict(seal_path=str(r/'PREDICTION_SEAL.json'),seal_sha=digest(r/'PREDICTION_SEAL.json'))
        expect_reject(lambda:validate_seal(req,r))
        dump(r/'SCORING_RELEASE.json',dict(release_final_scoring=True,seal_sha256=req['seal_sha']));validate_seal(req,r)
        (r/'recommendations.csv').write_text('modified');expect_reject(lambda:validate_seal(req,r));results.append('four_artifact_seal_release_and_mutation')
    return {'status':'PASS','metric_tests':selftest(),'access_tests':results,'uses_real_targets':False}
if __name__=='__main__':print(json.dumps(run(),indent=2))
