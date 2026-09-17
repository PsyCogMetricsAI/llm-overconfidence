#!/usr/bin/env python3
"""Portable corrected historical checks. References are comparison evidence, never fit inputs.
Automated gate receipts bind every checked artifact; scientific/public-release acceptance
still requires review of the checker and receipts. No private verifier trees are read.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,sys
import numpy as np
from csv_digest import canonical_csv_digest
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('historical_driver',P/'historical_reproduction/code/run_historical_predictions.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
h=d.sha;js=d.js;rows=d.rows

def compare(a,b,path=''):
 if isinstance(b,dict):
  assert isinstance(a,dict) and a.keys()==b.keys(),('KEYS',path)
  for k in b:compare(a[k],b[k],path+'/'+k)
 elif isinstance(b,list):
  assert len(a)==len(b),path
  for j,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(j))
 elif isinstance(b,(float,int)) and not isinstance(b,bool):assert math.isclose(a,b,abs_tol=1e-10,rel_tol=0),('VALUE',path,a,b)
 else:assert a==b,('VALUE',path,a,b)
def table(p,record):return d.table_identity(p,record)
def setsha(x):return hashlib.sha256('\n'.join(sorted(x)).encode()).hexdigest()
def qualified(rs,kind):return [r for r in rs if r['valid_source_fit'].lower()=='true' and (kind=='initial' or r['valid_summaries'].lower()=='true')]

def verify_source(o,kind,ref,bind):
 assert not (o/'predictions_oof.csv').exists(),'Source gate must precede predictor execution'
 fm=js(o/'HISTORICAL_FEATURE_MANIFEST.json');assert fm['driver_sha256']==h(d.__file__)
 assert set(d.mandatory(kind))<=set(fm['files'])
 for n,digest in fm['files'].items():assert h(o/n)==digest,('SOURCE_HASH',n)
 array=o/'source_arrays.npz' if kind=='initial' else o/'../../confidence_validity/source_arrays.npz';d.check_arrays(array)
 snap=P/'historical_reproduction/snapshots'/('p2_initial' if kind=='initial' else 'p3_expanded')
 for src in snap.glob('*.py'):
  expected=src.read_text()
  assert (o/'code'/src.name).read_text()==expected,('SCIENTIFIC_CODE_CHANGED',src.name)
 assert len(ref['feature_tables'])==50
 expected_rows=[]
 for rel,record in ref['feature_tables'].items():table(o/rel,record)
 split=js(o/('splits.json' if kind=='initial' else 'split_manifest.json'))
 for k in range(5):
  test=qualified(rows(o/f'features/outer_{k}/test.csv'),kind);expected_rows.extend(test)
  assert all(split['outer_org_fold'][r['org_id']]==k for r in test)
 assert len(expected_rows)==len({r['model_id'] for r in expected_rows})==ref['n_models']
 assert len({r['org_id'] for r in expected_rows})==ref['n_orgs']
 # Before seal, inspect target bytes only as an opaque SHA; do not decode outcomes.
 for n in ['sealed/targets.csv']+(['sealed/bins.csv'] if kind=='initial' else []):assert h(o/n)==ref['tables'][n]['bytes_sha256'],('OPAQUE_TARGET_IDENTITY',n)
 for n in (['source_B_features.csv'] if kind=='initial' else ['evaluation_cohort.csv','source_features.csv']):table(o/n,ref['tables'][n])
 log=o/'target_access_log.jsonl';assert not log.exists() or not log.read_text().strip(),'Premature target query'
 bind('HISTORICAL_FEATURE_MANIFEST.json')
 return {'all50_corrected_feature_tables':'EXACT_CANONICAL_VALUES','qualified_models':len(expected_rows),'qualified_orgs':ref['n_orgs'],'source_array_identity':'EXACT','original_scientific_modules':'UNCHANGED_EXCEPT_INITIAL_PLAN_PATH','target_queries_before_fit':0}

def verify_seal(o,kind,ref,bind,after=False):
 d.OUT=o;d.require_review('SOURCE',kind)
 pred=rows(o/'predictions_oof.csv');assert len(pred)==len({r['model_id'] for r in pred})==ref['n_models'];assert len({r['org_id'] for r in pred})==ref['n_orgs']
 table(o/'predictions_oof.csv',ref['tables']['predictions_oof.csv'])
 sp=js(o/('splits.json' if kind=='initial' else 'split_manifest.json'));fold=sp['outer_org_fold']
 columns={f'{g}_{ep}' for g in ['base','full'] for ep in ['Y_cal','Y_sel']} if kind=='initial' else {f'{g}_{f}_{ep}' for g in ['B','C','L','F'] for f in ['ridge','hgb','best'] for ep in ['Y_cal','Y_sel']}
 assert set(pred[0])==columns|{'model_id','org_id','outer_fold'}|({'target'} if kind=='expanded' else set())
 assert all(fold[r['org_id']]==int(r['outer_fold']) for r in pred)
 assert {int(r['outer_fold']) for r in pred}==set(range(5))
 assert all(math.isfinite(float(r[c])) and 0<=float(r[c])<=1 for r in pred for c in columns)
 seal=js(o/'PREDICTION_SEAL.json')
 if kind=='initial':assert seal['predictions_sha256']==h(o/'predictions_oof.csv')
 else:
  for rel,want in seal['artifacts'].items():assert h(o/rel)==want;bind(rel)
  for rel in ['candidates.csv','recommendations.csv']:table(o/rel,ref['tables'][rel])
 logs=[json.loads(s) for s in (o/'target_access_log.jsonl').read_text().splitlines()];train=[r for r in logs if r['kind']=='training_query'];final=[r for r in logs if r['kind']=='final_scoring']
 assert len(train)==45 and len(final)==int(after) and len(logs)==45+int(after)
 contexts=set()
 for rec in train:
  k=rec['outer_fold'];bits=rec['context'].split('/');assert bits[0]==f'outer{k}';contexts.add(rec['context']);name='train' if bits[-1]=='final_train' else bits[1]+('_train' if bits[-1]=='train' else '_valid')
  rr=qualified(rows(o/f'features/outer_{k}/{name}.csv'),kind);ids={r['model_id'] for r in rr};orgs={r['org_id'] for r in rr};assert all(fold[g]!=k for g in orgs)
  if kind=='initial':assert set(rec['model_ids'])==ids and set(rec['allowed_orgs'])==orgs and rec['count']==len(ids)
  else:assert rec['requested_models_sha256']==rec['returned_models_sha256']==setsha(ids) and rec['requested_orgs_sha256']==rec['returned_orgs_sha256']==setsha(orgs) and rec['n_models']==len(ids)
 assert len(contexts)==45
 for n in ['predictions_oof.csv','PREDICTION_SEAL.json','selection_log.json','HISTORICAL_SOURCE_REVIEW.json']:bind(n)
 # Access log is append-only at final scoring; bind pre-score prefix separately.
 return {'n_models':len(pred),'n_orgs':ref['n_orgs'],'prediction_columns':len(columns),'exact_training_queries':45,'final_queries':len(final),'predictions_match_accepted_corrected':'EXACT_CANONICAL_VALUES','target_access_log_sha256':h(o/'target_access_log.jsonl')}

def verify_statistics(o,kind,ref,bind):
 checks=verify_seal(o,kind,ref,bind,after=True)
 pred=rows(o/'predictions_oof.csv');target={r['model_id']:r for r in rows(o/'sealed/targets.csv')};table(o/'sealed/targets.csv',ref['tables']['sealed/targets.csv'])
 orgs=sorted({r['org_id'] for r in pred});groups={g:[r for r in pred if r['org_id']==g] for g in orgs};draws=np.random.default_rng(20260914).integers(0,len(orgs),size=(2000,len(orgs)))
 assert np.array_equal(draws,np.load(o/'bootstrap_indices.npy'))
 def errors(col):
  ep='_'.join(col.split('_')[-2:]);return np.array([np.mean([(float(r[col])-float(target[r['model_id']][ep]))**2 for r in groups[g]]) for g in orgs])
 f=P/'reference_originals/corrected_historical'/kind
 if kind=='initial':
  result=js(o/'results_CV.json');gold=js(f/'results_CV.json');compare(result['endpoints'],gold['endpoints']);assert result['verdict']==gold['verdict']
  for ep,v in result['endpoints'].items():
   b,a=errors('base_'+ep),errors('full_'+ep);compare(v['base_loss'],float(b.mean()));compare(v['full_loss'],float(a.mean()));compare(v['relative_improvement'],float((b.mean()-a.mean())/b.mean()));compare(v['delta_ci97_5'],np.quantile(b[draws].mean(1)-a[draws].mean(1),[.0125,.9875]).tolist())
  bind('results_CV.json')
 else:
  result=js(o/'results_comparison.json');gold=js(f/'results_comparison.json');compare(result['comparisons'],gold['comparisons']);compare(js(o/'decision_results.json'),js(f/'decision_results.json'))
  for n in ['prediction_status','decision_status','decision_claim']:assert result[n]==gold[n]
  for key,v in result['comparisons'].items():
   pair,ep=key.split('/');full,base=pair.split('_vs_');b,a=errors(base+'_'+ep),errors(full+'_'+ep);compare(v['base_MSE'],float(b.mean()));compare(v['full_MSE'],float(a.mean()));compare(v['I'],float((b.mean()-a.mean())/b.mean()));compare(v['delta_CI95'],np.quantile((b[draws]-a[draws]).mean(1),[.025,.975]).tolist())
  table(o/'effect_matrix.csv',ref['tables']['effect_matrix.csv'])
  for n in ['results_comparison.json','decision_results.json','effect_matrix.csv']:bind(n)
 checks.update(statistics='INDEPENDENT_ORGANIZATION_MSE_AND_SEEDED_BOOTSTRAP_RECOMPUTED',all_result_fields='MATCH_ACCEPTED_CORRECTED_REFERENCE',interval='97.5%' if kind=='initial' else '95%')
 return checks

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('kind',choices=['initial','expanded']);ap.add_argument('phase',choices=['source','seal','results']);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();o=a.out.resolve();ref=d.refs(a.kind);files={};phase=a.phase.upper();dest=o/f'HISTORICAL_{"RESULT" if phase=="RESULTS" else phase}_REVIEW.json';assert not dest.exists(),'Do not overwrite an existing receipt'
 def bind(n):files[n]=h(o/n)
 checks=verify_source(o,a.kind,ref,bind) if phase=='SOURCE' else verify_seal(o,a.kind,ref,bind) if phase=='SEAL' else verify_statistics(o,a.kind,ref,bind)
 report={'verdict':'ACCEPT','scope':'Automated corrected historical verification; independent reviewer must assess checker/results before scientific release','kind':a.kind,'phase':phase,'driver_sha256':h(d.__file__),'checker_sha256':h(__file__),'reference_manifest_sha256':h(P/'reference_originals/corrected_historical'/a.kind/'REFERENCE_MANIFEST.json'),'files':files,'checks':checks}
 if phase=='SOURCE':report['feature_manifest_sha256']=h(o/'HISTORICAL_FEATURE_MANIFEST.json')
 else:report.update(source_review_sha256=h(o/'HISTORICAL_SOURCE_REVIEW.json'),prediction_seal_sha256=h(o/'PREDICTION_SEAL.json'),predictions_sha256=h(o/'predictions_oof.csv'))
 d.put(dest,report);print(dest)
if __name__=='__main__':main()
