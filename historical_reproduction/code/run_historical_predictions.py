#!/usr/bin/env python3
"""Corrected historical cold fits. Only paths/contracts are adapted, not scientific modules."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,csv,hashlib,json,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];PUBLIC=ROOT.parent
sys.path.insert(0,str(PUBLIC/'core_reproduction'))
from array_identity import check_arrays_against_identity
sys.path.insert(0,str(PUBLIC/'independent_verification'))
from csv_digest import canonical_csv_digest

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
def rows(p):return list(csv.DictReader(Path(p).open()))
def refs(kind):
 base=PUBLIC/'reference_originals/corrected_historical'/kind;r=js(base/'REFERENCE_MANIFEST.json')
 for n,want in r['json_files'].items():assert sha(base/n)==want,('CORRECTED_REFERENCE_CHANGED',n)
 return r
def run(script,*args):
 log=OUT/'logs'/f'{time.time_ns()}_{Path(script).stem}_{"_".join(args)}.log';log.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic();print('RUN',script,*args,flush=True)
 env={**os.environ,'CORE_RAW_ROOT':str(RAW)}
 with log.open('w') as f:subprocess.run([sys.executable,str(script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=43200)
 with (OUT/'run_events.jsonl').open('a') as f:f.write(json.dumps({'script':str(script),'args':args,'seconds':time.monotonic()-start,'log':str(log)})+'\n')
 print('DONE',script,*args,flush=True)
def check_arrays(path):return check_arrays_against_identity(path,js(PUBLIC/'core_reproduction/corrected_inputs/CORRECTED_CORE_ARRAYS.identity.json'))
def table_record(path):
 got=canonical_csv_digest(path)
 with Path(path).open(newline='') as f:
  reader=csv.DictReader(f);names=[n for n in reader.fieldnames if n in {'model_id','org_id','repo','item_id','target'}]
  identities=[[r[n] for n in names] for r in reader]
 got['identity_sha256']=hashlib.sha256(json.dumps([names,identities],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 return got
def table_identity(path,record):
 got=table_record(path);assert got=={k:v for k,v in record.items() if k!='bytes_sha256'},('CORRECTED_TABLE_IDENTITY',str(path));return got

def check_shared():
 cv=SHARED/'confidence_validity';check_arrays(cv/'source_arrays.npz');table_identity(cv/'source_B_features.csv',refs('initial')['tables']['source_B_features.csv'])
 import gzip
 with gzip.open(PUBLIC/'core_reproduction/corrected_inputs/CORRECTED_RAW_IDENTITIES.json.gz','rt') as f:ident=json.load(f)
 index={(r['bench'],r['filename']):r['output_sha256'] for r in ident['files']}
 actual=rows(cv/'target_file_index.csv');assert len(actual)==6706 and len({r['model_id'] for r in actual})==6706
 import numpy as np
 with np.load(cv/'source_arrays.npz',allow_pickle=False) as z:
  assert [(r['model_id'],r['org_id']) for r in actual]==list(zip(z['model_ids'].astype(str),z['org_ids'].astype(str))), 'TARGET_INDEX_IDENTITY_ORDER'
 for r in actual:
  p=Path(r['path']);assert p.resolve()==(RAW/'hellaswag'/p.name).resolve(),('RAW_PATH',p)
  assert r['sha256']==index['hellaswag',p.name],('CORRECTED_RAW_IDENTITY',p)
 return {'source_arrays_sha256':sha(cv/'source_arrays.npz'),'source_B_features_sha256':sha(cv/'source_B_features.csv'),'target_file_index_sha256':sha(cv/'target_file_index.csv')}

def mandatory(kind):
 base=['DESIGN_RELEASE.json','sealed/targets.csv']+[f'code/{p.name}' for p in (ROOT/'snapshots'/('p2_initial' if kind=='initial' else 'p3_expanded')).glob('*.py')]
 base += [f'features/outer_{k}/{n}.csv' for k in range(5) for n in ['train','test']+[f'inner{j}_{role}' for j in range(4) for role in ['train','valid']]]
 if kind=='initial':base+=['source_arrays.npz','source_B_features.csv','target_file_index.csv','splits.json','run_config.json','fit_diagnostics.json','CV2_qualification.json','source_exclusions.json','sealed/bins.csv','sealed/SCORE_CHECKS.json','contracts/PROTOCOL.md']
 else:base+=['protocol.json','A1_OUTPUT_HASHES.json','features/FEATURES_READY.json','A3_qualification.json','evaluation_cohort.csv','source_features.csv','fit_diagnostics.json','split_manifest.json','cohort_contract.json','sealed/TARGETS_READY.json','HISTORICAL_REBUILD_DEPENDENCY.json','../../confidence_validity/source_arrays.npz','../../confidence_validity/target_file_index.csv','../../confidence_validity/code/prepare_data.py','../../confidence_validity/code/score_targets.py']
 return base

def manifest(kind):
 names=set(mandatory(kind))
 names.update(str(p.relative_to(OUT)) for p in (OUT/'features/reference_cache').glob('*') if p.is_file())
 for n in names:assert (OUT/n).is_file(),('MISSING_SOURCE_ARTIFACT',n)
 put(OUT/'HISTORICAL_FEATURE_MANIFEST.json',{'kind':kind,'driver_sha256':sha(__file__),'files':{n:sha(OUT/n) for n in sorted(names)}})

def require_review(phase,kind):
 p=OUT/f'HISTORICAL_{phase}_REVIEW.json';r=js(p)
 assert r['verdict']=='ACCEPT' and r['kind']==kind and r['phase']==phase
 assert r['driver_sha256']==sha(__file__) and r['checker_sha256']==sha(PUBLIC/'independent_verification/release_historical.py')
 assert r['files'], 'Empty review is invalid'
 for n,h in r['files'].items():assert sha(OUT/n)==h,('REVIEW_BINDING_CHANGED',n)
 if phase=='SOURCE':
  assert r['feature_manifest_sha256']==sha(OUT/'HISTORICAL_FEATURE_MANIFEST.json')
  fm=js(OUT/'HISTORICAL_FEATURE_MANIFEST.json');assert set(mandatory(kind))<=set(fm['files'])
  for n,h in fm['files'].items():assert sha(OUT/n)==h,('FEATURE_BINDING_CHANGED',n)
 else:
  assert r['source_review_sha256']==sha(OUT/'HISTORICAL_SOURCE_REVIEW.json')
  assert r['prediction_seal_sha256']==sha(OUT/'PREDICTION_SEAL.json')
  assert r['predictions_sha256']==sha(OUT/'predictions_oof.csv')
  assert r['checks']['target_access_log_sha256']==sha(OUT/'target_access_log.jsonl'), 'Access log changed after seal review'
 return r

def initial_init():
 bindings=check_shared();assert not OUT.exists(),'Fresh output required';OUT.mkdir(parents=True)
 for p in (ROOT/'snapshots/p2_initial').glob('*.py'):copy(p,OUT/'code'/p.name)
 for n in ['source_arrays.npz','source_B_features.csv','target_file_index.csv','data_contract.json','eligibility.csv','input_scan.json']:copy(SHARED/'confidence_validity'/n,OUT/n)
 copy(ROOT/'contracts/initial/PROTOCOL.md',OUT/'contracts/PROTOCOL.md')
 put(OUT/'ADAPTATION.json',{'scope':'Corrected public raw/source input; initial PLAN path relocation only','bindings':bindings,'driver_sha256':sha(__file__)})
 run(OUT/'code/run_mainline.py','prepare')
 put(OUT/'DESIGN_RELEASE.json',{'verdict':'ACCEPT','scope':'Mechanical source-computation binding only; supervised fit requires actual SOURCE review','plan_sha256':sha(OUT/'contracts/PROTOCOL.md'),'code_sha256':{p.name:sha(p) for p in (OUT/'code').glob('*.py')}})
 report=OUT/'TARGET_INPUT_CHECK.md';report.write_text('Public source-array, target-index and raw-item identities were checked before target scoring.\n')
 put(OUT/'TARGET_SCORING_AUTHORIZATION.json',{'status':'ACCEPT','authorize_full_target_scoring':True,'design_report_path':str(report),'design_report_sha256':sha(report)})
 run(OUT/'code/score_targets.py','full-score')

def expanded_init():
 check_shared();assert not OUT.exists(),'Fresh output required';src=SHARED/'confidence_comparison/mainline_v2';ref=refs('expanded')
 # Every table must match independently accepted corrected sources before reuse.
 for rel,want in ref['feature_tables'].items():table_identity(src/rel,want)
 for n in ['evaluation_cohort.csv','source_features.csv']:table_identity(src/n,ref['tables'][n])
 assert sha(src/'sealed/targets.csv')==ref['tables']['sealed/targets.csv']['bytes_sha256'],'OPAQUE_TARGET_IDENTITY'
 ready=js(src/'features/FEATURES_READY.json')
 for n,h in ready['files'].items():assert sha(src/n)==h
 assert not (src/'predictions_oof.csv').exists(),'Do not clone a trained upstream tree'
 shutil.copytree(src,OUT)
 for d in ['predictions','estimators','reviews','_verifier']:
  p=OUT/d
  if p.exists():shutil.rmtree(p)
 for n in ['target_access_log.jsonl','SCORING_RELEASE.json','HISTORICAL_SOURCE_REVIEW.json','HISTORICAL_SEAL_REVIEW.json']:
  p=OUT/n
  if p.exists():p.unlink()
 base=OUT.parents[1]
 for n in ['source_arrays.npz','target_file_index.csv','code/prepare_data.py','code/score_targets.py']:copy(SHARED/'confidence_validity'/n,base/'confidence_validity'/n)
 put(OUT/'HISTORICAL_REBUILD_DEPENDENCY.json',{'shared_cold_root':str(SHARED),'features_ready_sha256':sha(src/'features/FEATURES_READY.json'),'reference_manifest_sha256':sha(PUBLIC/'reference_originals/corrected_historical/expanded/REFERENCE_MANIFEST.json'),'scope':'Fresh corrected public source clone. All50feature table values checked against independently accepted corrected reference identities; no learned predictions.'})
 manifest('expanded')

def evaluate(kind):
 require_review('SOURCE',kind);require_review('SEAL',kind)
 seal=js(OUT/'PREDICTION_SEAL.json')
 rel={'release_final_scoring':True,'scope':'Mechanical release after checked sealed predictions'}
 rel.update({'prediction_sha256':seal['predictions_sha256']} if kind=='initial' else {'seal_sha256':sha(OUT/'PREDICTION_SEAL.json')})
 put(OUT/'SCORING_RELEASE.json',rel);run(OUT/'code'/('score_mainline.py' if kind=='initial' else 'score_comparison.py'))

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('kind',choices=['initial','expanded']);ap.add_argument('phase',choices=['init','features','predict','evaluate']);ap.add_argument('--shared',type=Path,required=True);ap.add_argument('--raw',type=Path,required=True,help='Corrected per-item raw root, never original unmasked data');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();OUT=a.out.resolve();SHARED=a.shared.resolve();RAW=a.raw.resolve()
 import sklearn
 assert sklearn.__version__=='1.6.1','Use pinned requirements'
 if a.phase=='init':initial_init() if a.kind=='initial' else expanded_init()
 elif a.phase=='features':assert a.kind=='initial';run(OUT/'code/run_mainline.py','features');manifest(a.kind)
 elif a.phase=='predict':require_review('SOURCE',a.kind);run(OUT/'code/run_mainline.py','predict')
 else:evaluate(a.kind)
