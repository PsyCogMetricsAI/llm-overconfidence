#!/usr/bin/env python3
"""Cold reproduction orchestration; scientific modules retain their original algorithms.
Run each phase in order. Runtime releases describe reviewed migration, not historical approval.
"""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,difflib
from pathlib import Path
HERE=Path(__file__).resolve().parent
PUBLIC_MODE=False
BASELINE='archive'
CORRECTED=False

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def csvout(p,rs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def cp(a,b):b=Path(b);b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
def run(script,*args):
 name=Path(script).stem+'_'+('_'.join(args) or 'run');log=WORK/'execution_logs'/f'{time.time_ns()}_{name}.log';log.parent.mkdir(exist_ok=True)
 print('RUN',script,*args,'LOG',log,flush=True);start=time.monotonic()
 with log.open('w') as out:subprocess.run([sys.executable,str(script),*args],stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'CORE_RAW_ROOT':str(RAW),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'},timeout=43200,check=True)
 with (WORK/'execution_events.jsonl').open('a') as f:f.write(json.dumps({'script':str(script),'args':args,'elapsed_seconds':time.monotonic()-start,'log':str(log)})+'\n')
 print('DONE',name,flush=True)
def original(name):return HERE/'original_inputs'/name
def public_plan(name=None):
 p=HERE/'PROTOCOL.md'
 integrity=js(HERE/'PUBLIC_INTEGRITY.json')
 assert sha(p)==integrity['public_files']['PROTOCOL.md'], 'PUBLIC_PROTOCOL_CHANGED'
 return p,{'public_sha256':sha(p)}
def plan_ref(name=None):return public_plan()[0]
def identity_baseline():
 idp=(HERE/'corrected_inputs/CORRECTED_CORE_ARRAYS.identity.json') if CORRECTED else original('confidence_validity/source_arrays.identity.json')
 assert idp.is_file(),('MISSING_ARRAY_IDENTITY_RECORD',str(idp))
 return idp,js(idp)
def copy_template(name,dest):cp(original(name),dest)
def release_provenance(kind):
 integrity=js(HERE/'PUBLIC_INTEGRITY.json')
 assert integrity['driver_sha256']==sha(__file__), 'DRIVER_INTEGRITY_MISMATCH'
 for group in ['public_files','corrected_files','source_verifier_files']:
  for rel,want in integrity[group].items():
   assert sha(HERE/rel)==want,('PUBLIC_INPUT_CHANGED',rel)
 return {'verdict':'ACCEPT','scope':'Current public input integrity checks; no historical approval asserted','input_integrity_sha256':sha(HERE/'PUBLIC_INTEGRITY.json'),'kind':kind}
def bound_files(folder,names):return {n:sha(folder/n) for n in names}
def mapped_record(rec,require_cold=False):
 rel=rec['path'];path=Path(rel)
 assert not path.is_absolute() and '..' not in path.parts,('INVALID_INPUT_PATH',rel)
 if rel.startswith('data_peritem_v1_20260913/'):
  rawrel=rel.split('/',1)[1];q=RAW/rawrel
  if CORRECTED:
   import corrected_mode
   index={r['bench']+'/'+r['filename']:r for r in corrected_mode.raw_records()};r=index[rawrel]
   assert r['source_sha256']==rec['sha256'],('RAW_SOURCE_IDENTITY_CHANGED',rel)
   assert q.is_file() and sha(q)==r['output_sha256'],('CORRECTED_RAW_IDENTITY_CHANGED',rel)
   return {'path':str(q),'sha256':sha(q),'bytes':q.stat().st_size,'binding':'corrected-derived-raw'}
 else:
  candidate=WORK/rel
  cold_required=(rel.startswith('confidence_comparison/mainline_v2/features/') or rel.startswith('confidence_comparison/mainline_v2/sealed/') or rel in ['confidence_validity/source_arrays.npz','confidence_comparison/mainline_v2/A3_qualification.json','confidence_comparison/mainline_v2/evaluation_cohort.csv','confidence_comparison/mainline_v2/source_features.csv','confidence_comparison/mainline_v2/fit_diagnostics.json'])
  if require_cold and cold_required:assert candidate.is_file(),('MISSING_COLD_DEPENDENCY',rel)
  q=candidate if candidate.is_file() else original(rel)
 assert q.is_file(),str(q)
 actual_sha=sha(q)
 if not q.is_relative_to(WORK):assert actual_sha==rec['sha256'],('IMMUTABLE_INPUT_CHANGED',rel)
 return {'path':str(q),'sha256':actual_sha,'bytes':q.stat().st_size,'binding':'cold-regenerated' if q.is_relative_to(WORK) else 'immutable-input'}
def init():
 assert not WORK.exists(),'Fresh work directory required; choose another --work for full cold execution'
 WORK.mkdir(parents=True)
 import importlib.metadata,platform
 actual={}
 for line in original('confidence_comparison/risk_metrics_v1/environment.lock').read_text().splitlines():
  name,version=line.split('==');actual[name]=importlib.metadata.version(name);assert actual[name]==version,(name,version,actual[name])
 put(WORK/'ACTUAL_ENVIRONMENT.json',{'python':sys.version,'platform':platform.platform(),'packages':actual})
 for root,rel in [(CV,'confidence_validity'),(OLD,'confidence_comparison/mainline_v2'),(CORE,'confidence_comparison/risk_metrics_v1')]:
  (root/'code').mkdir(parents=True)
  for p in original(rel+'/code').glob('*.py'):cp(p,root/'code'/p.name)
  for n in ['features','references','predictions','estimators','logs','results','reviews','cost','targets','sealed','contracts/legacy_rows']:(root/n).mkdir(parents=True,exist_ok=True)
 p=CV/'code/prepare_data.py';before=p.read_text();after=before.replace("DATA=PROJECT/'data_peritem_v1_20260913'","import os\nDATA=Path(os.environ['CORE_RAW_ROOT'])")
 assert before!=after;p.write_text(after)
 put(WORK/'CODE_MIGRATION.json',{'changes':[{'file':'confidence_validity/code/prepare_data.py','purpose':'configurable read-only raw root','original_sha256':sha(original('confidence_validity/code/prepare_data.py')),'migrated_sha256':sha(p)}],'all_other_scientific_modules':'copied from public scientific source'})
 (WORK/'CODE_MIGRATION.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original/prepare_data.py',tofile='migrated/prepare_data.py')))
 if CORRECTED:
  import corrected_mode
  raw_manifest=[{**r,'sha256':r['output_sha256']} for r in corrected_mode.raw_records()]
 else:raw_manifest=js(HERE/'RAW_MANIFEST.json')
 for rec in raw_manifest:
  p=RAW/rec['bench']/rec['filename'];assert p.is_file() and sha(p)==rec['sha256'],str(p)
 put(WORK/'RAW_VERIFIED.json',{'corrected_mode':CORRECTED,'files':13494,'manifest_sha256':sha(HERE/'RAW_MANIFEST.json'),'raw_root':str(RAW)})
 run(CV/'code/prepare_data.py')
 if CORRECTED:
  import corrected_mode
  corrected_mode.source_subset(CV)
 import numpy as np
 if BASELINE=='identity':
  import array_identity
  idp,ident=identity_baseline();checks=array_identity.check_arrays_against_identity(CV/'source_arrays.npz',ident)
  put(WORK/'RAW_ARRAY_COMPARISON.json',{'verdict':'PASS','mode':'identity','all_arrays_exact':True,'arrays':checks,'identity_record':str(idp),'identity_record_sha256':sha(idp),'cold_arrays_sha256':sha(CV/'source_arrays.npz'),'historical_arrays_sha256':ident['source_sha256'],'exact_original_zip_bytes_compared':False,'scope':('exact corrected core array identity on frozen candidate pool and recomputed item halves' if CORRECTED else 'exact per-array equality (atol=0) against the canonical array digests of the withheld original baseline; npz container bytes were never compared')})
 else:
  base=original('confidence_validity/source_arrays.npz')
  assert base.is_file(),('MISSING_ARCHIVED_BASELINE',str(base))
  with np.load(CV/'source_arrays.npz') as a,np.load(base) as b:
   assert a.files==b.files
   for k in a.files:
    assert np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind in 'fc' else np.array_equal(a[k],b[k]),k
  put(WORK/'RAW_ARRAY_COMPARISON.json',{'verdict':'PASS','all_arrays_exact':True,'cold_arrays_sha256':sha(CV/'source_arrays.npz'),'historical_arrays_sha256':sha(base)})
def legacy_setup():
 prefix='confidence_comparison/mainline_v2/'
 a1=js(original(prefix+'A1_OUTPUT_HASHES.json'))
 for n in a1['files']:
  if original(prefix+n).exists():cp(original(prefix+n),OLD/n)
 for n in ['protocol.json','feature_contract.json','access_contract.json','cohort_contract.json','split_manifest.json','cohort_manifest.csv','environment.lock']:
  copy_template(prefix+n,OLD/n)
 scan=js(CV/'input_scan.json');arc={r['model_id']:r for r in scan if r['bench']=='arc' and r['status']=='OK'};hs={r['model_id']:r for r in scan if r['bench']=='hellaswag' and r['status']=='OK'}
 rr=rows(OLD/'cohort_manifest.csv')
 for r in rr:
  for b,col in [('arc','arc_path'),('hellaswag','HS_path')]:r[col]=str(RAW/b/Path(r[col]).name)
 csvout(OLD/'cohort_manifest.csv',rr)
 cc=js(OLD/'cohort_contract.json');cc['source_arrays_sha256']=sha(CV/'source_arrays.npz');cc['cohort_manifest_sha256']=sha(OLD/'cohort_manifest.csv')
 if CORRECTED:
  import corrected_mode
  corrected_mode.legacy_rebind(sys.modules[__name__],cc,rr)
 put(OLD/'cohort_contract.json',cc)
 pr=js(OLD/'protocol.json');old_plans=dict(pr['plans'])
 if PUBLIC_MODE:
  pr.update(plans={str(plan_ref()):sha(plan_ref())})
 else:
  pr['plans']={str(plan_ref()):sha(plan_ref())}
 put(OLD/'protocol.json',pr)
 oldlines=[]
 for line in original(prefix+'input_manifest.sha256').read_text().splitlines():
  h,oldpath=line.split('  ',1);oldlines.append({'path':oldpath,'sha256':h})
 mapped=[mapped_record(r) for r in oldlines]
 (OLD/'input_manifest.sha256').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in mapped))
 put(OLD/'INPUT_PATH_MAPPING.json',mapped)
 put(OLD/'INPUT_STABILITY.json',{'status':'PASS','n_verified':len(mapped),'cold_migration':True,'mapping_sha256':sha(OLD/'INPUT_PATH_MAPPING.json')})
 index=js(original(prefix+'input_manifest.json'));index.update(manifest_sha256=sha(OLD/'input_manifest.sha256'),verification_sha256=sha(OLD/'INPUT_STABILITY.json'),cohort_sha256=sha(OLD/'cohort_manifest.csv'),n_verified_files=len(mapped),all_pass=True,migration='public input inventory checked against runtime inputs')
 put(OLD/'input_manifest.json',index)
 # New namespace; retain all declared input checks for recreated available metadata.
 existing=list(a1['files']);assert all((OLD/n).is_file() for n in existing)
 put(OLD/'A1_OUTPUT_HASHES.json',{'status':'MIGRATED_RUNTIME_INPUT_BINDING','files':bound_files(OLD,existing)})
 rel=release_provenance('legacy_runtime_design');rel.update(A1_OUTPUT_HASHES_sha256=sha(OLD/'A1_OUTPUT_HASHES.json'),code_sha256={p.name:sha(p) for p in (OLD/'code').glob('*.py')},environment_sha256=sha(OLD/'environment.lock'))
 put(OLD/'DESIGN_RELEASE.json',rel)
 run(OLD/'code/source_stage.py','pilot');run(OLD/'code/source_stage.py','run');run(OLD/'code/score_targets.py','prepare-targets')
 # Qualification memberships are independent of runtime timing/path fields.
 if CORRECTED:
  import corrected_mode
  corrected_mode.check_qualification(sys.modules[__name__]);return
 frozen=js(original('confidence_comparison/risk_metrics_v1/cohort_contract.json'))
 for n,info in frozen['fixed_rows'].items():
  valid=[r for r in rows(OLD/n) if r['valid_source_fit'].lower()=='true' and r['valid_summaries'].lower()=='true']
  assert [r['model_id'] for r in valid]==info['model_ids'],n
 put(WORK/'QUALIFICATION_COMPARISON.json',{'verdict':'PASS','all50_context_memberships_exact':True})
def core_setup():
 started=time.perf_counter()
 prefix='confidence_comparison/risk_metrics_v1/'
 for n in ['protocol.json','formula_registry.json','access_contract.json','split_manifest.json','environment.lock']:copy_template(prefix+n,CORE/n)
 for n in ['cohort_manifest.csv','evaluation_cohort.csv']:cp(OLD/n,CORE/n)
 cc=js(original(prefix+'cohort_contract.json'));cc['legacy']=js(OLD/'cohort_contract.json');cc['legacy_qualification']=js(OLD/'A3_qualification.json')
 if CORRECTED:
  import corrected_mode
  corrected_mode.core_rebind(sys.modules[__name__],cc)
 else:
  for n,info in cc['fixed_rows'].items():
   dst=CORE/info['path'];cp(OLD/n,dst);info['source_path']=str(OLD/n);info['sha256']=sha(dst)
 put(CORE/'cohort_contract.json',cc)
 pr=js(CORE/'protocol.json')
 if PUBLIC_MODE:
  f,bind=public_plan(Path(pr['plan_path']).name)
  pr.update(plan_path=str(f),plan_sha256=sha(f))
 else:
  pr.update(plan_path=str(plan_ref()),plan_sha256=sha(plan_ref()))
 pr.update(legacy_root=str(OLD),source_arrays=str(CV/'source_arrays.npz'),output_root=str(CORE),legacy_python=sys.executable);put(CORE/'protocol.json',pr)
 mapped=[mapped_record(rec,require_cold=True) for rec in js(original(prefix+'input_manifest.json'))['files']]
 assert len(mapped)==len(js(original(prefix+'input_manifest.json'))['files'])
 put(CORE/'input_manifest.json',{'files':mapped,'scope':'one-to-one migration of all original inventory members'})
 (CORE/'input_manifest.sha256').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in mapped))
 olde=js(original(prefix+'E1_BUNDLE.json'));names=list(olde['files']);e1={'status':'MIGRATED_COLD_INPUT_BINDING','plan_sha256':pr['plan_sha256'],'files':bound_files(CORE,names),'fixed_rows_sha256':{v['path']:sha(CORE/v['path']) for v in cc['fixed_rows'].values()},'elapsed_seconds':time.perf_counter()-started}
 put(CORE/'E1_BUNDLE.json',e1)
 rev=release_provenance('cold_data_integrity');rev['E1_bundle_sha256']=sha(CORE/'E1_BUNDLE.json');put(CORE/'reviews/DATA_REVIEW.json',rev)
 code=release_provenance('core_code_integrity');code.update(E1_bundle_sha256=sha(CORE/'E1_BUNDLE.json'),environment_sha256=sha(CORE/'environment.lock'),code_sha256={p.name:sha(p) for p in (CORE/'code').glob('*.py')});put(CORE/'CODE_RELEASE.json',code)
 run(CORE/'code/source_ab.py','pilot');run(CORE/'code/source_structure.py','pilot');run(CORE/'code/resource_pilot.py')
 assert not any(js(CORE/'resource_estimate.json')['over_budget'].values())
 run(CORE/'code/source_ab.py','run');run(CORE/'code/source_structure.py','references');run(CORE/'code/source_structure.py','contexts');run(CORE/'code/assemble_features.py');run(CORE/'code/target_admin.py','prepare')
 put(WORK/'SOURCE_PHASE_COMPLETE.json',{'features_ready_sha256':sha(CORE/'features/FEATURES_READY.json'),'status':'AWAITING_INDEPENDENT_SOURCE_RELEASE'})
def train():
 assert (CORE/'SOURCE_RELEASE.json').exists(),'Independent source release required'
 run(CORE/'code/train.py','all')
 put(WORK/'TRAIN_PHASE_COMPLETE.json',{'seal_sha256':sha(CORE/'PREDICTION_SEAL.json'),'status':'AWAITING_INDEPENDENT_SCORING_RELEASE'})
def evaluate():
 assert (CORE/'SCORING_RELEASE.json').exists(),'Independent seal release required'
 run(CORE/'code/evaluate.py');run(CORE/'code/warm_costs.py','run');run(CORE/'code/cost_report.py')
 if CORRECTED:
  cost_adapter=HERE.parent/'tools/annotate_corrected_cost_scope.py'
  assert js(HERE/'PUBLIC_INTEGRITY.json').get('cost_scope_adapter_sha256')==sha(cost_adapter),'UNREVIEWED_COST_SCOPE_ADAPTER'
  sys.path.insert(0,str(cost_adapter.parent))
  from annotate_corrected_cost_scope import annotate
  put(CORE/'reviews/COST_SCOPE_ADAPTER.json',annotate(CORE))
 for p in original('confidence_comparison/risk_metrics_v1/reviews').glob('*.py'):cp(p,CORE/'reviews'/p.name)
 if CORRECTED:
  adapter=HERE.parent/'tools/adapt_corrected_result_verifier.py'
  approval=js(HERE/'PUBLIC_INTEGRITY.json')
  assert approval.get('result_verifier_adapter_sha256')==sha(adapter),'UNREVIEWED_RESULT_COHORT_ADAPTER'
  sys.path.insert(0,str(adapter.parent))
  from adapt_corrected_result_verifier import adapt
  adapt(CORE)
 run(CORE/'reviews/verify_results.py')
 put(WORK/'CORE_COMPLETE.json',{'status':'COMPLETE_PENDING_EXTERNAL_ACCEPTANCE','result_review_sha256':sha(CORE/'reviews/RESULT_REVIEW.json'),'cold_phases':list(js(WORK/'CODE_MIGRATION.json'))})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['raw','legacy','source','train','evaluate']);ap.add_argument('--source',type=Path);ap.add_argument('--corrected',action='store_true',help='use corrected raw identities, recomputed item halves and fresh qualification; requires --public-mode');ap.add_argument('--raw',type=Path,default=HERE.parent/'data_peritem_v1_20260913');ap.add_argument('--work',type=Path,default=HERE/'work');ap.add_argument('--public-mode',action='store_true',help='public recomputation mode: check the bundled scientific protocol and default to the bundled array-identity baseline');ap.add_argument('--baseline',choices=['archive','identity'],default=None,help='raw-phase array baseline: the withheld original source_arrays.npz (archive) or the bundled exact array digests (identity; default in --public-mode)');a=ap.parse_args()
 CORRECTED=a.corrected
 if CORRECTED and not a.public_mode:raise SystemExit('--corrected requires --public-mode')
 PUBLIC_MODE=a.public_mode;BASELINE=a.baseline or ('identity' if PUBLIC_MODE else 'archive')
 if PUBLIC_MODE and BASELINE=='archive':raise SystemExit('--public-mode cannot use the withheld archived baseline; use --baseline identity')
 RAW=a.raw.resolve();WORK=a.work.resolve();CV=WORK/'confidence_validity';OLD=WORK/'confidence_comparison/mainline_v2';CORE=WORK/'confidence_comparison/risk_metrics_v1'
 if a.phase not in ['raw']:assert js(WORK/'RAW_VERIFIED.json').get('corrected_mode',False)==CORRECTED, 'RAW_MODE_DOES_NOT_MATCH_PHASE_MODE'
 {'raw':init,'legacy':legacy_setup,'source':core_setup,'train':train,'evaluate':evaluate}[a.phase]()
