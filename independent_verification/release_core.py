#!/usr/bin/env python3
"""Independent-authored, fail-closed runtime source/seal release automation.
Runs archived numeric verifiers and a declared public seal adaptation; creates fresh hash-bound runtime approvals.
These releases certify these checks on this run, never new-data confirmation.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,time

def h(p):
 q=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):q.update(b)
 return q.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def corrected_feature_check(w,m,review):
 marker=w.parents[1]/'RAW_VERIFIED.json'
 assert marker.is_file() and js(marker).get('corrected_mode') is True, 'CORRECTED_RAW_MODE_REQUIRED'
 for rel in ['corrected_inputs/CORRECTED_FEATURE_IDENTITIES.json','../independent_verification/csv_digest.py']:
  assert review.get('source_verifier_files',{}).get(rel)==h(m/rel),('SOURCE_IDENTITY_BINDING_CHANGED',rel)
 from csv_digest import canonical_csv_digest
 identity=m/'corrected_inputs/CORRECTED_FEATURE_IDENTITIES.json';expected=js(identity)
 actual={str(p.relative_to(w)):p for p in (w/'features').glob('outer_*/*.csv')}
 assert expected['n_files']==50 and set(actual)==set(expected['files']), 'CORRECTED_FEATURE_FILE_SET_MISMATCH'
 for rel,p in actual.items():
  assert canonical_csv_digest(p)==expected['files'][rel],('CORRECTED_FEATURE_VALUES_MISMATCH',rel)
 return {'files':50,'identity_sha256':h(identity),'scope':'exact canonical corrected feature values and table layout'}
def main():
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['source','seal']);a.add_argument('--core',type=Path,required=True);a.add_argument('--migration-root',type=Path,required=True);a.add_argument('--public-mode',action='store_true',help='public recomputation mode: apply the declared, hash-bound mechanical plan adapter to the copied seal verifier (the public verifier template remains unchanged)');v=a.parse_args();w=v.core.resolve();m=v.migration_root.resolve()
 review=js(m/'PUBLIC_INTEGRITY.json');assert review['driver_sha256']==h(m/'run_core.py')
 for group in ['public_files','corrected_files','source_verifier_files']:
  for rel,want in review[group].items():assert h(m/rel)==want,('PUBLIC_INPUT_CHANGED',rel)
 original=m/'original_inputs/confidence_comparison/risk_metrics_v1';release=w/('SOURCE_RELEASE.json' if v.phase=='source' else 'SCORING_RELEASE.json');assert not release.exists(),'Existing release must not be silently overwritten'
 for p in (original/'code').glob('*.py'):assert h(p)==h(w/'code'/p.name),p.name
 name='verify_source.py' if v.phase=='source' else 'verify_seal_independent.py';src=original/'reviews'/name;dst=w/'reviews'/name;dst.parent.mkdir(exist_ok=True)
 plan_adapter=None
 if v.phase=='seal' and v.public_mode:
  sys.path.insert(0,str(m.parent/'tools'))
  from adapt_public_seal_verifier import (adapt_seal_verifier,expected_source_sha,load_binding,sha256_bytes)
  binding,public_sha,public_rel=load_binding(m);text=src.read_text()
  adapted,plan_adapter=adapt_seal_verifier(text,source_sha256=sha256_bytes(text.encode()),expected_source_sha256=expected_source_sha(m),public_protocol_sha256=public_sha,public_protocol_relpath=public_rel,plan_binding_sha256=h(m/'PUBLIC_INTEGRITY.json'))
  plan_adapter['public_protocol_release_path']=binding['public_protocol']
  dst.write_text(adapted);assert h(dst)==plan_adapter['adapted_sha256'],'adapted seal verifier hash mismatch'
  (w/'reviews'/'PLAN_ADAPTER.json').write_text(json.dumps(plan_adapter,indent=1)+'\n')
 else:
  shutil.copy2(src,dst);assert h(src)==h(dst)
 log=w/'reviews'/('INDEPENDENT_'+v.phase.upper()+'_RUN.log');start=time.monotonic()
 with log.open('w') as f:subprocess.run([sys.executable,str(dst)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=43200)
 if v.phase=='source':
  corrected_identity=corrected_feature_check(w,m,review) if v.public_mode else None
  audit=js(w/'reviews/SOURCE_NUMERIC_AUDIT.json');assert audit['status']=='PASS' and audit['target_labels_read'] is False
  assert audit['n_contexts_checked']==125 and audit['n_reference_sets_checked']==55 and audit['n_feature_files_checked']==50
  report={'verdict':'ACCEPT','scope':'Source numeric checks on freshly reconstructed inputs; no target labels read','source_numeric_audit_sha256':h(w/'reviews/SOURCE_NUMERIC_AUDIT.json'),'automation':'public source and seal checker','automation_sha256':h(__file__)}
  if corrected_identity:report['corrected_feature_identity']=corrected_identity
  write(w/'reviews/SOURCE_REVIEW.json',report)
  r={'verdict':'ACCEPT','features_ready_sha256':h(w/'features/FEATURES_READY.json'),'review_path':'reviews/SOURCE_REVIEW.json','review_sha256':h(w/'reviews/SOURCE_REVIEW.json'),'code_release_sha256':h(w/'CODE_RELEASE.json'),'E1_bundle_sha256':h(w/'E1_BUNDLE.json'),'target_readiness_sha256':h(w/'targets/TARGETS_READY.json')}
 else:
  report=js(w/'reviews/SEAL_REVIEW.json');assert report['verdict']=='ACCEPT' and report['target_values_read'] is False and all(report['checks'].values())
  r={'verdict':'ACCEPT','seal_sha256':h(w/'PREDICTION_SEAL.json'),'review_path':'reviews/SEAL_REVIEW.json','review_sha256':h(w/'reviews/SEAL_REVIEW.json'),'code_release_sha256':h(w/'CODE_RELEASE.json'),'source_release_sha256':h(w/'SOURCE_RELEASE.json'),'E1_bundle_sha256':h(w/'E1_BUNDLE.json')}
 r.update(released_by='public scientific checker',scope='Fresh runtime gate after original audit; not historical acceptance reuse',automation_sha256=h(__file__),original_verifier_sha256=h(src),elapsed_seconds=time.monotonic()-start)
 if plan_adapter:r.update(plan_adapter_record='reviews/PLAN_ADAPTER.json',plan_adapter=plan_adapter,scope='Fresh runtime gate after original audit with the declared public mechanical plan adapter on the copied seal verifier; the archived verifier bytes are preserved in original_inputs and the archived assertion is kept')
 write(release,r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
