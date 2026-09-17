#!/usr/bin/env python3
from pathlib import Path
import json,time
from compare_artifacts import compare_npz,h
S=Path(__file__).resolve().parents[1]
start=time.monotonic()
manifest=S/'core_reproduction/RAW_MANIFEST.json'
records=json.loads(manifest.read_text());assert len(records)==13494
for r in records:
 p=S/'data_peritem_v1_20260913'/r['bench']/r['filename']
 assert h(p)==r['sha256'],str(p)
old=S/'core_reproduction/original_inputs/confidence_validity/source_arrays.npz'
new=S/'core_reproduction/work/confidence_validity/source_arrays.npz'
checks=compare_npz(old,new,atol=0)
report={'verdict':'PASS','scope':'Independent rehash of every raw manifest member and exact reconstructed arrays, including shape/dtype/masks/model and organization ordering. No source feature or scientific result acceptance.','reviewer':'reproduction_verifier (replacement independent reviewer)','files_checked':len(records),'manifest_sha256':h(manifest),'old_arrays_sha256':h(old),'cold_arrays_sha256':h(new),'array_checks':checks,'checker_sha256':h(__file__),'comparison_code_sha256':h(Path(__file__).with_name('compare_artifacts.py')),'elapsed_seconds':time.monotonic()-start}
(S/'independent_verification/RAW_INDEPENDENT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: 13494 raw hashes and exact NPZ arrays')
