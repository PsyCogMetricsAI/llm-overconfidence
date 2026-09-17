#!/usr/bin/env python3
"""Independent frozen-agreement check, separate from original scientific audits."""
from pathlib import Path
import argparse,json,math
import numpy as np
from compare_artifacts import h,compare_npz,compare_csv

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['source','seal','results']);ap.add_argument('--core',required=True,type=Path);ap.add_argument('--frozen',required=True,type=Path);a=ap.parse_args();w=a.core.resolve();f=a.frozen.resolve();checks={};atol=1e-9
 def check(rel):
  x=f/rel;y=w/rel;assert x.exists() and y.exists(),rel
  if x.suffix=='.csv':c=compare_csv(x,y,atol)
  elif x.suffix=='.npz':c=compare_npz(x,y,atol)
  elif x.suffix=='.npy':
   u=np.load(x,allow_pickle=False);v=np.load(y,allow_pickle=False);assert u.shape==v.shape and u.dtype==v.dtype;assert np.array_equal(np.isfinite(u),np.isfinite(v));sel=np.isfinite(u);diff=float(np.max(abs(u[sel]-v[sel]))) if sel.any() else 0.;assert diff<=atol,(rel,diff);c={'shape':list(u.shape),'max_abs_difference':diff}
  else:
   def eq(u,v,p=''):
    if isinstance(u,dict):
     assert u.keys()==v.keys(),p
     for k in u:eq(u[k],v[k],p+'/'+k)
    elif isinstance(u,list):
     assert len(u)==len(v),p
     for i,(aa,bb) in enumerate(zip(u,v)):eq(aa,bb,p+'/'+str(i))
    elif isinstance(u,(int,float)) and not isinstance(u,bool):assert math.isclose(u,v,rel_tol=0,abs_tol=atol),(p,u,v)
    else:assert u==v,(p,u,v)
   eq(json.loads(x.read_text()),json.loads(y.read_text()));c={'all_fields':'match'}
  checks[str(rel)]={'old_sha256':h(x),'new_sha256':h(y),'checks':c}
 if a.phase=='source':
  paths=sorted((f/'features').glob('outer_*/*.csv'));assert len(paths)==50
  for p in paths:check(p.relative_to(f))
  for folder in ['features','features/structure_contexts','references']:
   old=sorted((f/folder).glob('*.npz'));new=sorted((w/folder).glob('*.npz'));assert {p.name for p in old}=={p.name for p in new},folder
   for p in old:check(p.relative_to(f))
  assert len(list((w/'features/structure_contexts').glob('*.npz')))==125
  assert len(list((w/'references').glob('*.npz')))==55
 elif a.phase=='seal':
  for rel in ['predictions_oof.csv','candidates.csv','recommendations.csv']:check(Path(rel))
 elif a.phase=='results':
  assert json.loads((w/'reviews/RESULT_REVIEW.json').read_text())['verdict']=='ACCEPT'
  assert h(f/'reviews/verify_results.py')==h(w/'reviews/verify_results.py')
  for rel in ['effect_matrix.csv','PRIMARY_RESULTS.json','SECONDARY_RESULTS.json','selection_results.json','bootstrap_delta.npy','bootstrap_indices.npy','bootstrap_max_statistics.npy','bootstrap_org_order.json','case_bootstrap_indices.npy','paired_group_differences.npy']:check(Path('results')/rel)
  check(Path('targets/targets.csv'))
 report={'verdict':'PASS','phase':a.phase,'scope':'Frozen scientific output agreement; runtime costs are machine-dependent and excluded. Original scientific verifiers remain separately required.','atol':atol,'checker_sha256':h(__file__),'comparison_code_sha256':h(Path(__file__).with_name('compare_artifacts.py')),'checks':checks}
 out=Path(__file__).resolve().parent/f'CORE_{a.phase.upper()}_COMPARISON.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
if __name__=='__main__':main()
