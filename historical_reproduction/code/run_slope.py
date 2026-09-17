#!/usr/bin/env python3
"""Re-fit the historical raw-log-likelihood response law from item NPZ files.
The snapshot algorithms are imported unchanged; only input/output paths are rebound.
No archived fitted values enter either fit or check2. Public output retains only
the accepted descriptive raw statistics and makes no psychological trait claim.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,csv,importlib.util,json,hashlib,sys,time,resource
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,default=ROOT.parent/'data_peritem_v1_20260913');a.add_argument('--out',type=Path,default=ROOT/'work/slope');a.add_argument('--phase',choices=['arc','hellaswag','evaluate','all'],default='all');a.add_argument('--corrected',action='store_true',default=True,help='Corrected raw mode (always enabled)');v=a.parse_args();v.out.mkdir(parents=True,exist_ok=True)
 t=time.monotonic()
 if v.corrected and v.phase!='evaluate':
  import gzip
  with gzip.open(ROOT.parent/'core_reproduction/corrected_inputs/CORRECTED_RAW_IDENTITIES.json.gz','rt') as f: records=json.load(f)['files']
  for rec in records:
   if v.phase not in ['all',rec['bench']]:continue
   path=v.raw/rec['bench']/rec['filename']
   assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['output_sha256'], ('CORRECTED_RAW_SHA',str(path))
 for bench in ['arc','hellaswag']:
  if v.phase not in [bench,'all']:continue
  dest=v.out/bench
  if (dest/'fit_meta_arc.json').exists():raise RuntimeError('Refusing to overwrite completed fit; use a new output directory')
  m=load('slope_'+bench,ROOT/'snapshots/p1_slope/fit_person_slope_rasch.py');m.BENCH=bench;m.DATA_DIR=str(v.raw/bench);m.OUT_DIR=str(dest);m.CACHE=str(dest/'code/raw_L_cache.npz');m.TARGET_MODELS=6739 if bench=='arc' else 6748
  m.main()
 if v.phase in ['evaluate','all']:
  def rows(bench):return {r['repo']:r for r in csv.DictReader((v.out/bench/'theta_s_sigma_arc.csv').open())}
  aa,hh=rows('arc'),rows('hellaswag');paired=v.out/'crossbench_theta_s_arc_hs.csv'
  with paired.open('w',newline='') as out:
   w=csv.writer(out);w.writerow(['repo','theta_arc','theta_hs','s_arc','s_hs','sigma_arc','sigma_hs','flags_arc','flags_hs'])
   for k in sorted(set(aa)&set(hh)):
    r=[]
    for field in ['theta','s','sigma']:
     for row in [aa[k],hh[k]]:
      x=float(row[field]) if row[field] else np.nan;r.append(f'{x:.10g}' if np.isfinite(x) else '')
    w.writerow([k]+r+[aa[k]['flags'],hh[k]['flags']])
  c=load('slope_checks',ROOT/'snapshots/p1_slope/run_checks_N1c.py');c.CB=str(paired);c.ATLAS=str(ROOT/'inputs/atlas_table.csv');c.SCALING=str(ROOT/'inputs/results_atlas_scaling.json')
  rf,canon,_=c.load_family_map();computed=c.check2(c.load_crossbench(),rf,canon)
  result={'variants':{'raw':computed['variants']['raw']},'family_definition':computed['family_definition'],'scope':'Corrected raw log-slope association; no psychological trait interpretation.'}
  dump(v.out/'slope_check2.json',result)
  ref=json.loads((ROOT.parent/'reference_originals/corrected_historical/slope_check2.json').read_text());expected=ref['variants']['raw']
  got=result['variants']['raw']
  checks={'n_matches':got['n']==expected['n'],'fields':{}}
  for kind in ['pearson','spearman']:
   for key in ['point','jackknife_CI95_normal']:
    x=np.asarray(got[kind][key]);y=np.asarray(expected[kind][key]);checks['fields'][kind+'/'+key]={'reconstructed':x.tolist(),'archived':y.tolist(),'max_abs_delta':float(np.max(np.abs(x-y))),'within_1e_9':bool(np.max(np.abs(x-y))<=1e-9)}
  checks['pass']=checks['n_matches'] and all(x['within_1e_9'] for x in checks['fields'].values());dump(v.out/'NUMERIC_COMPARISON.json',checks);print(json.dumps(checks),flush=True)
 dump(v.out/f'RUN_{v.phase}.json',{'phase':v.phase,'elapsed_seconds':time.monotonic()-t,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'raw_root':str(v.raw.resolve()),'output':str(v.out.resolve()),'python':sys.version,'numpy':np.__version__,'scope':'New ALS fits from per-item raw data; archived results used only after all fitting for comparison.'})
if __name__=='__main__':main()
