"""Frozen-contract I/O and measurements; no target reader."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import csv,hashlib,json,time,resource,contextlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MODEL_ID='scientific reproduction'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def readjson(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def write_rows(p,data,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in data for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def protocol(root=ROOT):return readjson(root/'protocol.json')
def require_design(root=ROOT):
    r=readjson(root/'DESIGN_RELEASE.json')
    assert r['verdict']=='ACCEPT'
    assert r['A1_OUTPUT_HASHES_sha256']==digest(root/'A1_OUTPUT_HASHES.json')
    for name,h in readjson(root/'A1_OUTPUT_HASHES.json')['files'].items():assert digest(root/name)==h,name
    for name,h in r['code_sha256'].items():assert digest(root/'code'/name)==h,name
    assert set(r['code_sha256'])=={p.name for p in (root/'code').glob('*.py')}
    assert digest(root/'environment.lock')==r['environment_sha256']
    for path,h in protocol(root)['plans'].items():assert digest(path)==h,path
    import sklearn
    assert sklearn.__version__=='1.6.1'
    return r
@contextlib.contextmanager
def timer(root,stage,**labels):
    wall=time.perf_counter();cpu=time.process_time()
    event=dict(stage=stage,**labels)
    try:yield event
    finally:
        event.update(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
          peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          RSS_scope='process lifetime high-water mark through stage; not isolated stage increment')
        (root/'cost').mkdir(parents=True,exist_ok=True)
        with (root/'cost/events.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
def event_list(root=ROOT):
    p=root/'cost/events.jsonl'
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def load_source(root=ROOT):
    # Source-only artifact path; never open a target file here.
    path=root.parents[1]/'confidence_validity/source_arrays.npz'
    assert digest(path)==readjson(root/'cohort_contract.json')['source_arrays_sha256']
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    manifest=rows(root/'cohort_manifest.csv')
    assert [r['model_id'] for r in manifest]==data['model_ids'].astype(str).tolist()
    assert [r['org_id'] for r in manifest]==data['org_ids'].astype(str).tolist()
    return data,manifest
def qualified(data):
    return [r for r in data if str(r['valid_source_fit']).lower()=='true' and str(r['valid_summaries']).lower()=='true']
def design(data,names):return np.array([[float(r[n]) for n in names] for r in data],float)
