"""Risk metrics v1 shared source-only I/O, gates and cost measurements."""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import csv,hashlib,json,time,resource,contextlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT.parent/'mainline_v2'
MODEL_ID='scientific reproduction'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def readjson(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def write_rows(p,data,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in data for k in r))
    tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    tmp.replace(p)
def protocol(root=ROOT):return readjson(root/'protocol.json')
def require_data(root=ROOT):
    r=readjson(root/'reviews/DATA_REVIEW.json');b=readjson(root/'E1_BUNDLE.json')
    assert r['verdict']=='ACCEPT' and r['E1_bundle_sha256']==digest(root/'E1_BUNDLE.json')
    for name,h in b['files'].items():assert digest(root/name)==h,name
    for name,h in b['fixed_rows_sha256'].items():assert digest(root/name)==h,name
    assert digest(protocol(root)['plan_path'])==protocol(root)['plan_sha256']
    return r
require_data_release=require_data
def require_design(root=ROOT):
    require_data(root);r=readjson(root/'CODE_RELEASE.json')
    assert r['verdict']=='ACCEPT' and r['E1_bundle_sha256']==digest(root/'E1_BUNDLE.json')
    assert r['environment_sha256']==digest(root/'environment.lock')
    assert set(r['code_sha256'])=={p.name for p in (root/'code').glob('*.py')}
    for name,h in r['code_sha256'].items():assert digest(root/'code'/name)==h,name
    import sklearn
    assert sklearn.__version__=='1.6.1'
    return r
@contextlib.contextmanager
def timer(root,stage,**labels):
    start=time.perf_counter();cpu=time.process_time();event=dict(stage=stage,pid=os.getpid(),**labels)
    try:yield event
    finally:
        event.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,RSS_scope='process lifetime high water; not incremental stage RSS')
        d=root/'cost';d.mkdir(exist_ok=True)
        with (d/f'events_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
def event_list(root=ROOT):
    return [json.loads(s) for p in sorted((root/'cost').glob('events_*.jsonl')) for s in p.read_text().splitlines()]
def load_source(root=ROOT):
    p=Path(protocol(root)['source_arrays'])
    assert digest(p)==readjson(root/'cohort_contract.json')['legacy']['source_arrays_sha256']
    with np.load(p,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    rr=rows(root/'cohort_manifest.csv')
    assert [r['model_id'] for r in rr]==data['model_ids'].astype(str).tolist()
    assert [r['org_id'] for r in rr]==data['org_ids'].astype(str).tolist()
    return data,rr
def qualified(data):
    return [r for r in data if str(r['valid_source_fit']).lower()=='true' and str(r['valid_summaries']).lower()=='true']
def design(data,names):return np.array([[float(r[n]) for n in names] for r in data],dtype=np.float64)
def sethash(values):return hashlib.sha256('\n'.join(sorted(map(str,values))).encode()).hexdigest()
def verify_features(root=ROOT):
    r=readjson(root/'features/FEATURES_READY.json')
    for name,h in r['files'].items():assert digest(root/name)==h,name
    return r
