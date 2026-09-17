#!/usr/bin/env python3
"""Independent numerical artifact comparison; never imports production modules."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

def compare_npz(a,b,atol=1e-10):
 records={}
 with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
  assert set(x.files)==set(y.files),('keys',x.files,y.files)
  for k in x.files:
   u,v=x[k],y[k];assert u.shape==v.shape,(k,u.shape,v.shape)
   assert u.dtype==v.dtype,(k,str(u.dtype),str(v.dtype))
   if u.dtype.kind in 'fc':
    assert np.array_equal(np.isnan(u),np.isnan(v)),(k,'nan mask')
    assert np.array_equal(np.isposinf(u),np.isposinf(v)),(k,'positive inf')
    assert np.array_equal(np.isneginf(u),np.isneginf(v)),(k,'negative inf')
    f=np.isfinite(u);d=float(np.max(abs(u[f]-v[f]))) if f.any() else 0.;assert d<=atol,(k,d,atol)
    records[k]={'shape':list(u.shape),'dtype':str(u.dtype),'max_abs_difference':d}
   else:assert np.array_equal(u,v),k;records[k]={'shape':list(u.shape),'dtype':str(u.dtype),'exact':True}
 return records

def compare_csv(a,b,atol=1e-10,ignore=()):
 with open(a) as f:x=list(csv.DictReader(f))
 with open(b) as f:y=list(csv.DictReader(f))
 assert len(x)==len(y),(len(x),len(y));assert x and set(x[0])==set(y[0])
 maximum={};numeric=set();exact=set()
 for i,(u,v) in enumerate(zip(x,y)):
  for k in u:
   if k in ignore:continue
   if u[k]==v[k]:exact.add(k);continue
   try:p,q=float(u[k]),float(v[k])
   except ValueError:raise AssertionError((i,k,u[k],v[k]))
   assert math.isnan(p)==math.isnan(q),(i,k,'nan')
   if math.isnan(p):continue
   if not(math.isfinite(p) and math.isfinite(q)):assert p==q;continue
   d=abs(p-q);assert d<=atol,(i,k,p,q,d);maximum[k]=max(maximum.get(k,0),d);numeric.add(k)
 return {'rows':len(x),'columns':len(x[0]),'ignored_columns':list(ignore),'max_abs_differences':maximum}

def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('old',type=Path);p.add_argument('new',type=Path);p.add_argument('--atol',type=float,default=1e-10);p.add_argument('--ignore',nargs='*',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=compare_npz(a.old,a.new,a.atol) if a.old.suffix=='.npz' else compare_csv(a.old,a.new,a.atol,a.ignore)
 a.output.write_text(json.dumps({'verdict':'PASS','old':str(a.old),'new':str(a.new),'old_sha256':h(a.old),'new_sha256':h(a.new),'atol':a.atol,'checks':result},indent=2)+'\n')
