#!/usr/bin/env python3
"""Supporting symbolic/numerical checks; no empirical fit, no novelty assertion."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import sympy as sp
ROOT=Path(__file__).resolve().parents[1]
N,S1,S2,s,sigma,cF,cS,cFS=sp.symbols('N S1 S2 s sigma cF cS cFS',real=True)
M=sp.Matrix([[cF*s*s*N,cF*s*S1,cFS*s*N],[cF*s*S1,cF*S2,cFS*S1],[cFS*s*N,cFS*S1,cS*N]])/sigma**2
expected=cF*(cF*cS-cFS**2)*s**2*N*(N*S2-S1**2)/sigma**6
assert sp.simplify(M.det()-expected)==0
A,Cright,Cwrong=sp.symbols('A Cright Cwrong')
assert sp.expand(A*Cright+(1-A)*Cwrong-A-((1-A)*Cwrong-A*(1-Cright)))==0
rng=np.random.default_rng(20260916)
max_logit=max_resid=max_partition=0.
for _ in range(100):
 nmod,nitem=25,13
 center=lambda x:x-x.mean()
 ta,tc,g=[center(rng.normal(size=nmod)) for _ in range(3)];b=rng.normal(size=nitem);z=center(rng.normal(size=nitem));aa,al=[np.exp(rng.normal(size=nitem)) for _ in range(2)];L=center(rng.normal(size=nitem));ss,tt=np.exp(rng.normal(size=2))
 tap=ss*ta;tcp=tt*(ta+tc)-ss*ta;bp=ss*b;aap=aa/ss;zp=tt*z;alp=al/tt
 e1=aa*(ta[:,None]-b);e1p=aap*(tap[:,None]-bp)
 e2=al*(ta[:,None]+tc[:,None]+z)+g[:,None]*L;e2p=alp*(tap[:,None]+tcp[:,None]+zp)+g[:,None]*L
 max_logit=max(max_logit,float(abs(e1-e1p).max()),float(abs(e2-e2p).max()))
 X=np.column_stack([np.ones(nmod),ta,rng.normal(size=nmod)])
 res=lambda v:v-X@np.linalg.lstsq(X,v,rcond=None)[0]
 max_resid=max(max_resid,float(abs(res(tcp)-tt*res(tc)).max()))
 y=rng.integers(0,2,100);c=rng.random(100);parts=np.array_split(rng.permutation(100),4)
 max_partition=max(max_partition,abs(sum(len(i)/100*(c[i].mean()-y[i].mean()) for i in parts)-(c.mean()-y.mean())))
assert max_logit<1e-10 and max_resid<1e-10 and max_partition<1e-12
examples=[]
for errors in [np.arange(50,70),np.arange(20)]:
 c=np.r_[np.full(50,.9),np.full(50,.7)];y=np.ones(100);y[errors]=0
 gold=np.zeros(100,int);w=np.where(y==1,0,1);p=np.repeat(((1-c)/3)[:,None],4,axis=1);p[np.arange(100),w]=c
 assert np.allclose(p.sum(1),1) and np.array_equal((p.argmax(1)==gold).astype(float),y)
 examples.append({'accuracy':float(y.mean()),'mean_confidence':float(c.mean()),'signed_gap':float(c.mean()-y.mean()),'top_half_risk':float((1-y[:50]).mean())})
assert np.isclose(examples[0]['top_half_risk'],0) and np.isclose(examples[1]['top_half_risk'],.4)
# General finite intersection bound construction, for all small integer N/e/k/t.
count=0
for n in range(2,18):
 for e in range(n+1):
  for k in range(1,n):
   for t in range(max(0,e-(n-k)),min(e,k)+1):
    conf=.6;eps=.5*min((conf-.25)*n/k,(1-conf)*n/(n-k));hi=conf+eps*(n-k)/n;lo=conf-eps*k/n
    assert .25<lo<hi<1 and abs((k*hi+(n-k)*lo)/n-conf)<1e-12
    err=np.r_[np.ones(t),np.zeros(k-t),np.ones(e-t),np.zeros(n-k-e+t)]
    assert err.sum()==e and err[:k].sum()==t;count+=1
out={'status':'PASS','scope':'Algebraic and finite-case validation, not empirical reproduction','symbolic_fisher_determinant':str(sp.factor(expected)),'mean_gap_identity':True,'partition_max_abs_error':max_partition,'joint_logit_invariance_max_abs_error':max_logit,'controlled_residual_scaling_max_abs_error':max_resid,'constructed_pair':examples,'finite_intersection_cases':count,'versions':{'python':sys.version,'numpy':np.__version__,'sympy':sp.__version__}}
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT.parent/'runs/algebra');d=parser.parse_args().out;d.mkdir(parents=True,exist_ok=True);(d/'ALGEBRA_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
