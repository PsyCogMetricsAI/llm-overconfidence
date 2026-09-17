#!/usr/bin/env python3
"""Independent supporting-math check, no experimental data or execution-agent import."""
from pathlib import Path
import json,hashlib
import numpy as np
import sympy as s
N,S1,S2,cF,cS,cFS,k,q=s.symbols('N S1 S2 cF cS cFS k q',positive=True)
M=s.Matrix([[N*cF*k*k,cF*k*S1,cFS*k*N],[cF*k*S1,cF*S2,cFS*S1],[cFS*k*N,cFS*S1,cS*N]])/q**2
assert s.factor(M.det())==s.factor(N*k*k*cF*(cF*cS-cFS*cFS)*(N*S2-S1*S1)/q**6)
a,b,A,C,z,alpha,L,g,st,tt=s.symbols('a b A C z alpha L g st tt')
assert s.simplify((a/st)*(st*A-st*b)-a*(A-b))==0
assert s.simplify((alpha/tt)*(st*A+(tt*(A+C)-st*A)+tt*z)+g*L-(alpha*(A+C+z)+g*L))==0
ar,cr,cw=s.symbols('ar cr cw');assert s.expand(ar*cr+(1-ar)*cw-ar-((1-ar)*cw-ar*(1-cr)))==0
# Different seed and item count verify weighted partition and projection statement.
rng=np.random.default_rng(90731);maxerr=0.
for _ in range(50):
 ability=rng.normal(size=31);conf=rng.normal(size=31);X=np.c_[np.ones(31),ability,rng.normal(size=31)];Q=np.eye(31)-X@np.linalg.pinv(X);st,tt=np.exp(rng.normal(size=2));new=tt*(ability+conf)-st*ability
 maxerr=max(maxerr,float(np.max(abs(Q@new-tt*(Q@conf)))))
assert maxerr<1e-10
# Main constructed pair: same means and total20errors; wrong-answer locations determine retained risk.
cs=np.r_[np.repeat(.9,50),np.repeat(.7,50)];rates=[]
for wrong in [range(50,70),range(20)]:
 y=np.ones(100);y[list(wrong)]=0;assert y.mean()==.8 and np.isclose(cs.mean(),.8);rates.append(float((1-y[:50]).mean()))
assert rates==[0.,.4]
out={'verdict':'PASS','scope':'Symbolic determinant, both latent logit invariances, mean-gap identity, independent projection checks and main constructed pair. Not an empirical fit or global identification proof.','random_seed':90731,'projection_max_abs_error':maxerr,'constructed_retained_errors':rates,'checker':'independent algebra checker','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(Path(__file__).parent/'ALGEBRA_INDEPENDENT_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
