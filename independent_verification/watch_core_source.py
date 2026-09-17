#!/usr/bin/env python3
"""Wait for fresh source phase, then run independent comparison/audit gates.
Does not run or modify scientific production code. Any failed check aborts release.
"""
from pathlib import Path
import subprocess,sys,time
s=Path(__file__).resolve().parents[1];start=time.monotonic();w=s/'core_reproduction/work';core=w/'confidence_comparison/risk_metrics_v1';review=s/'independent_verification'
assert not (core/'SOURCE_RELEASE.json').exists()
while not (w/'SOURCE_PHASE_COMPLETE.json').exists():
 assert time.monotonic()-start<43200,'source readiness timeout'
 time.sleep(5)
for args in [[str(review/'compare_core.py'),'source','--core',str(core),'--frozen',str(s/'reference_originals/risk_metrics_v1')],[str(review/'release_core.py'),'source','--core',str(core),'--migration-root',str(s/'core_reproduction')]]:
 subprocess.run([sys.executable,*args],check=True,timeout=43200)
print('INDEPENDENT CORE SOURCE GATE COMPLETE',flush=True)
