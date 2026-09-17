"""Trainer API: outcomes only from an independent, contract-checking subprocess."""
import json, subprocess, sys
from pathlib import Path

def _request(command, payload):
    p=subprocess.run([sys.executable,str(Path(__file__).with_name('score_targets.py')),command],input=json.dumps(payload),text=True,capture_output=True,check=True)
    return json.loads(p.stdout)

def training_targets(outer_fold,allowed_orgs,model_ids,context,target='hellaswag'):
    return _request('query',dict(outer_fold=int(outer_fold),allowed_orgs=list(allowed_orgs),model_ids=list(model_ids),context=context,target=target))

def scoring_targets(seal_path,seal_sha):
    return _request('scoring-query',dict(seal_path=str(seal_path),seal_sha=seal_sha))
