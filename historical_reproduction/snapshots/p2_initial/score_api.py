"""Trainer-facing API. No import of scorer, no direct target-file read.
Target outcomes are returned only by the independent whitelisted subprocess.
"""
import json,subprocess,sys
from pathlib import Path
_SCORER=Path(__file__).resolve().with_name('score_targets.py')
def _request(command,payload):
    result=subprocess.run([sys.executable,str(_SCORER),command],input=json.dumps(payload),text=True,capture_output=True,check=True)
    return json.loads(result.stdout)
def training_targets(outer_fold,allowed_orgs,model_ids,context=None):
    return _request('query',dict(outer_fold=int(outer_fold),allowed_orgs=list(allowed_orgs),model_ids=list(model_ids),context=context))
def scoring_targets(prediction_path,prediction_sha):
    return _request('scoring-query',dict(prediction_path=str(prediction_path),prediction_sha=prediction_sha))
