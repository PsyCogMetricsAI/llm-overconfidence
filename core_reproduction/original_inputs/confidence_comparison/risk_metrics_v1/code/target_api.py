"""Subprocess-only target API for fold workers and post-seal evaluator."""
import subprocess,sys,json
from pathlib import Path
def request(stage,payload):
    p=subprocess.run([sys.executable,str(Path(__file__).with_name('target_admin.py')),stage],input=json.dumps(payload),text=True,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr)
    return json.loads(p.stdout)
def training_targets(outer_fold,allowed_orgs,model_ids,context):return request('query',dict(outer_fold=outer_fold,allowed_orgs=list(allowed_orgs),model_ids=list(model_ids),context=context,target='hellaswag'))
def scoring_targets(seal_sha256):return request('score',dict(seal_sha256=seal_sha256))
