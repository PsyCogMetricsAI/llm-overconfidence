"""Durable process metadata/logs for already released production nodes."""
from common import *
import argparse,subprocess,sys,datetime
COMMANDS={'source-pilot':['source_stage.py','pilot'],'source':['source_stage.py','run'],'predict':['run_mainline.py','predict'],'score':['score_comparison.py'],'warm':['costs.py','warm'],'cost':['costs.py','report']}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=COMMANDS);args=ap.parse_args();require_design()
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True);state=runtime/f'{args.stage}.json'
    if state.exists() and readjson(state).get('status')=='COMPLETE':raise RuntimeError('completed stage is immutable; no automatic rerun')
    command=[sys.executable,str(ROOT/'code'/COMMANDS[args.stage][0]),*COMMANDS[args.stage][1:]]
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();start=time.monotonic()
    with (runtime/f'{args.stage}.log').open('a') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        data=dict(stage=args.stage,status='RUNNING',launcher_pid=os.getpid(),child_pid=child.pid,start_utc=now(),command=command,log=str(runtime/f'{args.stage}.log'))
        dump(state,data);rc=child.wait()
    data.update(status='COMPLETE' if rc==0 else 'FAILED',returncode=rc,end_utc=now(),elapsed_seconds=time.monotonic()-start);dump(state,data)
    print(json.dumps(data),flush=True);sys.exit(rc)
