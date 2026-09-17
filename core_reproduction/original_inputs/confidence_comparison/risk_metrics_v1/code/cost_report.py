"""Summarize measured stage costs without inventing cold end-to-end timing."""
from common import *
def run(root=ROOT):
    require_design(root);ev=event_list(root);by={}
    for r in ev:
        d=by.setdefault(r['stage'],dict(n_events=0,sum_wall_seconds=0.,sum_cpu_seconds=0.,process_peak_RSS_bytes=0))
        d['n_events']+=1;d['sum_wall_seconds']+=r['wall_seconds'];d['sum_cpu_seconds']+=r['cpu_seconds'];d['process_peak_RSS_bytes']=max(d['process_peak_RSS_bytes'],r['peak_RSS_bytes'])
    inner=[r for r in ev if r['stage']=='supervised_inner_tuning'];final=[r for r in ev if r['stage']=='supervised_final_fit']
    inner_unique={tuple(r[k] for k in ['outer','inner','group','endpoint','configuration']) for r in inner}
    final_unique={tuple(r[k] for k in ['outer','group','endpoint','family','role','configuration']) for r in final}
    assert len(inner_unique)==21080 and 620<=len(final_unique)<=930
    warm=readjson(root/'cost/warm_summary.json')
    dump(root/'results/cost_report.json',dict(status='COMPLETE_PENDING_V_RESULT',stages=by,actual_supervised_inner_fits=len(inner),actual_supervised_final_fits=len(final),unique_supervised_inner_fits=len(inner_unique),unique_supervised_final_fits=len(final_unique),repeated_fit_attempts=len(inner)+len(final)-len(inner_unique)-len(final_unique),nominal_total_fits_upper_bound=22010,extra_LLM_calls=0,all_recorded_pilot_and_debug_costs_retained=True,E1_input_hash_wall_seconds=readjson(root/'E1_BUNDLE.json')['elapsed_seconds'],new_output_disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),warm_report_path='cost/warm_summary.json',warm_report_sha256=digest(root/'cost/warm_summary.json'),warm_status=warm.get('status'),complete_cold_end_to_end_seconds=None,complete_cold_status='UNMEASURED: legacy half0 caches reused; per-stage sums and current incremental costs reported, not a cold end-to-end claim',wall_scope='sum of stage durations; parallel wall times are not added into elapsed critical-path runtime',mask_control_cost='all20 computability flags require shared source/reference checks even for B0/Bd; no zero-reference-cost claim',cost_events_sha256={str(p.relative_to(root)):digest(p) for p in sorted((root/'cost').glob('events_*.jsonl'))}))
    print(json.dumps(dict(status='E7_COMPLETE_PENDING_V_RESULT',inner_fits=len(inner),final_fits=len(final))),flush=True)
if __name__=='__main__':run()
