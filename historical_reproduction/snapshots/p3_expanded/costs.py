"""Prespecified warm measurements and complete cold-cost accounting, no target reader."""
from common import *
from source_stage import summaries,read_arc
from legacy_primitives import project_source
from predictors import predict_saved
import joblib,platform,sys

def warm(root=ROOT):
    require_design(root)
    for name,h in readjson(root/'PREDICTION_SEAL.json')['artifacts'].items():assert digest(root/name)==h
    if (root/'cost/warm_measurements.csv').exists():raise RuntimeError('warm run already complete')
    cohort=rows(root/'evaluation_cohort.csv');manifest={r['model_id']:r for r in rows(root/'cohort_manifest.csv')};groups=protocol(root)['feature_groups']
    chosen=sorted(cohort,key=lambda r:(hashlib.sha256(f"20260914|{r['model_id']}".encode()).hexdigest(),r['model_id']))[:100]
    logs=readjson(root/'selection_log.json');fits={};brefs={};saved_paths=[]
    for r in logs:
        p=r['best_estimator'];assert digest(root/p['path'])==p['sha256'];fits[r['outer_fold'],r['group'],r['endpoint']]=joblib.load(root/p['path']);saved_paths.append(p)
    sp=readjson(root/'split_manifest.json');Fids=[i for i,h in sp['source_item_half'].items() if h==0]
    # b order follows source_arrays item order, preserved by source_item_half insertion order.
    for r in chosen:
        key=r['reference_sha256'];p=root/'features/reference_cache'/f'{key}.npz';meta=readjson(p.with_suffix('.json'));assert digest(p)==meta['b_file_sha256'];brefs[key]=np.load(p)['b']
    out=[];orderlog=[];rng=np.random.default_rng(20260914)
    for rep in range(3):
        for row in chosen:
            mid=row['model_id'];k=int(manifest[mid]['outer_fold']);order=rng.permutation(['B','C','L','F']).tolist();orderlog.append(dict(rep=rep,model_id=mid,group_order=order))
            start=time.perf_counter();cpu=time.process_time();raw=read_arc(manifest[mid]);io=time.perf_counter()-start;ioc=time.process_time()-cpu
            index={str(s):i for i,s in enumerate(raw['item_ids'])}
            for group in order:
                start=time.perf_counter();cpu=time.process_time();stats,obs=summaries(raw['loglik'],raw['char_lens'],raw['gold'],raw['item_ids'],raw['n_options'],extended=group!='B')
                if group in ['L','F']:
                    e=np.full(len(Fids),np.nan);v=np.zeros(len(Fids),bool)
                    for j,s in enumerate(Fids):
                        if s in index:e[j]=obs[0][index[s]];v[j]=obs[3][index[s]]
                    projection=project_source(e,v,brefs[row['reference_sha256']]);assert projection['valid_source_fit'];stats.update(projection)
                X=np.array([[float(stats[n]) for n in groups[group]]]);yp={ep:float(predict_saved(fits[k,group,ep],X)[0]) for ep in ['Y_sel','Y_cal']}
                out.append(dict(model_id=mid,outer_fold=k,rep=rep,group=group,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,shared_io_wall_seconds=io,shared_io_cpu_seconds=ioc,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**yp))
    write_rows(root/'cost/warm_measurements.csv',out);dump(root/'cost/warm_order.json',orderlog)

def report(root=ROOT):
    require_design(root);events=event_list(root);wm=rows(root/'cost/warm_measurements.csv');pr=protocol(root)
    total=lambda stage,key='wall_seconds',group=None:sum(float(e.get(key,0)) for e in events if e['stage']==stage and (group is None or e.get('group')==group))
    sharedio=total('source_io');sharedtarget=total('target_scoring');als=total('reference_ALS');projection=total('source_projection');summaries_C=total('source_summaries');summaries_B=total('source_summaries','base_component_wall_seconds')
    log=readjson(root/'selection_log.json');summary={};final_by={(r['outer_fold'],r['group'],r['endpoint']):r['configs'][r['best_index']]['id'] for r in log}
    for g in ['B','C','L','F']:
        tuning=total('supervised_inner_tuning',group=g)
        # Best-deployment cold account pays both-family tuning and only selected-family final fit.
        final=sum(e['wall_seconds'] for e in events if e['stage']=='supervised_final_fit' and e['group']==g and e['configuration']==final_by[e['outer'],g,e['endpoint']])
        source=(summaries_B if g=='B' else summaries_C)+(als+projection if g in ['L','F'] else 0)
        vv=np.array([float(r['wall_seconds']) for r in wm if r['group']==g]);vvio=np.array([float(r['wall_seconds'])+float(r['shared_io_wall_seconds']) for r in wm if r['group']==g])
        final_cpu=sum(e['cpu_seconds'] for e in events if e['stage']=='supervised_final_fit' and e['group']==g and e['configuration']==final_by[e['outer'],g,e['endpoint']])
        source_cpu=total('source_summaries','base_component_cpu_seconds' if g=='B' else 'cpu_seconds')+(total('reference_ALS','cpu_seconds')+total('source_projection','cpu_seconds') if g in ['L','F'] else 0)
        cpu_total=total('source_io','cpu_seconds')+total('target_scoring','cpu_seconds')+source_cpu+total('supervised_inner_tuning','cpu_seconds',g)+final_cpu
        summary[g]=dict(cold_standalone_wall_seconds=sharedio+sharedtarget+source+tuning+final,cold_standalone_cpu_seconds=cpu_total,cold_shared_IO_wall_seconds=sharedio,cold_shared_target_scoring_wall_seconds=sharedtarget,cold_summaries_wall_seconds=summaries_B if g=='B' else summaries_C,cold_required_ALS_projection_wall_seconds=als+projection if g in ['L','F'] else 0,cold_all17_tuning_wall_seconds=tuning,cold_selected_final_wall_seconds=final,
          warm_no_IO_median_seconds=float(np.median(vv)),warm_no_IO_IQR_seconds=np.quantile(vv,[.25,.75]).tolist(),warm_with_shared_IO_median_seconds=float(np.median(vvio)),warm_with_shared_IO_IQR_seconds=np.quantile(vvio,[.25,.75]).tolist(),warm_measurements=len(vv))
    primary=readjson(root/'results_comparison.json')['comparisons']['F_best_vs_C_best/Y_sel'];ici=primary['I_CI95']
    ratio_cold=summary['F']['cold_standalone_wall_seconds']/summary['C']['cold_standalone_wall_seconds'];ratio_warm=summary['F']['warm_with_shared_IO_median_seconds']/summary['C']['warm_with_shared_IO_median_seconds']
    src=rows(root/'source_features.csv');cohort=rows(root/'cohort_manifest.csv')
    out=dict(model_id=MODEL_ID,hardware=dict(platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),python=sys.version),threads=1,groups=summary,
      cold_actual_all_benchmark_stages_wall_seconds=sum(e['wall_seconds'] for e in events),cold_actual_all_benchmark_stages_cpu_seconds=sum(e['cpu_seconds'] for e in events),
      actual_unique_reference_fits=sum(e['stage']=='reference_ALS' for e in events),actual_projection_contexts=sum(e['stage']=='source_projection' for e in events),actual_inner_supervised_fits=sum(e['stage']=='supervised_inner_tuning' for e in events),actual_final_supervised_fits=sum(e['stage']=='supervised_final_fit' for e in events),
      max_process_lifetime_RSS_bytes=max(e['peak_RSS_bytes'] for e in events),RSS_scope='per-stage lifetime high-water marks retained in cost/events.jsonl; not independent stage increments',
      source_labels_all_groups=sum(int(r['n_valid_source']) for r in src),target_item_labels_all_groups=sum(int(r['n_valid_HS']) for r in cohort),extra_LLM_calls_all_groups=0,
      label_scope='common observed source/target label budget; no new inference; target counts metadata only; per-fit Ntrain and inner validation requests separately logged',
      amortization_population=dict(Q0_models=len(cohort),evaluation_models=len(rows(root/'evaluation_cohort.csv')),reference_populations='features/reference_cache/*.json:identity.reference_model_ids; per-fold supervised train IDs in selection_log'),
      benchmark_audit_cost='F-based common-cohort eligibility audit/reference fits included in actual benchmark total; standalone B/C deployment excludes unnecessary ALS, L/F each pays requisite shared ALS as standalone',
      physical_sharing='summaries C and parameter fits computed once; standalone accounts allocate prerequisites fully to each group; no claim all allocations sum to physical total',
      wall_time_scope='event sums are measured stage work times, not elapsed parallel makespan; standalone cold sums charge common target scoring and source I/O consistently; actual operator elapsed times logged separately',
      supervised_fit_vs_prediction=dict(fit_wall_seconds=sum(e.get('fit_wall_seconds',0) for e in events),prediction_wall_seconds=sum(e.get('prediction_wall_seconds',0) for e in events)),
      stage_events='cost/events.jsonl',warm_measurements='cost/warm_measurements.csv',warm_fixed_models=len({r['model_id'] for r in wm}),warm_repetitions=3,
      disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and '.venv' not in p.parts and 'manuscript_before_v2' not in p.parts and 'repair' not in p.parts),
      cost_ratios_F_over_C=dict(cold=ratio_cold,warm_including_shared_IO=ratio_warm),secondary_low_cost_tradeoff=dict(cold=bool(ici is not None and ici[0]>-.01 and ratio_cold<=.8),warm=bool(ici is not None and ici[0]>-.01 and ratio_warm<=.8)),
      limitations='Measured first necessary fit phases on one machine. Saved-estimator loading/reference setup excluded only in explicitly warm marginal scenario. Timing overhead and artifact writes not perfectly separable. No cross-hardware cost uncertainty or overall_WIN.')
    dump(root/'cost_report.json',out)
    print(json.dumps(dict(cost_ratios=out['cost_ratios_F_over_C'],fits=out['actual_inner_supervised_fits'])),flush=True)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['warm','report']);args=a.parse_args()
    (warm if args.stage=='warm' else report)()
