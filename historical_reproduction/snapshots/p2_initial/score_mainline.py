#!/usr/bin/env python3
"""CV6 prewritten scoring; executes only after independent all-prediction release.
Frozen paired organization bootstrap, no fitting and no target access at import.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,json
import numpy as np
from cv_core import SEED,digest,group_mse
ROOT=Path(__file__).resolve().parents[1]

def evaluate(y,pbase,pfull,orgs,outer,bootstrap_idx):
    y=np.asarray(y,float);pbase=np.asarray(pbase,float);pfull=np.asarray(pfull,float);orgs=np.asarray(orgs);outer=np.asarray(outer)
    names=np.unique(orgs);lb=np.array([np.mean((y[orgs==g]-pbase[orgs==g])**2) for g in names]);lf=np.array([np.mean((y[orgs==g]-pfull[orgs==g])**2) for g in names])
    B=float(lb.mean());F=float(lf.mean());D=B-F
    folds={str(k):float(group_mse(y[outer==k],pbase[outer==k],orgs[outer==k])-group_mse(y[outer==k],pfull[outer==k],orgs[outer==k])) for k in range(5) if np.any(outer==k)}
    out={'base_loss':B,'full_loss':F,'delta':D,'fold_delta':folds,'positive_outer_folds':sum(v>0 for v in folds.values()),'relative_improvement':None,'delta_ci97_5':None,'relative_improvement_ci97_5':None,'supported':False,'inconclusive_reason':None}
    if B<=1e-12:out['inconclusive_reason']='BASELINE_LOSS_LE1E12';return out
    out['relative_improvement']=D/B
    bboot=lb[bootstrap_idx].mean(1);fboot=lf[bootstrap_idx].mean(1);dboot=bboot-fboot
    out['delta_ci97_5']=np.quantile(dboot,[.0125,.9875]).tolist()
    if np.any(bboot<=1e-12) or not np.isfinite(dboot).all():out['inconclusive_reason']='CI_NOT_COMPUTABLE';return out
    out['relative_improvement_ci97_5']=np.quantile(dboot/bboot,[.0125,.9875]).tolist()
    out['supported']=bool(out['relative_improvement']>=.05 and out['delta_ci97_5'][0]>0 and out['positive_outer_folds']>=4 and len(folds)==5)
    return out

def decide(endpoints,sufficient=True):
    if not sufficient or any(v['inconclusive_reason'] for v in endpoints.values()):return 'INCONCLUSIVE'
    cal=endpoints['Y_cal']['supported'];sel=endpoints['Y_sel']['supported']
    return 'JOINT_SUPPORT' if cal and sel else ('CAL_ONLY' if cal else ('SELECT_ONLY' if sel else 'NO_DEMONSTRATED_GAIN'))

def main():
    from run_mainline import require_release
    require_release();path=ROOT/'predictions_oof.csv';sha=digest(path);seal=json.loads((ROOT/'PREDICTION_SEAL.json').read_text());assert seal['predictions_sha256']==sha and seal['all_five_outer_folds_present']
    assert (ROOT/'prediction.sha256').read_text().split()[0]==sha
    from score_api import scoring_targets
    target=scoring_targets(str(path),sha)
    if isinstance(target,dict):target=target['rows']
    ymap={r['model_id']:r for r in target}
    rows=list(csv.DictReader(path.open()));assert set(ymap)=={r['model_id'] for r in rows}
    orgs=np.array([r['org_id'] for r in rows]);outer=np.array([int(r['outer_fold']) for r in rows]);names=np.unique(orgs)
    for r in rows:assert ymap[r['model_id']]['org_id']==r['org_id']
    rng=np.random.default_rng(SEED);idx=rng.integers(0,len(names),size=(2000,len(names)));np.save(ROOT/'bootstrap_indices.npy',idx)
    endpoints={}
    for ep in ['Y_cal','Y_sel']:
        Y=np.array([ymap[r['model_id']][ep] for r in rows],float);B=np.array([float(r['base_'+ep]) for r in rows]);F=np.array([float(r['full_'+ep]) for r in rows]);endpoints[ep]=evaluate(Y,B,F,orgs,outer,idx)
    sufficient=len(rows)>=300 and len(names)>=30 and seal['sample_sufficient'];verdict=decide(endpoints,sufficient)
    results={'model_id':'scientific reproduction','verdict':verdict,'direction':'ARC_to_HellaSwag','baseline_item_scope':'all valid ARC items F union C; additional parameters from F only','n_models':len(rows),'n_orgs':len(names),'sample_sufficient':sufficient,'endpoints':endpoints,'predictions_sha256':sha,'bootstrap_seed':SEED,'bootstrap_reps':2000,'bootstrap_org_order':names.tolist(),'bootstrap_indices_sha256':digest(ROOT/'bootstrap_indices.npy'),'ci_percentiles':[1.25,98.75],'ci_scope':'conditional on fixed OOF predictions and existing target items; organization-paired resampling; no refitting or target-item bootstrap','primary_support_rule':'relative point improvement >=0.05 AND delta CI lower >0 AND >=4/5 outer-fold delta>0; INCONCLUSIVE has priority','no_zero_target_information_claim':True}
    (ROOT/'results_CV.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    text=['Model ID: Codex / GPT-6 family（精确 serving ID 未暴露）','','# CV 精简主线报告',f'','判决：**'+verdict+'**。独立 V-CV 尚待审核。',f'','ARC→HellaSwag；合格模型 '+str(len(rows))+'，机构 '+str(len(names))+'（results_CV.json:n_models,n_orgs）。基础五统计使用全部有效ARC题F∪C，追加(theta,log s,log sigma)仅由F题估计；只比较这两个模型。','','| 终点 | 基础机构MSE | 完整机构MSE | Δ | 相对改进 | Δ的97.5%CI | 正Δ外折 | 支持 |','|---|---:|---:|---:|---:|---|---:|---|']
    for ep,e in endpoints.items():text.append('| '+ep+' | '+' | '.join(str(e[k]) for k in ['base_loss','full_loss','delta','relative_improvement','delta_ci97_5','positive_outer_folds','supported'])+' |')
    text+=['','各数见 results_CV.json:endpoints.<终点>.<字段>；选lambda见 selection_log.json；机构bootstrap固定预测、不重训，同一抽样用于两终点。区间不覆盖新题库或完整训练过程不确定性；5%为方案操作门槛，区间不证明改进至少5%。','','外折训练机构的目标Y用于监督训练/调参，测试机构目标Y未输入对应预测器。所有五折预测SHA封存后才整体评分；这不是零目标域信息，也不是人员级历史盲。','','结果只支持或未能支持联合参数的跨机构预测增量；不识别单个s/sigma贡献、概率尺度之外效应或因果机制。模型只预测目标可靠性标量，没有改进逐题拒答排序算法。','','未运行温度、反向迁移、额外覆盖率、逐参数消融或全量R复现。']
    (ROOT/'CV_REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
