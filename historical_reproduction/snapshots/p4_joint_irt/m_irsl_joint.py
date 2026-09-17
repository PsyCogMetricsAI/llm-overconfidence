#!/usr/bin/env python3
"""M3' — REDO ② with the project's IRSL Beta-IRT (fair same-model comparison) + re-run the JOINT model
on the 6748-model matrix (corrected char-norm口径), GPU-accelerated.

(A) IRSL Beta-IRT (beta_irt.calibrate_2pl), SAME model, two response variables:
      accuracy-IRT  = binary loss on correctness (BIN)        -> theta_acc
      confidence-IRT= beta loss on confidence    (CONF in (0,1)) -> theta_conf
    Compare theta_acc vs theta_conf (this is the proper "logprob/confidence-IRT vs accuracy-IRT").
(B) Joint dual-latent (joint_irt.fit_joint): C=BIN, Y=CONF -> theta_A (ability), theta_C (overconfidence).
    Does theta_C SEPARATE from ability at 6748 scale, or COLLAPSE onto the difference score (C3)?
Usage: m_irsl_joint.py <arc|hellaswag>
"""
import os, sys, json
import numpy as np
from scipy import stats
import torch
from beta_irt import calibrate_2pl
from joint_irt import fit_joint

HERE = os.path.dirname(os.path.abspath(__file__))
bench = sys.argv[1]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
d = np.load(os.path.join(HERE, 'aligned', f'matrix_{bench}.npz'), allow_pickle=True)
BIN = d['BIN'].astype(np.float64)                       # items x models, correctness 0/1
LPm = d['LP_mean'].astype(np.float64)                   # items x models, gold per-token mean logprob
LPs = d['LP_sum'].astype(np.float64)
CONF = d['CONF'].astype(np.float64)                    # chosen_prob_cn (OUTSIDE the Beta channel)
items, models = [str(x) for x in d['items']], [str(x) for x in d['models']]
# ALIGN to run_joint_irt.py: confidence response Y = exp(gold logprob_mean) in (0,1) (NOT chosen_prob);
# length L̃ = token count (logprob_sum/logprob_mean), per-item cross-model median, centered.
RESP = np.exp(LPm).clip(1e-6, 1 - 1e-6)                 # gold-option probability (IRSL/joint Beta response)
with np.errstate(divide='ignore', invalid='ignore'):
    tok = LPs / LPm
Lt = np.nanmedian(tok, axis=1); Lt = Lt - np.nanmean(Lt)   # centered token-count proxy (L̃)


def P(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); v = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[v], b[v])[0, 1]) if v.sum() > 5 else float('nan')


def S(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); v = ~np.isnan(a) & ~np.isnan(b)
    return float(stats.spearmanr(a[v], b[v]).statistic) if v.sum() > 5 else float('nan')


acc = np.nanmean(BIN, 0); conf = np.nanmean(RESP, 0); signed = conf - acc   # A, C(=gold-prob level), C-A
mean_logprob = np.nanmean(LPm, 0)                       # degeneracy-trap guard target
out = {'bench': bench, 'n_items': len(items), 'n_models': len(models), 'device': dev,
       'response': 'Y=exp(gold logprob_mean); Lt=token-count (aligned to run_joint_irt.py)'}

# (A) IRSL Beta-IRT: same model, two responses. resmat = models x items.
torch.manual_seed(0)
fit_acc = calibrate_2pl(torch.tensor(BIN.T, dtype=torch.float32), device=dev, loss_kind='binary', max_epochs=50)
fit_conf = calibrate_2pl(torch.tensor(RESP.T, dtype=torch.float32), device=dev, loss_kind='beta', max_epochs=50)
th_acc, th_conf = fit_acc['theta'], fit_conf['theta']
out['A_irsl_beta_irt'] = {
    'theta_acc_vs_accuracy': P(th_acc, acc),                 # sanity: should be high
    'theta_conf_vs_confidence': P(th_conf, conf),            # sanity
    'corr_thetaAcc_thetaConf_pearson': P(th_acc, th_conf),   # ★ logprob/conf-IRT vs accuracy-IRT
    'corr_thetaAcc_thetaConf_spearman': S(th_acc, th_conf),
    'theta_conf_vs_accuracy': P(th_conf, acc),
    'acc_loss_drop': [round(fit_acc['loss_history'][0], 4), round(fit_acc['loss_history'][-1], 4)],
    'conf_loss_drop': [round(fit_conf['loss_history'][0], 4), round(fit_conf['loss_history'][-1], 4)]}

# (B) Joint dual-latent: C=BIN, Y=exp(gold logprob), Lt=token-count (ALIGNED to run_joint_irt.py)
# HS (9586 items) needs lower lr + grad-clip to avoid eta2 saturation -> lgamma(0) NaN; ARC unchanged (lr=0.05, no clip).
LR = 0.02 if bench == 'hellaswag' else 0.05
GC = 5.0 if bench == 'hellaswag' else None
jt = fit_joint(BIN.T, RESP.T, Lt, n_epochs=2500, lr=LR, grad_clip=GC, device=dev)
thA, thC = jt['theta_A'], jt['theta_C']
# ★ M10 prereq: persist per-model vectors for the cross-benchmark θ^C-residual test (decisive re-review).
meanCONF = np.nanmean(CONF, 0)                          # chosen-answer confidence per model (NOT seen by Beta fit)
with np.errstate(divide='ignore', invalid='ignore'):
    tok_len = np.nanmean(LPs / LPm, 0)                  # per-model mean answer token-length (length confound)
np.savez(os.path.join(HERE, 'aligned', f'joint_vectors_{bench}.npz'),
         models=np.array(models), theta_A=thA, theta_C=thC, gamma_m=jt['gamma_m'],
         acc=acc, meanY=conf, meanCONF=meanCONF, gauge=mean_logprob, tok_len=tok_len,
         th_acc_irsl=th_acc, th_conf_irsl=th_conf, signed=signed)
out['B_joint'] = {
    'thetaA_vs_accuracy': P(thA, acc),                       # ability channel sanity
    'guard_thetaA_vs_mean_logprob': P(thA, mean_logprob),    # ★ degeneracy trap: should NOT be ~1
    'guard_thetaC_vs_mean_logprob': P(thC, mean_logprob),
    'corr_thetaC_thetaA_pearson': P(thC, thA),               # ★ collapse if |.|>0.7 (discriminant gate)
    'corr_thetaC_thetaA_spearman': S(thC, thA),
    'corr_thetaC_signedbias': P(thC, signed),                # ★ if ~1 -> theta_C is just C-A (difference score)
    'corr_thetaA_thetaAcc_irsl': P(thA, th_acc),
    'loss_drop': [round(jt['loss_history'][0], 4), round(jt['loss_history'][-1], 4)],
    'note': 'Lt=token-count L̃; 2500 epochs'}
out['B_joint']['verdict'] = ('theta_C COLLAPSES onto ability/difference-score (NOT an independent latent)'
                             if abs(out['B_joint']['corr_thetaC_thetaA_pearson']) > 0.7
                             or abs(out['B_joint']['corr_thetaC_signedbias']) > 0.9
                             else 'theta_C appears separable -> investigate')

OUT = os.path.join(HERE, 'results', f'results_irsl_joint_{bench}.json')
json.dump(out, open(OUT, 'w'), indent=2)
print('WROTE', OUT, '| device', dev)
a = out['A_irsl_beta_irt']
print(f"(A) IRSL Beta-IRT: thetaAcc~acc={a['theta_acc_vs_accuracy']:+.3f} thetaConf~conf={a['theta_conf_vs_confidence']:+.3f} "
      f"| ★thetaAcc~thetaConf P={a['corr_thetaAcc_thetaConf_pearson']:+.3f} S={a['corr_thetaAcc_thetaConf_spearman']:+.3f}")
b = out['B_joint']
print(f"(B) Joint: thetaA~acc={b['thetaA_vs_accuracy']:+.3f} | ★thetaC~thetaA={b['corr_thetaC_thetaA_pearson']:+.3f} "
      f"thetaC~(C-A)={b['corr_thetaC_signedbias']:+.3f}")
print(f"    verdict: {b['verdict']}")
