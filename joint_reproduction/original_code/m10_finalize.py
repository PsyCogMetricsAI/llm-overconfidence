#!/usr/bin/env python3
"""M10 finalize — master critical re-review verdict on the θ^C independence claim.

Recomputes (does NOT hardcode) the decisive numbers and writes results/results_m10_master_verdict.json.
The automated m10/m10b verdicts (INDEPENDENT/SURVIVES) are OVERRIDDEN here because they did not account
for HS residual degeneracy (resid_share -> 0 once θ^A is removed -> cross-bench corr is against noise).
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def zsc(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def P(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[v], b[v])[0, 1]) if v.sum() > 5 else float('nan')


def rsd(y, cols):
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(c, float) for c in cols] + [np.ones(len(y))])
    v = ~np.isnan(y) & ~np.isnan(X).any(1)
    beta, *_ = np.linalg.lstsq(X[v], y[v], rcond=None)
    r = np.full(len(y), np.nan)
    r[v] = y[v] - X[v] @ beta
    return r


def load(b):
    d = np.load(os.path.join(HERE, 'aligned', f'joint_vectors_{b}.npz'), allow_pickle=True)
    v = {k: d[k] for k in d.files}
    v['models'] = v['models'].astype(str)
    if P(v['theta_A'], v['acc']) < 0:
        v['theta_A'] = -v['theta_A']
    if P(v['theta_C'], v['signed']) < 0:
        v['theta_C'] = -v['theta_C']
    return v


arc, hs = load('arc'), load('hellaswag')
mH = {m: i for i, m in enumerate(hs['models'])}
shared = [m for m in arc['models'] if m in mH]
mA = {m: i for i, m in enumerate(arc['models'])}
iA = np.array([mA[m] for m in shared]); iH = np.array([mH[m] for m in shared])

res = {}
for nm, v in [('arc', arc), ('hs', hs)]:
    base = [zsc(v['theta_A']), zsc(v['meanY'])]
    r = rsd(v['theta_C'], base)
    rg = rsd(v['theta_C'], base + [zsc(v['gauge']), zsc(v['tok_len'])])
    res[nm] = {
        'thetaC_corr_thetaA': P(v['theta_C'], v['theta_A']),
        'thetaC_corr_signed': P(v['theta_C'], v['signed']),
        'R2_thetaC_on_(thetaA,meanY)': 1 - float(np.nanvar(r) / np.nanvar(v['theta_C'])),
        'resid_share_after_thetaA_meanY': float(np.nanvar(r) / np.nanvar(v['theta_C'])),
        'resid_var_absolute': float(np.nanvar(r)),
        'resid_corr_meanCONF': P(r, v['meanCONF']),
        'resid_share_after_+gauge+len': float(np.nanvar(rg) / np.nanvar(v['theta_C'])),
        'resid_corr_meanCONF_after_+gauge+len': P(rg, v['meanCONF']),
    }

# smoking gun: θ^A-leftover-after-acc transfers cross-bench -> acc-only residuals leak ability
sg = P(rsd(arc['theta_A'], [zsc(arc['acc'])])[iA], rsd(hs['theta_A'], [zsc(hs['acc'])])[iH])

verdict = {
    'n_shared_models': len(shared),
    'yardstick_thetaA_crossbench': P(arc['theta_A'][iA], hs['theta_A'][iH]),
    'smokinggun_thetaA_resid_on_acc_crossbench': sg,
    'per_bench': res,
    'DECISION': 'NO_INDEPENDENT_OVERCONFIDENCE_AXIS__C3_HOLDS',
    'reasoning': [
        'HS: R2(theta_C | theta_A, meanY) = %.4f -> residual share %.4f (var %.6f ~ 0): theta_C has NO independent '
        'component on HS; it IS (ability, confidence-level).' % (
            res['hs']['R2_thetaC_on_(thetaA,meanY)'], res['hs']['resid_share_after_thetaA_meanY'], res['hs']['resid_var_absolute']),
        'ARC: a small %.1f%% residual loads on meanCONF (%.2f), weakening to %.2f after gauge+length -> small, '
        'bench-specific, partly gauge/length.' % (
            100 * res['arc']['resid_share_after_thetaA_meanY'], res['arc']['resid_corr_meanCONF'],
            res['arc']['resid_corr_meanCONF_after_+gauge+len']),
        'The +0.61 cross-bench "independent axis" (m10) was ability leakage: theta^A-residual-on-acc itself '
        'transfers cross-bench at %.3f; acc-only residuals carry theta^A.' % sg,
        'Once the joint OWN theta^A is removed, HS residual -> ~0 variance, so the cross-bench test degenerates '
        '(corr against noise) and CANNOT confirm independence; meanCONF residual sign is opposite across benches '
        '(ARC %.2f vs HS %.2f-on-noise) -> not a consistent trait.' % (
            res['arc']['resid_corr_meanCONF'], res['hs']['resid_corr_meanCONF']),
        'Automated m10/m10b verdicts (INDEPENDENT/SURVIVES) are OVERRIDDEN by this master review for the HS '
        'residual-degeneracy reason.',
    ],
}

OUT = os.path.join(HERE, 'results', 'results_m10_master_verdict.json')
json.dump(verdict, open(OUT, 'w'), indent=2)
print('WROTE', OUT)
print('DECISION:', verdict['DECISION'])
print('HS  R2(theta_C|thetaA,meanY)=%.4f resid_share=%.4f (var %.6f)' % (
    res['hs']['R2_thetaC_on_(thetaA,meanY)'], res['hs']['resid_share_after_thetaA_meanY'], res['hs']['resid_var_absolute']))
print('ARC R2=%.4f resid_share=%.4f resid~meanCONF=%.2f (->%.2f after gauge+len)' % (
    res['arc']['R2_thetaC_on_(thetaA,meanY)'], res['arc']['resid_share_after_thetaA_meanY'],
    res['arc']['resid_corr_meanCONF'], res['arc']['resid_corr_meanCONF_after_+gauge+len']))
print('smoking gun (thetaA-resid-on-acc cross-bench)=%.3f  yardstick thetaA cross-bench=%.3f' % (
    sg, verdict['yardstick_thetaA_crossbench']))
