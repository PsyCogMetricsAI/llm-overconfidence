#!/usr/bin/env python3
"""M10b — MASTER CRITICAL RE-REVIEW of the +0.61 cross-bench θ^C-residual (assume it is an ARTIFACT).

The killer confound m10 did NOT control: it residualized θ^C on raw `acc`, but raw acc is only a 0.88
proxy for the JOINT's OWN ability latent θ^A. And θ^A transfers cross-bench at 0.96 (ability is a real
trait). So ANY leftover θ^A in the residual is amplified into "cross-bench reproducibility" that has
nothing to do with overconfidence. θ^C~θ^A = -0.56 (ARC) / -0.92 (HS) -> lots of θ^A to leak.

Decisive ladder: residualize θ^C on progressively richer bases that INCLUDE θ^A itself.
  - if cross-bench resid corr SURVIVES removing {θ^A, meanY, gauge, length}  -> genuine independent axis
  - if it COLLAPSES toward 0 once θ^A is removed                              -> it was ability leakage (C3 holds)
Yardstick stays θ^A cross-bench (0.96). NULL = label shuffle.
Usage: m10b_decisive.py
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


def resid(y, cols):
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(c, float) for c in cols] + [np.ones(len(y))])
    v = ~np.isnan(y) & ~np.isnan(X).any(1)
    beta, *_ = np.linalg.lstsq(X[v], y[v], rcond=None)
    r = np.full(len(y), np.nan)
    r[v] = y[v] - X[v] @ beta
    return r


def load(bench):
    d = np.load(os.path.join(HERE, 'aligned', f'joint_vectors_{bench}.npz'), allow_pickle=True)
    v = {k: d[k] for k in d.files}
    v['models'] = v['models'].astype(str)
    if P(v['theta_A'], v['acc']) < 0:
        v['theta_A'] = -v['theta_A']
    if P(v['theta_C'], v['signed']) < 0:
        v['theta_C'] = -v['theta_C']
    return v


def Z(v, k):
    return zsc(v[k])


# residualization ladders (each maps a bench-dict -> list of covariate columns)
LADDERS = {
    '1_acc,meanY (m10 orig)': lambda v: [Z(v, 'acc'), Z(v, 'meanY')],
    '2_acc,meanY+quad+inter': lambda v: [Z(v, 'acc'), Z(v, 'meanY'), Z(v, 'acc') ** 2, Z(v, 'meanY') ** 2, Z(v, 'acc') * Z(v, 'meanY')],
    '3_thetaA,meanY': lambda v: [Z(v, 'theta_A'), Z(v, 'meanY')],
    '4_thetaA,meanY,gauge': lambda v: [Z(v, 'theta_A'), Z(v, 'meanY'), Z(v, 'gauge')],
    '5_thetaA,meanY,gauge,len,len2': lambda v: [Z(v, 'theta_A'), Z(v, 'meanY'), Z(v, 'gauge'), Z(v, 'tok_len'), Z(v, 'tok_len') ** 2],
    '6_thetaA,meanY,gauge,len+quad+inter': lambda v: [
        Z(v, 'theta_A'), Z(v, 'meanY'), Z(v, 'gauge'), Z(v, 'tok_len'),
        Z(v, 'theta_A') ** 2, Z(v, 'meanY') ** 2, Z(v, 'tok_len') ** 2,
        Z(v, 'theta_A') * Z(v, 'meanY'), Z(v, 'theta_A') * Z(v, 'tok_len')],
}


def main():
    arc, hs = load('arc'), load('hellaswag')
    mH = {m: i for i, m in enumerate(hs['models'])}
    shared = [m for m in arc['models'] if m in mH]
    mA = {m: i for i, m in enumerate(arc['models'])}
    iA = np.array([mA[m] for m in shared]); iH = np.array([mH[m] for m in shared])
    rng = np.random.RandomState(0)

    out = {'n_shared': len(shared),
           'yardstick_thetaA_crossbench': P(arc['theta_A'][iA], hs['theta_A'][iH]),
           'sanity_acc_crossbench': P(arc['acc'][iA], hs['acc'][iH]),
           'raw_thetaC_crossbench': P(arc['theta_C'][iA], hs['theta_C'][iH]),
           # the smoking-gun probe: does θ^A-leftover-after-acc itself transfer cross-bench?
           'thetaA_resid_on_acc_crossbench': P(resid(arc['theta_A'], [Z(arc, 'acc')])[iA],
                                               resid(hs['theta_A'], [Z(hs, 'acc')])[iH]),
           'thetaC_corr_thetaA': {'arc': P(arc['theta_C'], arc['theta_A']), 'hs': P(hs['theta_C'], hs['theta_A'])},
           'ladder': {}}

    for name, fn in LADDERS.items():
        rA = resid(arc['theta_C'], fn(arc))
        rH = resid(hs['theta_C'], fn(hs))
        cb = P(rA[iA], rH[iH])
        # NULL via label shuffle on this rung
        nulls = np.array([P(rA[iA], rH[iH][rng.permutation(len(iH))]) for _ in range(200)])
        out['ladder'][name] = {
            'crossbench_resid_corr': cb,
            'resid_share_arc': float(np.nanvar(rA) / (np.nanvar(arc['theta_C']) + 1e-12)),
            'resid_share_hs': float(np.nanvar(rH) / (np.nanvar(hs['theta_C']) + 1e-12)),
            'resid_corr_thetaA_arc': P(rA, arc['theta_A']),    # leakage check: should be ~0 once θ^A removed
            'null_p95_abs': float(np.nanpercentile(np.abs(nulls), 95)),
        }

    # verdict: compare rung1 (acc only) vs rung3 (θ^A) vs rung6 (everything)
    r1 = out['ladder']['1_acc,meanY (m10 orig)']['crossbench_resid_corr']
    r3 = out['ladder']['3_thetaA,meanY']['crossbench_resid_corr']
    r6 = out['ladder']['6_thetaA,meanY,gauge,len+quad+inter']['crossbench_resid_corr']
    survives = abs(r6) >= 0.4
    collapsed = abs(r6) < 0.15
    out['VERDICT'] = {
        'r_acc_only': r1, 'r_thetaA': r3, 'r_full': r6,
        'thetaA_resid_on_acc_crossbench': out['thetaA_resid_on_acc_crossbench'],
        'decision': ('INDEPENDENT_AXIS_SURVIVES_thetaA+gauge+len' if survives
                     else 'WAS_ABILITY/GAUGE/LEN_LEAKAGE_C3_HOLDS' if collapsed
                     else 'WEAKENED_INCONCLUSIVE'),
    }

    OUT = os.path.join(HERE, 'results', 'results_m10b_decisive.json')
    json.dump(out, open(OUT, 'w'), indent=2)
    print('WROTE', OUT)
    print(f"n_shared={out['n_shared']}  yardstick θ^A cross-bench={out['yardstick_thetaA_crossbench']:+.3f}")
    print(f"SMOKING GUN — θ^A-residual-on-acc cross-bench = {out['thetaA_resid_on_acc_crossbench']:+.3f}  "
          f"(if high, acc-only residual leaks θ^A)")
    print(f"θ^C~θ^A: ARC={out['thetaC_corr_thetaA']['arc']:+.3f} HS={out['thetaC_corr_thetaA']['hs']:+.3f}")
    print('--- cross-bench θ^C-residual ladder (null p95 ~0.04) ---')
    for name, r in out['ladder'].items():
        print(f"  {name:42s} cross-bench={r['crossbench_resid_corr']:+.3f}  "
              f"resid_share(arc/hs)={r['resid_share_arc']:.2f}/{r['resid_share_hs']:.2f}  "
              f"resid~θ^A(arc)={r['resid_corr_thetaA_arc']:+.2f}")
    print(f"VERDICT: {out['VERDICT']['decision']}  (r: acc-only={r1:+.3f} -> +θ^A={r3:+.3f} -> full={r6:+.3f})")


if __name__ == '__main__':
    main()
