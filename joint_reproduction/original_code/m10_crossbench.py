#!/usr/bin/env python3
"""M10 — DECISIVE cross-benchmark test for the θ^C independence claim (master critical re-review).

The M11 attack agent found a minority reproducible θ^C residual in ARC that loads on chosen-confidence
(meanCONF, a channel OUTSIDE the Beta fit). Disproof test (assume the claim is WRONG):

  A REAL independent overconfidence latent is a per-model TRAIT -> it must reproduce ACROSS benchmarks
  on the SAME models (ARC∩HS). A length/leak/per-fit artifact will NOT reproduce cross-bench.

Calibration yardstick: θ^A (ability) IS a known real cross-bench latent. Its cross-bench corr defines
"what a real latent looks like" on these shared models. Decision:
  - |corr(θ^C_resid_ARC, θ^C_resid_HS)| >= 0.4 and comparable to θ^A's cross-bench corr  -> INDEPENDENT axis
  - ~0  while θ^A cross-bench corr is high                                               -> bench-specific artifact (C3 holds)

Re-review item (1): NONLINEAR length control. θ^C~answer-length=0.71, so also residualize on
{len, len^2, A·len, meanY·len} and re-test -- a length artifact dies, a real latent survives.

Signs: θ^C / θ^A are each identified up to a sign per independent fit -> align per bench BEFORE
comparing (θ^A so corr(θ^A,acc)>0; θ^C so corr(θ^C, signed=C-A)>0, the dominant stable loading).
Usage: m10_crossbench.py
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
    """Residual of y on [cols..., 1] (OLS), NaN-safe."""
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(c, float) for c in cols] + [np.ones(len(y))])
    v = ~np.isnan(y) & ~np.isnan(X).any(1)
    beta, *_ = np.linalg.lstsq(X[v], y[v], rcond=None)
    r = np.full(len(y), np.nan)
    r[v] = y[v] - X[v] @ beta
    return r


def basis(v, with_len):
    A, Y = zsc(v['acc']), zsc(v['meanY'])
    cols = [A, Y, A * A, Y * Y, A * Y]                      # rich (A,meanY): linear+quad+interaction
    if with_len:
        L = zsc(v['tok_len'])
        cols += [L, L * L, A * L, Y * L]                   # + nonlinear length control
    return cols


def load(bench):
    d = np.load(os.path.join(HERE, 'aligned', f'joint_vectors_{bench}.npz'), allow_pickle=True)
    v = {k: d[k] for k in d.files}
    v['models'] = v['models'].astype(str)
    # sign-align (each fit identifies θ up to sign): θ^A so corr(.,acc)>0; θ^C so corr(.,signed)>0
    if P(v['theta_A'], v['acc']) < 0:
        v['theta_A'] = -v['theta_A']
    if P(v['theta_C'], v['signed']) < 0:
        v['theta_C'] = -v['theta_C']
    return v


def main():
    arc, hs = load('arc'), load('hellaswag')
    mH = {m: i for i, m in enumerate(hs['models'])}
    shared = [m for m in arc['models'] if m in mH]
    mA = {m: i for i, m in enumerate(arc['models'])}
    iA = np.array([mA[m] for m in shared]); iH = np.array([mH[m] for m in shared])

    out = {'n_arc': len(arc['models']), 'n_hs': len(hs['models']), 'n_shared': len(shared),
           'note': 'θ^C residualized on rich (A,meanY); decisive = cross-bench corr vs θ^A calibration'}

    # CALIBRATION: θ^A (real latent) cross-bench corr on shared models + raw θ^C cross-bench
    out['calibration_thetaA_crossbench'] = P(arc['theta_A'][iA], hs['theta_A'][iH])
    out['raw_thetaC_crossbench'] = P(arc['theta_C'][iA], hs['theta_C'][iH])
    out['sanity_acc_crossbench'] = P(arc['acc'][iA], hs['acc'][iH])         # accuracy itself cross-bench
    out['sanity_meanCONF_crossbench'] = P(arc['meanCONF'][iA], hs['meanCONF'][iH])

    rng = np.random.RandomState(0)
    for name, wl in [('rich_AY', False), ('rich_AY+nonlin_len', True)]:
        rA = resid(arc['theta_C'], basis(arc, wl))
        rH = resid(hs['theta_C'], basis(hs, wl))
        rcorr = P(rA[iA], rH[iH])
        # meanCONF partial within each bench on this basis (the attack's out-of-channel signal)
        cA = resid(arc['meanCONF'], basis(arc, wl))
        cH = resid(hs['meanCONF'], basis(hs, wl))
        pc_arc = P(rA, cA); pc_hs = P(rH, cH)
        # NULL: shuffle HS residual labels -> cross-bench corr should collapse to ~0
        nulls = []
        for s in range(200):
            perm = rng.permutation(len(iH))
            nulls.append(P(rA[iA], rH[iH][perm]))
        nulls = np.array(nulls)
        out[name] = {
            'crossbench_thetaC_resid_corr': rcorr,
            'null_shuffle_mean': float(np.nanmean(nulls)),
            'null_shuffle_p95_abs': float(np.nanpercentile(np.abs(nulls), 95)),
            'resid_share_arc': float(np.nanvar(rA) / (np.nanvar(arc['theta_C']) + 1e-12)),
            'resid_share_hs': float(np.nanvar(rH) / (np.nanvar(hs['theta_C']) + 1e-12)),
            'meanCONF_partial_arc': pc_arc,
            'meanCONF_partial_hs': pc_hs,
            'meanCONF_partial_same_sign': bool(np.sign(pc_arc) == np.sign(pc_hs) and not np.isnan(pc_arc * pc_hs)),
        }

    # ---- decisive verdict ----
    rc = out['rich_AY']['crossbench_thetaC_resid_corr']
    rc_len = out['rich_AY+nonlin_len']['crossbench_thetaC_resid_corr']
    cal = out['calibration_thetaA_crossbench']
    p95 = out['rich_AY']['null_shuffle_p95_abs']
    real = (abs(rc) >= 0.4 and abs(rc) > p95 and abs(rc_len) >= 0.3)
    bench_specific = (abs(rc) < 0.15 and abs(cal) > 0.4)
    out['VERDICT'] = {
        'thetaC_resid_crossbench': rc,
        'thetaC_resid_crossbench_lenctrl': rc_len,
        'thetaA_crossbench_yardstick': cal,
        'null_p95': p95,
        'decision': ('INDEPENDENT_AXIS_CROSSBENCH_CONFIRMED' if real
                     else 'BENCH_SPECIFIC_ARTIFACT_C3_HOLDS' if bench_specific
                     else 'INCONCLUSIVE_WEAK'),
        'reasoning': ('θ^C residual reproduces cross-bench at the level of a real latent (vs θ^A yardstick) and '
                      'survives nonlinear length control' if real else
                      'θ^C residual does NOT reproduce cross-bench though ability (θ^A) does -> the ARC residual '
                      'is bench-specific (length/leak/fit), NOT a transferable overconfidence trait; C3 stands' if bench_specific else
                      'cross-bench signal is weak/mixed -> cannot claim an independent axis'),
    }

    OUT = os.path.join(HERE, 'results', 'results_m10_crossbench.json')
    json.dump(out, open(OUT, 'w'), indent=2)
    print('WROTE', OUT)
    print(f"n_shared={out['n_shared']}  |  θ^A cross-bench (yardstick) = {cal:+.3f}  acc cross-bench={out['sanity_acc_crossbench']:+.3f}")
    print(f"raw θ^C cross-bench = {out['raw_thetaC_crossbench']:+.3f}")
    print(f"★ θ^C RESID cross-bench (rich A,meanY)        = {rc:+.3f}   (null p95 |r|={p95:.3f})")
    print(f"★ θ^C RESID cross-bench (+ nonlinear length)  = {rc_len:+.3f}")
    print(f"  meanCONF partial: ARC={out['rich_AY']['meanCONF_partial_arc']:+.3f} HS={out['rich_AY']['meanCONF_partial_hs']:+.3f} "
          f"same_sign={out['rich_AY']['meanCONF_partial_same_sign']}")
    print(f"  meanCONF partial (+len): ARC={out['rich_AY+nonlin_len']['meanCONF_partial_arc']:+.3f} "
          f"HS={out['rich_AY+nonlin_len']['meanCONF_partial_hs']:+.3f}")
    print(f"VERDICT: {out['VERDICT']['decision']}")
    print(f"  {out['VERDICT']['reasoning']}")


if __name__ == '__main__':
    main()
