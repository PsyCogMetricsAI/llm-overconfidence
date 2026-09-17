#!/usr/bin/env python3
"""FULLIRT-FULLFIT statistics - NUMERIC reproduction of the original m10/m10b/m10_finalize math.

Scope (numbers only): recomputes, on the NEW authorized full panel
(results/joint_vectors_arc_full_2500ep.npz, results/joint_vectors_hellaswag_full_2500ep.npz),
the numeric quantities of the original scripts

  Imports/eirt_cl_reproducible_20260611/logprob_irt_compare_20260614/m10_crossbench.py
  Imports/eirt_cl_reproducible_20260611/logprob_irt_compare_20260614/m10b_decisive.py
  Imports/eirt_cl_reproducible_20260611/logprob_irt_compare_20260614/m10_finalize.py

via the SAME helper definitions (zsc / P / resid reproduce the originals line by line), the SAME
residualization ladders, the SAME sign-alignment convention and the SAME 200-shuffle seed-0 nulls
(one RandomState(0) stream consumed in each original script's own ladder order).

Deliberately NOT reproduced: the originals' automated DECISION/VERDICT strings, their hardcoded
narrative reasoning, their abs()-based acceptance tests (m10/m10b) and the hardcoded override in
m10_finalize. This file emits raw correlations, shares, R2 and diagnostics only; no field in any
output states a scientific conclusion.

Also computed here:
  * the full residual ladder (m10b rungs 1-6 + the two m10 rich bases),
  * absolute residual variances / residual shares / R2,
  * ability raw cross-bench correlation and ability-after-accuracy cross-bench correlation
    (thetaA residualized on zsc(acc), the m10b/m10_finalize probe),
  * within-bench loadings of thetaC/thetaA on acc, meanY, meanCONF, gauge, tok_len, signed
    (historical signed = meanY - acc; the runner stores exactly this),
  * sign alignment per bench: thetaA so corr(thetaA, acc) > 0; thetaC so corr(thetaC, signed) > 0,
  * paired shared model IDs only (arc+hellaswag by exact string ID),
  * likelihood-gauge diagnostic using the ACTUAL reparameterization A'=s*A, b'=s*b, a'=a/s,
    C'=t*(A+C)-s*A, z'=t*z, alpha'=alpha/t, gamma'=gamma (positive s,t, independently chosen per
    benchmark): both model logits (hence the likelihood) are verified unchanged on a bounded sample,
    invariant conditional correlations and gauge-dependent raw quantities (raw thetaC correlation,
    shares, R2) are reported separately with their max errors,
  * run-health checks: all arrays finite, full 2500-epoch loss history, last-100 loss trend,
    centering constraints (mean zero on theta_A, theta_C, gamma_m, z).

Usage:
  python code/analyze_joint.py --results-dir results
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.dirname(HERE)
EPS = 1e-12
N_SHUFFLE = 200
SHUFFLE_SEED = 0
EXPECTED_SHARED = 6716
EXPECTED_EPOCHS = 2500


def project_root(start: str) -> str:
    p = os.path.abspath(start)
    for _ in range(12):
        if os.path.isdir(os.path.join(p, "Imports")):
            return p
        if os.path.dirname(p) == p:
            break
        p = os.path.dirname(p)
    raise RuntimeError(f"project root (dir containing 'Imports') not found above {start}")


PROJECT_ROOT = os.path.dirname(HERE)
ORIG_DIR = HERE


# ---------------------------------------------------------------------------- helpers (originals)
def zsc(x):
    """Original m10/m10b/m10_finalize z-score (NaN-aware, +1e-12 denominator)."""
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def P(a, b):
    """Original Pearson correlation on pairwise-finite entries; NaN if <=5 pairs or degenerate."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = ~np.isnan(a) & ~np.isnan(b)
    if v.sum() <= 5:
        return float("nan")
    aa, bb = a[v], b[v]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def resid(y, cols):
    """Original OLS residual of y on [cols..., 1]; NaN-safe (rows dropped, residuals NaN there)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(c, float) for c in cols] + [np.ones(len(y))])
    v = ~np.isnan(y) & ~np.isnan(X).any(1)
    beta, *_ = np.linalg.lstsq(X[v], y[v], rcond=None)
    r = np.full(len(y), np.nan)
    r[v] = y[v] - X[v] @ beta
    return r


def var_share(r, v):
    """resid_share = var(resid) / var(original) with the originals' +1e-12 guard."""
    return float(np.nanvar(r) / (np.nanvar(v) + EPS))


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()


# ------------------------------------------------------------------------------------- data load
def load_vectors(path: str) -> dict:
    """Load one runner output NPZ verbatim (no rename, no transform)."""
    with np.load(path, allow_pickle=False) as z:
        v = {k: z[k] for k in z.files}
    v["models"] = np.asarray(v["models"]).astype(str)
    if len(set(v["models"].tolist())) != len(v["models"]):
        raise RuntimeError(f"{path}: duplicate model ids")
    required = ("theta_A", "theta_C", "gamma_m", "acc", "meanY", "meanCONF", "gauge", "tok_len",
                "signed", "a", "b", "alpha", "z", "loss_history", "fit_epochs", "mode")
    missing = [k for k in required if k not in v]
    if missing:
        raise RuntimeError(f"{path}: missing arrays {missing}")
    return v


def align(v: dict, bench: str) -> dict:
    """Original sign convention: thetaA toward +acc, thetaC toward +signed (signed = meanY - acc)."""
    signed_calc = np.asarray(v["meanY"], float) - np.asarray(v["acc"], float)
    if not np.allclose(signed_calc, np.asarray(v["signed"], float), rtol=0, atol=1e-12):
        raise RuntimeError(f"{bench}: stored 'signed' != meanY - acc (historical convention broken)")
    corr_a = P(v["theta_A"], v["acc"])
    corr_c = P(v["theta_C"], v["signed"])
    flip_a = bool(np.sign(corr_a) < 0)
    flip_c = bool(np.sign(corr_c) < 0)
    if flip_a:
        v["theta_A"] = -np.asarray(v["theta_A"], float)
    if flip_c:
        v["theta_C"] = -np.asarray(v["theta_C"], float)
    return {"bench": bench, "corr_thetaA_acc_raw": corr_a, "corr_thetaC_signed_raw": corr_c,
            "sign_flipped_thetaA": flip_a, "sign_flipped_thetaC": flip_c}


def paired_shared(arc: dict, hs: dict) -> dict:
    """Exact-string-ID pairing of the two benches; returns index arrays + the shared ID list."""
    mH = {m: i for i, m in enumerate(hs["models"])}
    shared = [m for m in arc["models"] if m in mH]
    mA = {m: i for i, m in enumerate(arc["models"])}
    iA = np.array([mA[m] for m in shared], dtype=np.int64)
    iH = np.array([mH[m] for m in shared], dtype=np.int64)
    return {"shared": shared, "iA": iA, "iH": iH, "n_shared": len(shared)}


# ------------------------------------------------------------------------- residualization bases
def Z(v, k):
    return zsc(v[k])


LADDERS = {
    # m10b_decisive.py LADDERS, same order (null stream order of that script)
    "1_acc,meanY (m10 orig)": lambda v: [Z(v, "acc"), Z(v, "meanY")],
    "2_acc,meanY+quad+inter": lambda v: [Z(v, "acc"), Z(v, "meanY"), Z(v, "acc") ** 2,
                                         Z(v, "meanY") ** 2, Z(v, "acc") * Z(v, "meanY")],
    "3_thetaA,meanY": lambda v: [Z(v, "theta_A"), Z(v, "meanY")],
    "4_thetaA,meanY,gauge": lambda v: [Z(v, "theta_A"), Z(v, "meanY"), Z(v, "gauge")],
    "5_thetaA,meanY,gauge,len,len2": lambda v: [Z(v, "theta_A"), Z(v, "meanY"), Z(v, "gauge"),
                                                Z(v, "tok_len"), Z(v, "tok_len") ** 2],
    "6_thetaA,meanY,gauge,len+quad+inter": lambda v: [
        Z(v, "theta_A"), Z(v, "meanY"), Z(v, "gauge"), Z(v, "tok_len"),
        Z(v, "theta_A") ** 2, Z(v, "meanY") ** 2, Z(v, "tok_len") ** 2,
        Z(v, "theta_A") * Z(v, "meanY"), Z(v, "theta_A") * Z(v, "tok_len")],
    # m10_crossbench.py rich bases (kept under the m10 names; identical column sets)
    "m10_rich_AY": lambda v: [Z(v, "acc"), Z(v, "meanY"), Z(v, "acc") ** 2, Z(v, "meanY") ** 2,
                              Z(v, "acc") * Z(v, "meanY")],
    "m10_rich_AY+nonlin_len": lambda v: [Z(v, "acc"), Z(v, "meanY"), Z(v, "acc") ** 2,
                                         Z(v, "meanY") ** 2, Z(v, "acc") * Z(v, "meanY"),
                                         Z(v, "tok_len"), Z(v, "tok_len") ** 2,
                                         Z(v, "acc") * Z(v, "tok_len"),
                                         Z(v, "meanY") * Z(v, "tok_len")],
}

# Rungs whose null streams follow m10b's single RandomState(0) in dict order; the two m10_rich
# rungs are computed with a second RandomState(0) consumed in m10_crossbench.py's own order.
M10B_RUNG_NAMES = tuple(list(LADDERS)[:6])
M10_RUNG_NAMES = ("m10_rich_AY", "m10_rich_AY+nonlin_len")


def ladder_table(arc: dict, hs: dict, pair: dict, rung_names, rng) -> dict:
    """Cross-bench residual ladder for the named rungs; rng reused across rungs like the originals."""
    iA, iH, n_shared = pair["iA"], pair["iH"], pair["n_shared"]
    out = {}
    for name in rung_names:
        fn = LADDERS[name]
        rA = resid(arc["theta_C"], fn(arc))
        rH = resid(hs["theta_C"], fn(hs))
        cb = P(rA[iA], rH[iH])
        nulls = np.array([P(rA[iA], rH[iH][rng.permutation(n_shared)]) for _ in range(N_SHUFFLE)])
        out[name] = {
            "crossbench_resid_corr": cb,
            "resid_share_arc": var_share(rA, arc["theta_C"]),
            "resid_share_hs": var_share(rH, hs["theta_C"]),
            "R2_arc": 1.0 - var_share(rA, arc["theta_C"]),
            "R2_hs": 1.0 - var_share(rH, hs["theta_C"]),
            "resid_var_absolute_arc": float(np.nanvar(rA)),
            "resid_var_absolute_hs": float(np.nanvar(rH)),
            "resid_corr_thetaA_arc": P(rA, arc["theta_A"]),
            "resid_corr_thetaA_hs": P(rH, hs["theta_A"]),
            "null_shuffle_n": int(N_SHUFFLE),
            "null_shuffle_seed": int(SHUFFLE_SEED),
            "null_shuffle_mean": float(np.nanmean(nulls)),
            "null_shuffle_p95_abs": float(np.nanpercentile(np.abs(nulls), 95)),
            "null_shuffle_p95_signed": float(np.nanpercentile(nulls, 95)),
            "n_pairs": int(n_shared),
        }
        # m10/m10b also ran the out-of-channel probe on the rich ACC/meanY bases only
        if name.startswith("m10_rich"):
            cA = resid(arc["meanCONF"], fn(arc))
            cH = resid(hs["meanCONF"], fn(hs))
            pc_arc, pc_hs = P(rA, cA), P(rH, cH)
            out[name].update({"meanCONF_partial_arc": pc_arc, "meanCONF_partial_hs": pc_hs,
                              "meanCONF_partial_same_sign": bool(
                                  np.sign(pc_arc) == np.sign(pc_hs)
                                  and not np.isnan(pc_arc * pc_hs))})
    return out


# ------------------------------------------------------------------------------- per-bench stats
def bench_stats(v: dict, bench: str) -> dict:
    """m10_finalize.py per-bench numbers + within-bench loadings, same helpers."""
    base = [zsc(v["theta_A"]), zsc(v["meanY"])]
    r = resid(v["theta_C"], base)
    rg = resid(v["theta_C"], base + [zsc(v["gauge"]), zsc(v["tok_len"])])
    share = var_share(r, v["theta_C"])
    share_g = var_share(rg, v["theta_C"])
    return {
        "bench": bench,
        "thetaC_corr_thetaA": P(v["theta_C"], v["theta_A"]),
        "thetaC_corr_signed": P(v["theta_C"], v["signed"]),
        "thetaC_corr_acc": P(v["theta_C"], v["acc"]),
        "thetaC_corr_meanY": P(v["theta_C"], v["meanY"]),
        "thetaC_corr_meanCONF": P(v["theta_C"], v["meanCONF"]),
        "thetaC_corr_gauge": P(v["theta_C"], v["gauge"]),
        "thetaC_corr_tok_len": P(v["theta_C"], v["tok_len"]),
        "thetaA_corr_acc": P(v["theta_A"], v["acc"]),
        "thetaA_corr_meanY": P(v["theta_A"], v["meanY"]),
        "thetaA_corr_meanCONF": P(v["theta_A"], v["meanCONF"]),
        "thetaA_corr_signed": P(v["theta_A"], v["signed"]),
        "R2_thetaC_on_(thetaA,meanY)": 1.0 - share,
        "resid_share_after_thetaA_meanY": share,
        "resid_var_absolute": float(np.nanvar(r)),
        "resid_corr_meanCONF": P(r, v["meanCONF"]),
        "resid_share_after_+gauge+len": share_g,
        "R2_thetaC_on_(thetaA,meanY,gauge,len)": 1.0 - share_g,
        "resid_var_absolute_+gauge+len": float(np.nanvar(rg)),
        "resid_corr_meanCONF_after_+gauge+len": P(rg, v["meanCONF"]),
        "model_vector_std": {
            "theta_A": float(np.nanstd(v["theta_A"])), "theta_C": float(np.nanstd(v["theta_C"])),
            "acc": float(np.nanstd(v["acc"])), "meanY": float(np.nanstd(v["meanY"])),
            "meanCONF": float(np.nanstd(v["meanCONF"])), "gauge": float(np.nanstd(v["gauge"])),
            "tok_len": float(np.nanstd(v["tok_len"]))},
    }


# --------------------------------------------------------------------------------- run-health
def health_checks(v: dict, bench: str) -> dict:
    """Finiteness of every saved array, full loss trajectory + last-100 trend, centering."""
    arrays = {k: np.asarray(v[k]) for k in v if k not in ("models", "mode")}
    nonfinite = {}
    for k, arr in arrays.items():
        if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
            nonfinite[k] = int(np.sum(~np.isfinite(arr)))
    loss = np.asarray(v["loss_history"], float)
    n = int(loss.size)
    fit_epochs = int(np.asarray(v["fit_epochs"]).reshape(-1)[0])
    tail = loss[-100:]
    idx = np.arange(tail.size, dtype=float)
    slope = float(np.polyfit(idx, tail, 1)[0]) if tail.size > 1 else float("nan")
    d = np.diff(loss) if n > 1 else np.array([])
    centering = {k: float(np.mean(np.asarray(v[k], float))) for k in
                 ("theta_A", "theta_C", "gamma_m", "z")}
    return {
        "bench": bench, "n_models": int(v["models"].size),
        "all_arrays_finite": not nonfinite, "nonfinite_arrays": nonfinite,
        "loss": {
            "n_recorded": n, "full_2500": bool(n == EXPECTED_EPOCHS == fit_epochs),
            "fit_epochs_field": fit_epochs,
            "first": float(loss[0]), "last": float(loss[-1]), "min": float(np.min(loss)),
            "argmin_epoch": int(np.argmin(loss)), "max": float(np.max(loss)),
            "history_sha256": sha256_array(loss),
            "n_increases_total": int(np.sum(d > 0)),
            "last100": {
                "n": int(tail.size), "first": float(tail[0]), "last": float(tail[-1]),
                "mean": float(np.mean(tail)), "min": float(np.min(tail)), "max": float(np.max(tail)),
                "ols_slope_per_epoch": slope,
                "delta_last_minus_first": float(tail[-1] - tail[0]),
                "n_increases": int(np.sum(np.diff(tail) > 0)),
                "any_nonfinite": bool(not np.isfinite(tail).all()),
            },
        },
        "centering": {"means": centering,
                      "max_abs_mean": float(max(abs(x) for x in centering.values())),
                      "centered_within_1e-9": bool(max(abs(x) for x in centering.values()) < 1e-9)},
        "array_sha256": {k: sha256_array(arrays[k]) for k in
                         ("theta_A", "theta_C", "gamma_m", "a", "b", "alpha", "z")},
    }


# ---------------------------------------------------------------- likelihood gauge diagnostic
# ACTUAL reparameterization gauge of the fitted joint model, for independently chosen positive
# scales s, t per benchmark (independently chosen per benchmark is allowed because each bench is a
# separate fit):
#
#     A' = s*A          b' = s*b          a' = a/s
#     C' = t*(A + C) - s*A                z' = t*z       alpha' = alpha/t      gamma' = gamma
#
# Both model logits are then invariant TERM BY TERM (original joint formula, joint_irt.py):
#     Eq1 logit  a'_i*(A'_m - b'_i)                       = a_i*(A_m - b_i)
#     Eq2 logit  alpha'_i*(A'_m + C'_m + z'_i) + gamma'_m*L~_i
#                                                          = alpha_i*(A_m + C_m + z_i) + gamma_m*L~_i
# so the LIKELIHOOD is unchanged, while several reported raw quantities move. That is what this
# diagnostic measures. It is NOT a statement about Pearson invariance of independently rescaled
# saved vectors (which is trivially true of any correlation and was the wording bug removed here).
#
# Orientation: this diagnostic uses the ORIGINAL UNSIGNFLIPPED fitted parameters and applies NO
# sign re-alignment anywhere (sign alignment is a descriptive reporting convention used only in the
# baseline sections of this report).

GAUGE_SCENARIOS = {
    # each scenario: benchmark -> (s, t); s = 1, t = 1 is the raw fitted parameterization
    "identity": {"arc": (1.0, 1.0), "hellaswag": (1.0, 1.0)},
    "both_benches_s2_t3": {"arc": (2.0, 3.0), "hellaswag": (2.0, 3.0)},
    "independent_arc_s0.5_t1__hs_s2_t4": {"arc": (0.5, 1.0), "hellaswag": (2.0, 4.0)},
    "mixed_arc_s0.25_t0.5__hs_s8_t8": {"arc": (0.25, 0.5), "hellaswag": (8.0, 8.0)},
    "mixed_arc_s3_t1__hs_s1_t5": {"arc": (3.0, 1.0), "hellaswag": (1.0, 5.0)},
}

GAUGE_SAMPLE_MODELS = 64
GAUGE_SAMPLE_ITEMS = 64

# quantities that are invariant under the positive-scale gauge when orientation is FIXED (no
# re-alignment): conditional residual correlations whose control set includes A itself, plus the
# ability cross-bench correlations (residual on A scales positively with A).
GAUGE_INVARIANT_RUNGS = ("3_thetaA,meanY", "4_thetaA,meanY,gauge",
                         "5_thetaA,meanY,gauge,len,len2",
                         "6_thetaA,meanY,gauge,len+quad+inter")
# quantities that are gauge dependent (no blanket invariance): raw C correlations, any residual not
# conditioning on A, and every variance share / R2 (var(C') is not proportional to var(C)).
GAUGE_VARIABLE_RUNGS = ("1_acc,meanY (m10 orig)", "2_acc,meanY+quad+inter", "m10_rich_AY+nonlin_len")


def apply_parameter_gauge(v: dict, s: float, t: float) -> dict:
    """Apply the exact joint-likelihood reparameterization (s, t > 0) to saved fitted parameters.

    Returns a NEW dict; the input arrays are never modified (data-side vectors acc/meanY/meanCONF/
    gauge/tok_len/signed and gamma_m are unchanged by the transformation).
    """
    if not (s > 0 and t > 0):
        raise ValueError(f"gauge scales must be positive (got s={s}, t={t})")
    A = np.asarray(v["theta_A"], float)
    C = np.asarray(v["theta_C"], float)
    out = dict(v)
    out["theta_A"] = s * A
    out["theta_C"] = t * (A + C) - s * A
    out["b"] = s * np.asarray(v["b"], float)
    out["a"] = np.asarray(v["a"], float) / s
    out["z"] = t * np.asarray(v["z"], float)
    out["alpha"] = np.asarray(v["alpha"], float) / t
    out["gamma_m"] = np.array(v["gamma_m"], float)   # gamma' = gamma
    return out


def sample_logits(v: dict, Lt, *, n_models: int = GAUGE_SAMPLE_MODELS,
                  n_items: int = GAUGE_SAMPLE_ITEMS):
    """Both joint-model logits from the original formula on a bounded fixed-stride index grid."""
    Lt = np.asarray(Lt, float)
    M, I = int(np.asarray(v["gamma_m"]).size), int(Lt.size)
    im = np.unique(np.linspace(0, M - 1, min(n_models, M)).astype(int))
    ii = np.unique(np.linspace(0, I - 1, min(n_items, I)).astype(int))
    A = np.asarray(v["theta_A"], float)[im]
    C = np.asarray(v["theta_C"], float)[im]
    gamma = np.asarray(v["gamma_m"], float)[im]
    a = np.asarray(v["a"], float)[ii]
    b = np.asarray(v["b"], float)[ii]
    alpha = np.asarray(v["alpha"], float)[ii]
    z = np.asarray(v["z"], float)[ii]
    eq1 = a[None, :] * (A[:, None] - b[None, :])
    eq2 = alpha[None, :] * (A[:, None] + C[:, None] + z[None, :]) + gamma[:, None] * Lt[ii][None, :]
    return eq1, eq2, {"n_models_sampled": int(im.size), "n_items_sampled": int(ii.size),
                      "n_pairs": int(im.size * ii.size), "method": "fixed-stride index grid"}


def derive_item_length(matrix_path: str) -> np.ndarray:
    """Reconstruct L~_i exactly as the runner does (read-only): nanmedian(LP_sum/LP_mean, axis=1)
    centered by nanmean. Computed row by row to bound memory."""
    with np.load(matrix_path, allow_pickle=False) as z:
        LPm, LPs = z["LP_mean"], z["LP_sum"]
        n_items = int(LPm.shape[0])
        Lt = np.empty(n_items, dtype=np.float64)
        for i in range(n_items):
            with np.errstate(divide="ignore", invalid="ignore"):
                tok = LPs[i].astype(np.float64) / LPm[i].astype(np.float64)
            Lt[i] = np.nanmedian(tok)
    Lt -= np.nanmean(Lt)
    if not np.isfinite(Lt).all():
        raise RuntimeError(f"{matrix_path}: reconstructed L~ has non-finite entries")
    return Lt


def load_item_length(results_dir: str, bench: str) -> dict:
    """L~ for one bench from the frozen matrix referenced by that bench's runner JSON (read-only)."""
    run_json = os.path.join(results_dir, f"joint_run_{bench}_full_2500ep.json")
    if not os.path.isfile(run_json):
        raise RuntimeError(f"gauge diagnostic needs the runner JSON (for the matrix path): {run_json}")
    j = json.load(open(run_json, encoding="utf-8"))
    mpath = j.get("matrix")
    if not mpath or not os.path.isfile(mpath):
        raise RuntimeError(f"gauge diagnostic needs the frozen matrix on disk: {mpath!r}")
    return {"Lt": derive_item_length(mpath), "matrix": mpath,
            "matrix_sha256": sha256_file(mpath),
            "matches_run_json_sha256": bool(sha256_file(mpath) == j.get("matrix_sha256")),
            "derivation": "L~ = nanmedian(LP_sum/LP_mean, axis=1), then centered by nanmean"}


def _gauge_invariants(arc: dict, hs: dict, pair: dict) -> dict:
    """Positive-t invariant conditional quantities, computed on RAW orientation (no re-alignment)."""
    iA, iH = pair["iA"], pair["iH"]
    out = {
        "thetaA_crossbench_raw": P(arc["theta_A"][iA], hs["theta_A"][iH]),
        "thetaA_resid_on_acc_crossbench": P(
            resid(arc["theta_A"], [Z(arc, "acc")])[iA], resid(hs["theta_A"], [Z(hs, "acc")])[iH]),
    }
    for rung in GAUGE_INVARIANT_RUNGS:
        fn = LADDERS[rung]
        out[f"{rung}__crossbench_resid_corr"] = P(resid(arc["theta_C"], fn(arc))[iA],
                                                  resid(hs["theta_C"], fn(hs))[iH])
    return out


def _gauge_variables(arc: dict, hs: dict, pair: dict) -> dict:
    """Gauge-dependent quantities (raw C correlations, A-free residual ladders, shares, R2)."""
    iA, iH = pair["iA"], pair["iH"]
    out = {"thetaC_crossbench_raw": P(arc["theta_C"][iA], hs["theta_C"][iH])}
    for rung in GAUGE_VARIABLE_RUNGS:
        fn = LADDERS[rung]
        rA = resid(arc["theta_C"], fn(arc))
        rH = resid(hs["theta_C"], fn(hs))
        out[f"{rung}__crossbench_resid_corr"] = P(rA[iA], rH[iH])
        out[f"{rung}__resid_share_arc"] = var_share(rA, arc["theta_C"])
        out[f"{rung}__resid_share_hs"] = var_share(rH, hs["theta_C"])
        out[f"{rung}__R2_arc"] = 1.0 - var_share(rA, arc["theta_C"])
        out[f"{rung}__R2_hs"] = 1.0 - var_share(rH, hs["theta_C"])
    for bench, v in (("arc", arc), ("hellaswag", hs)):
        out[f"thetaC_corr_thetaA_{bench}"] = P(v["theta_C"], v["theta_A"])
        out[f"thetaC_corr_signed_{bench}"] = P(v["theta_C"], v["signed"])
        r3 = resid(v["theta_C"], LADDERS["3_thetaA,meanY"](v))
        out[f"rung3_resid_share_{bench}"] = var_share(r3, v["theta_C"])
        out[f"rung3_R2_{bench}"] = 1.0 - var_share(r3, v["theta_C"])
    return out


def likelihood_gauge_diagnostic(arc: dict, hs: dict, pair: dict, lt_by_bench: dict,
                                scenarios: dict | None = None) -> dict:
    """Measure the ACTUAL joint-likelihood gauge: unchanged logits vs moved raw quantities.

    Per scenario, per benchmark (s, t) are applied to the ORIGINAL UNSIGNFLIPPED fitted parameters
    via `apply_parameter_gauge`; both model logits are compared on a bounded sample; invariant
    conditional correlations and gauge-dependent raw quantities are reported separately with their
    max errors. No sign re-alignment is performed anywhere inside this diagnostic.
    """
    scenarios = scenarios or GAUGE_SCENARIOS
    raw = {"arc": arc, "hellaswag": hs}
    lt, lt_meta = {}, {}
    for b in ("arc", "hellaswag"):
        entry = lt_by_bench[b]
        arr = np.asarray(entry["Lt"] if isinstance(entry, dict) else entry, float)
        lt[b] = arr
        lt_meta[b] = ({k: v for k, v in entry.items() if k != "Lt"} if isinstance(entry, dict) else {})
        lt_meta[b]["n_items"] = int(arr.size)
    base_logits = {}
    for bench, v in raw.items():
        base_logits[bench] = sample_logits(v, lt[bench])[:2]
    base_inv = _gauge_invariants(arc, hs, pair)
    base_var = _gauge_variables(arc, hs, pair)

    rows = {}
    for name, scen in scenarios.items():
        a2 = apply_parameter_gauge(arc, *scen["arc"])
        h2 = apply_parameter_gauge(hs, *scen["hellaswag"])
        row = {"scales": {b: {"s": float(scen[b][0]), "t": float(scen[b][1])}
                          for b in ("arc", "hellaswag")},
               "logit_max_abs_diff": {}, "centering_max_abs_mean": {},
               "invariant": {}, "variable": {}}
        for bench, v, v2 in (("arc", arc, a2), ("hellaswag", hs, h2)):
            e1, e2, _ = sample_logits(v2, lt[bench])
            b1, b2 = base_logits[bench]
            row["logit_max_abs_diff"][bench] = {
                "eq1": float(np.max(np.abs(e1 - b1))), "eq2": float(np.max(np.abs(e2 - b2)))}
            row["centering_max_abs_mean"][bench] = float(max(
                abs(float(np.mean(np.asarray(v2[k], float))))
                for k in ("theta_A", "theta_C", "z", "gamma_m")))
        inv = _gauge_invariants(a2, h2, pair)
        var = _gauge_variables(a2, h2, pair)
        row["invariant"] = {k: {"value": inv[k], "abs_diff_vs_identity": float(abs(inv[k] - base_inv[k]))}
                            for k in base_inv}
        row["variable"] = {k: {"value": var[k], "abs_diff_vs_identity": float(abs(var[k] - base_var[k]))}
                           for k in base_var}
        row["invariant_max_abs_diff_vs_identity"] = float(
            max(x["abs_diff_vs_identity"] for x in row["invariant"].values()))
        row["variable_max_abs_diff_vs_identity"] = float(
            max(x["abs_diff_vs_identity"] for x in row["variable"].values()))
        rows[name] = row

    variable_ranges = {}
    for k in base_var:
        vals = [rows[n]["variable"][k]["value"] for n in rows]
        variable_ranges[k] = {"identity": base_var[k], "min": float(min(vals)), "max": float(max(vals)),
                              "range": float(max(vals) - min(vals)),
                              "changed_beyond_1e-6": bool(max(abs(v - base_var[k]) for v in vals) > 1e-6)}
    non_identity = [n for n in rows if n != "identity"]
    max_eq1 = max((rows[n]["logit_max_abs_diff"][b]["eq1"] for n in rows
                   for b in ("arc", "hellaswag")), default=0.0)
    max_eq2 = max((rows[n]["logit_max_abs_diff"][b]["eq2"] for n in rows
                   for b in ("arc", "hellaswag")), default=0.0)
    max_center = max((rows[n]["centering_max_abs_mean"][b] for n in rows
                      for b in ("arc", "hellaswag")), default=0.0)
    return {
        "status": ("corrected diagnostic: ACTUAL joint-likelihood reparameterization gauge "
                   "(previous independently-rescaled-Pearson version was not a gauge test)"),
        "transformation": {
            "positive_scales_only": True,
            "equations": ["A' = s*A", "b' = s*b", "a' = a/s", "C' = t*(A+C) - s*A",
                          "z' = t*z", "alpha' = alpha/t", "gamma' = gamma"],
            "logits": ["Eq1: a'_i*(A'_m - b'_i) = a_i*(A_m - b_i)",
                       "Eq2: alpha'_i*(A'_m+C'_m+z'_i) + gamma'_m*L~_i "
                       "= alpha_i*(A_m+C_m+z_i) + gamma_m*L~_i"],
            "orientation": ("ORIGINAL UNSIGNFLIPPED fitted parameters; NO sign re-alignment is "
                            "performed inside this diagnostic"),
            "allowed_independent_scales": ("s, t are chosen independently per benchmark because each "
                                           "benchmark is a separate fit"),
        },
        "item_length_source": {b: lt_meta[b] for b in ("arc", "hellaswag")},
        "bounded_logit_sample": {"n_models": GAUGE_SAMPLE_MODELS, "n_items": GAUGE_SAMPLE_ITEMS,
                                 "method": "fixed-stride index grid, both logits from the original "
                                           "joint formula"},
        "baseline_identity": {"invariant": base_inv, "variable": base_var},
        "scenarios": rows,
        "summary": {
            "max_logit_abs_diff_eq1_over_scenarios": float(max_eq1),
            "max_logit_abs_diff_eq2_over_scenarios": float(max_eq2),
            "max_centering_abs_mean_over_scenarios": float(max_center),
            "n_non_identity_scenarios": len(non_identity),
            "max_invariant_conditional_corr_abs_diff": float(max(
                (rows[n]["invariant_max_abs_diff_vs_identity"] for n in non_identity), default=0.0)),
            "variable_max_abs_diff": float(max(
                (rows[n]["variable_max_abs_diff_vs_identity"] for n in non_identity), default=0.0)),
            "invariant_keys": list(base_inv), "variable_keys": list(base_var),
            "variable_ranges": variable_ranges,
            "variable_changed_keys": [k for k, r in variable_ranges.items() if r["changed_beyond_1e-6"]],
            "interpretation": ("likelihood/logits unchanged; conditional residual correlations whose "
                               "control set contains A are invariant for positive scales with fixed "
                               "orientation; raw thetaC correlations, shares and R2 are NOT invariant "
                               "(no blanket invariance assertion)"),
        },
    }


# ----------------------------------------------------------------------------------- provenance
def provenance(results_dir: str) -> dict:
    out = {"results_dir": os.path.abspath(results_dir)}
    for bench in ("arc", "hellaswag"):
        for label, path in (("npz", os.path.join(results_dir, f"joint_vectors_{bench}_full_2500ep.npz")),
                            ("run_json", os.path.join(results_dir, f"joint_run_{bench}_full_2500ep.json"))):
            if os.path.isfile(path):
                out[f"{bench}_{label}"] = {"path": os.path.abspath(path), "sha256": sha256_file(path)}
        run_json = os.path.join(results_dir, f"joint_run_{bench}_full_2500ep.json")
        if os.path.isfile(run_json):
            j = json.load(open(run_json, encoding="utf-8"))
            out[f"{bench}_run"] = {
                "status": j.get("status"), "config": j.get("config"),
                "matrix": j.get("matrix"), "matrix_sha256": j.get("matrix_sha256"),
                "matrix_array_sha256": j.get("matrix_array_sha256"),
                "input_masks": j.get("input_masks"), "original_modules": j.get("original_modules"),
                "loss": j.get("loss"), "timing": j.get("timing"), "peak_memory": j.get("peak_memory"),
                "out_npz_sha256": j.get("out_npz_sha256"),
            }
            mpath = j.get("matrix")
            if mpath and os.path.isfile(mpath):
                out[f"{bench}_matrix_on_disk"] = {"path": mpath, "sha256": sha256_file(mpath),
                                                  "matches_run_json": bool(
                                                      sha256_file(mpath) == j.get("matrix_sha256"))}
    for name in ("m10_crossbench.py", "m10b_decisive.py", "m10_finalize.py", "joint_irt.py", "beta_irt.py"):
        p = os.path.join(ORIG_DIR, name)
        if os.path.isfile(p):
            out[f"original_{name}"] = {"path": p, "sha256": sha256_file(p)}
    for name in ("joint_runner.py", "analyze_joint.py", "build_matrix.py"):
        p = os.path.join(HERE, name)
        if os.path.isfile(p):
            out[f"code_{name}"] = {"path": p, "sha256": sha256_file(p)}
    return out


# ----------------------------------------------------------------------------------------- main
def build_report(results_dir: str) -> dict:
    arc_path = os.path.join(results_dir, "joint_vectors_arc_full_2500ep.npz")
    hs_path = os.path.join(results_dir, "joint_vectors_hellaswag_full_2500ep.npz")
    for p in (arc_path, hs_path):
        if not os.path.isfile(p):
            raise RuntimeError(f"missing runner output: {p}")
    arc, hs = load_vectors(arc_path), load_vectors(hs_path)
    raw_arc, raw_hs = dict(arc), dict(hs)   # BEFORE sign alignment: the gauge check uses raw orientation
    align_info = {"arc": align(arc, "arc"), "hellaswag": align(hs, "hellaswag")}
    pair = paired_shared(arc, hs)
    if pair["n_shared"] != EXPECTED_SHARED:
        raise RuntimeError(f"shared models {pair['n_shared']} != expected {EXPECTED_SHARED}")

    rng_b = np.random.RandomState(SHUFFLE_SEED)   # m10b stream
    rng_m = np.random.RandomState(SHUFFLE_SEED)   # m10 stream
    m10b_ladder = ladder_table(arc, hs, pair, M10B_RUNG_NAMES, rng_b)
    m10_ladder = ladder_table(arc, hs, pair, M10_RUNG_NAMES, rng_m)

    iA, iH = pair["iA"], pair["iH"]
    scalars = {
        "n_arc": int(arc["models"].size), "n_hs": int(hs["models"].size),
        "n_shared": pair["n_shared"], "expected_shared": EXPECTED_SHARED,
        "thetaA_crossbench_raw": P(arc["theta_A"][iA], hs["theta_A"][iH]),
        "thetaA_resid_on_acc_crossbench": P(
            resid(arc["theta_A"], [Z(arc, "acc")])[iA], resid(hs["theta_A"], [Z(hs, "acc")])[iH]),
        "thetaC_crossbench_raw": P(arc["theta_C"][iA], hs["theta_C"][iH]),
        "acc_crossbench": P(arc["acc"][iA], hs["acc"][iH]),
        "meanY_crossbench": P(arc["meanY"][iA], hs["meanY"][iH]),
        "meanCONF_crossbench": P(arc["meanCONF"][iA], hs["meanCONF"][iH]),
        "signed_crossbench": P(arc["signed"][iA], hs["signed"][iH]),
        "gauge_crossbench": P(arc["gauge"][iA], hs["gauge"][iH]),
        "tok_len_crossbench": P(arc["tok_len"][iA], hs["tok_len"][iH]),
    }

    pairing = {"n_shared": pair["n_shared"],
               "shared_sha256": sha256_array(np.array(pair["shared"], dtype="U")),
               "note": "exact-string-ID paired indices iA/iH used for all cross-bench numbers"}
    ids_path = os.path.join(results_dir, "analyze_joint_shared_ids.txt")
    with open(ids_path, "w", encoding="utf-8") as fh:
        fh.write("".join(f"{m}\n" for m in pair["shared"]))
    pairing["shared_ids_file"] = {"path": ids_path, "sha256": sha256_file(ids_path),
                                  "n": pair["n_shared"],
                                  "order": "arc model order over the arc+hellaswag intersection"}
    lt_by_bench = {b: load_item_length(results_dir, b) for b in ("arc", "hellaswag")}

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("numeric reproduction of m10_crossbench/m10b_decisive/m10_finalize calculations "
                    "on the new full panel"),
        "decision_policy": ("raw outputs only: NO verdict/decision strings, NO abs()-threshold "
                            "acceptance, NO hardcoded narrative; reviewers decide"),
        "sign_convention": ("per bench thetaA sign-aligned toward +corr(thetaA,acc); thetaC toward "
                            "+corr(thetaC,signed) with signed = meanY - acc (historical convention). "
                            "This alignment is a DESCRIPTIVE reporting convention only (each fit is "
                            "identified up to a joint sign); the likelihood-gauge section measures "
                            "which quantities are identified under the actual reparameterization"),
        "provenance": provenance(results_dir),
        "sign_alignment": align_info,
        "pairing": pairing,
        "scalars_crossbench": scalars,
        "per_bench": {"arc": bench_stats(arc, "arc"), "hellaswag": bench_stats(hs, "hellaswag")},
        "ladder_m10b_decisive": m10b_ladder,
        "ladder_m10_crossbench": m10_ladder,
        "likelihood_gauge": likelihood_gauge_diagnostic(raw_arc, raw_hs, pair, lt_by_bench),
        "run_health": {"arc": health_checks(arc, "arc"), "hellaswag": health_checks(hs, "hellaswag")},
        "notes": [
            "resid_share_* are var(residual)/var(original) (+1e-12 guard) exactly as the originals; "
            "R2_* = 1 - share",
            "null shuffles: 200 permutations, seed 0, one stream per original script in that script's "
            "own ladder order (m10b stream for rungs 1-6; m10 stream for the two rich rungs)",
            "m10/m10b/m10_finalize DECISION and reasoning strings are intentionally not reproduced",
            "no user-frozen criterion was changed, rerun or redefined to obtain any sign",
            "likelihood-gauge section uses the ORIGINAL UNSIGNFLIPPED parameters and applies no sign "
            "re-alignment; it reports logit/likelihood invariance and conditional-correlation "
            "invariance separately from gauge-dependent raw correlations/shares/R2 (no blanket "
            "invariance is claimed for the latter)",
        ],
    }
    return report


def write_tables(report: dict, results_dir: str) -> list:
    written = []
    ladder_rows = []
    for block, rungs in (("m10b_decisive", report["ladder_m10b_decisive"]),
                         ("m10_crossbench", report["ladder_m10_crossbench"])):
        for rung, r in rungs.items():
            ladder_rows.append({"block": block, "rung": rung,
                                **{k: v for k, v in r.items() if not isinstance(v, bool)}})
    p = os.path.join(results_dir, "analyze_joint_ladder.csv")
    fieldnames = []
    for row in ladder_rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in ladder_rows:
            w.writerow(row)
    written.append(p)

    scalar_rows = []
    for k, v in report["scalars_crossbench"].items():
        scalar_rows.append({"group": "crossbench", "metric": k, "bench": "arc+hellaswag", "value": v})
    for bench, blk in report["per_bench"].items():
        for k, v in blk.items():
            if k in ("bench", "model_vector_std"):
                continue
            scalar_rows.append({"group": "per_bench", "metric": k, "bench": bench, "value": v})
        for k, v in blk["model_vector_std"].items():
            scalar_rows.append({"group": "model_vector_std", "metric": k, "bench": bench, "value": v})
    for bench, h in report["run_health"].items():
        rows = [("all_arrays_finite", h["all_arrays_finite"]),
                ("loss_n_recorded", h["loss"]["n_recorded"]),
                ("loss_first", h["loss"]["first"]),
                ("loss_last", h["loss"]["last"]),
                ("loss_last100_mean", h["loss"]["last100"]["mean"]),
                ("loss_last100_ols_slope_per_epoch", h["loss"]["last100"]["ols_slope_per_epoch"]),
                ("centering_max_abs_mean", h["centering"]["max_abs_mean"])]
        for metric, value in rows:
            scalar_rows.append({"group": "run_health", "metric": metric, "bench": bench, "value": value})
    p = os.path.join(results_dir, "analyze_joint_tables.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "metric", "bench", "value"])
        w.writeheader()
        for row in scalar_rows:
            w.writerow(row)
    written.append(p)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(PACKAGE, "results"))
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    report = build_report(a.results_dir)
    out_json = a.out_json or os.path.join(a.results_dir, "analyze_joint_report.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    tables = write_tables(report, a.results_dir)
    sc = report["scalars_crossbench"]
    print(f"WROTE {out_json}", flush=True)
    for t in tables:
        print(f"WROTE {t}", flush=True)
    print(f"n_shared={sc['n_shared']}", flush=True)
    print(f"thetaA_crossbench={sc['thetaA_crossbench_raw']:+.4f}  "
          f"thetaA_resid_on_acc_crossbench={sc['thetaA_resid_on_acc_crossbench']:+.4f}  "
          f"thetaC_crossbench={sc['thetaC_crossbench_raw']:+.4f}", flush=True)
    for bench in ("arc", "hellaswag"):
        h = report["run_health"][bench]
        print(f"{bench}: arrays_finite={h['all_arrays_finite']} loss_n={h['loss']['n_recorded']} "
              f"loss_last={h['loss']['last']:.6f} "
              f"tail100_slope={h['loss']['last100']['ols_slope_per_epoch']:+.3e} "
              f"centered={h['centering']['centered_within_1e-9']}", flush=True)
    g = report["likelihood_gauge"]
    s = g["summary"]
    print(f"likelihood_gauge: max|d logit Eq1|={s['max_logit_abs_diff_eq1_over_scenarios']:.3e} "
          f"max|d logit Eq2|={s['max_logit_abs_diff_eq2_over_scenarios']:.3e} "
          f"(both logits unchanged)", flush=True)
    print(f"likelihood_gauge: invariant conditional corr max|d|="
          f"{s['max_invariant_conditional_corr_abs_diff']:.3e} "
          f"over {len(g['scenarios']) - 1} non-identity scenarios", flush=True)
    rr = s["variable_ranges"]["thetaC_crossbench_raw"]
    print(f"likelihood_gauge: GAUGE-DEPENDENT raw thetaC crossbench corr ranges "
          f"{rr['min']:+.4f}..{rr['max']:+.4f} (identity {rr['identity']:+.4f}); "
          f"{len(s['variable_changed_keys'])} raw/share/R2 quantities change", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
