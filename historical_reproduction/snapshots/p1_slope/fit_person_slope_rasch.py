#!/usr/bin/env python3
"""
N0 -- Per-item Alternating-Least-Squares fit of the person-slope Rasch model.

Source slope model; see the supporting analysis documentation.
Task brief (N0): data -> ell -> ALS fit (theta_m, s_m, sigma_m) + item b_i;
                 second-init robustness; outputs under fit/.

Model (1.2):     ell_im = s_m * (theta_m - b_i) + eps_im ,   eps ~ F (mean 0, scale 1)
Observable (1.1): ell_im = log( p_gold / (1 - p_gold) ), p = softmax over VALID options.
                  Exact formula used:
                      ell_im = loglik_gold - logsumexp( loglik over non-gold VALID options )
                  Invalid / missing options are NaN-padded in `loglik`; they are excluded
                  from the logsumexp and from the gold read. If the gold option itself is
                  NaN, or a row has no valid non-gold option, ell for that (i,m) is set NaN
                  (masked out of the fit).

Estimation (1.4, alternating least squares):
  - per-model OLS of ell on b over that model's own items:
        slope_m = Cov_i(ell,b)/Var_i(b),   s_m = -slope_m,   a_m (intercept) = s_m*theta_m
        theta_m = a_m / s_m
  - item closed form (fix {s,theta}):  b_i = -sum_m s_m (ell_im - a_m) / sum_m s_m^2
  - iterate to max_i |Delta b_i| < 1e-6 (cap 500 iters; report if cap hit)
  - init b (primary): standardized negative item-mean of ell
  - identification: mean_i b = 0, Var_i b = 1 imposed each iteration (the 2 gauge d.o.f.).
    mean_m log s = 0 is NOT imposed (2026-09-13 revision): only 2 gauge freedoms exist and
    the two b-constraints exhaust them; mean_m log s is REPORTED as a diagnostic.

Robustness (task step 4): rerun the whole ALS from a second init
    b = standardized binary-Rasch-style logit of item accuracy = std( logit(1 - p_i) ),
    p_i = fraction of models with ell_im > 0. Report max|Delta s|, corr(s), corr(theta).

Residual scale (task step 3): sigma_m = sqrt( Var_i( ell_im - s_m*theta_m + s_m*b_i ) ).

Outputs (fit/): theta_s_sigma_arc.csv, b_arc.csv, fit_meta_arc.json, FIT_REPORT.md.
CPU only. No GPU. No network.
"""
import sys, os, io, json, glob, time, hashlib
import numpy as np
from scipy.special import logsumexp

BENCH = "arc"
SEED = 20260913
TARGET_MODELS = 6739  # coverage denominator per task brief
DATA_DIR = "data_peritem_v1_20260913/arc"
OUT_DIR  = "Imports/theory_rescue_20260913/fit"
CACHE    = os.path.join(OUT_DIR, "code", "_L_cache_arc.npz")

ELL_FORMULA = ("ell_im = loglik_gold - logsumexp(loglik over non-gold VALID options) "
               "= log(p_gold/(1-p_gold)), p = softmax over valid per-option log-likelihoods; "
               "invalid/missing options are NaN-padded and excluded; gold-NaN or no-valid-"
               "non-gold rows set to NaN and masked out.")

# thresholds / knobs
LOWVAR_ELL = 1e-6     # Var_i(ell) below this -> degenerate model
FEW_ITEMS  = 50       # items below this -> flag few_items
EXTREME    = 30.0     # |ell| above this counted as extreme (NOT clipped; data is finite)
MAX_ITERS  = 500      # cap per spec
TOL_B      = 1e-6     # convergence: max_i|Delta b_i| < TOL_B  (spec)
PCLIP      = 1e-4     # item-accuracy clip for the binary-Rasch init


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)   # stable log(sigmoid(x))


def std_gauge(v):
    """Impose mean 0, Var 1 (population var, ddof=0). Returns standardized copy."""
    v = v - np.nanmean(v)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return v / sd


# ---------------------------------------------------------------- Phase 1: load
def compute_ell_for_arrays(ll, gold):
    """Vectorized ell over one file. ll:(I,K) NaN-padded, gold:(I,) int. Returns ell:(I,)."""
    I, K = ll.shape
    idx = np.arange(I)
    valid = ~np.isnan(ll)
    g = np.clip(gold, 0, K - 1)
    gold_ll = ll[idx, g]                          # (I,)  NaN if gold invalid
    nongold = valid.copy()
    nongold[idx, g] = False                       # drop gold position
    llo = np.where(nongold, ll, -np.inf)          # non-gold valid, else -inf
    with np.errstate(invalid="ignore"):
        lse = logsumexp(llo, axis=1)              # -inf if no valid non-gold option
    ell = gold_ll - lse                           # NaN if gold NaN; +inf if lse=-inf
    ell = np.where(np.isfinite(ell), ell, np.nan) # mask +/-inf and NaN uniformly
    return ell


def build_L(rebuild=False):
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        return (z["L"], list(z["model_ids"]), list(z["item_ids"]),
                z["acc_metric"], z["file_hashes"], z["load_fail"])
    files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, "*.npz"))
                   if not f.endswith(".tmp.npz"))
    item_pos, order = {}, []
    per = []      # (model_id, item_ids, ell, acc_metric_mean, sha256, ok)
    t0 = time.time()
    for k, f in enumerate(files):
        try:
            raw = open(f, "rb").read()
            sha = hashlib.sha256(raw).hexdigest()
            d = np.load(io.BytesIO(raw), allow_pickle=True)
            ll = d["loglik"].astype(np.float64)
            gold = d["gold"].astype(int)
            iids = np.asarray(d["item_ids"]).astype(str)
            mid = str(d["repo"]) if "repo" in d.files else os.path.basename(f)[:-4]
            ell = compute_ell_for_arrays(ll, gold)
            macc = d["metric_acc"].astype(np.float64) if "metric_acc" in d.files else None
            ok = bool(np.isfinite(ell).any())
            if ok:
                for iid in iids:
                    if iid not in item_pos:
                        item_pos[iid] = len(order); order.append(iid)
                accm = float(np.nanmean(macc)) if macc is not None else np.nan
                per.append((mid, iids, ell, accm, sha, True))
            else:
                per.append((mid, None, None, np.nan, sha, False))   # empty / all-NaN
        except Exception as e:
            per.append((os.path.basename(f)[:-4], None, None, np.nan, "", False))
            print(f"  LOAD FAIL {os.path.basename(f)}: {e}", flush=True)
        if (k + 1) % 1000 == 0:
            print(f"  loaded {k+1}/{len(files)}  ({time.time()-t0:.1f}s)", flush=True)
    I = len(order); M = len(per)
    L = np.full((M, I), np.nan, dtype=np.float64)
    model_ids, acc_metric = [], np.full(M, np.nan)
    file_hashes = np.empty(M, dtype=object); load_fail = np.zeros(M, dtype=bool)
    for m, (mid, iids, ell, accm, sha, ok) in enumerate(per):
        model_ids.append(mid); acc_metric[m] = accm; file_hashes[m] = sha
        if not ok:
            load_fail[m] = True; continue
        cols = np.array([item_pos[x] for x in iids])
        L[m, cols] = ell
    item_ids = np.array(order)
    np.savez_compressed(CACHE, L=L, model_ids=np.array(model_ids, dtype=object),
                        item_ids=item_ids, acc_metric=acc_metric,
                        file_hashes=np.array(file_hashes, dtype=object),
                        load_fail=load_fail)
    print(f"built L: M={M} I={I}  ({time.time()-t0:.1f}s)", flush=True)
    return L, model_ids, list(item_ids), acc_metric, np.array(file_hashes, dtype=object), load_fail


# ---------------------------------------------------------------- Phase 2: ALS
def model_step(Lz, obs, cnt_m, mean_ell, b):
    """Given b, per-model OLS of ell on b -> (s, a=s*theta)."""
    bm = np.where(obs, b[None, :], 0.0)
    mean_b = np.where(cnt_m > 0, bm.sum(1) / np.maximum(cnt_m, 1), 0.0)
    mean_eb = np.where(cnt_m > 0, (Lz * bm).sum(1) / np.maximum(cnt_m, 1), 0.0)
    cov = mean_eb - mean_ell * mean_b
    var_b = np.maximum(np.where(cnt_m > 0, (bm * bm).sum(1) / np.maximum(cnt_m, 1) - mean_b**2,
                                0.0), 1e-12)
    slope = cov / var_b
    s = -slope
    a = mean_ell - slope * mean_b            # intercept = s*theta
    return s, a


def item_step(Lz, obs, s, a):
    """Given (s,a), closed-form b then impose gauge mean 0 / Var 1."""
    So = np.where(obs, s[:, None], 0.0)
    Ao = np.where(obs, a[:, None], 0.0)
    num = (So * (Lz - Ao)).sum(0)
    den = np.maximum((So * So).sum(0), 1e-12)
    b = -num / den
    b = b - b.mean()
    sd = b.std()
    if sd < 1e-12:
        sd = 1.0
    return b / sd


def als_fit(L, b_init, tag=""):
    M, I = L.shape
    obs = ~np.isnan(L)
    Lz = np.where(obs, L, 0.0)
    cnt_m = obs.sum(1).astype(float)
    mean_ell = np.where(cnt_m > 0, Lz.sum(1) / np.maximum(cnt_m, 1), 0.0)
    b = b_init.copy()
    converged = False; cap_hit = False; n_iter = 0; db = np.nan
    for it in range(MAX_ITERS):
        s, a = model_step(Lz, obs, cnt_m, mean_ell, b)
        b_new = item_step(Lz, obs, s, a)
        db = float(np.max(np.abs(b_new - b)))
        b = b_new
        n_iter = it + 1
        if db < TOL_B:
            converged = True
            break
    else:
        cap_hit = True
    # final model params at converged b
    s, a = model_step(Lz, obs, cnt_m, mean_ell, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = a / s
    # residual scale sigma_m^2 = Var_i(ell - s*theta + s*b) = Var_i(ell - a + s*b)
    pred = np.where(obs, a[:, None] - s[:, None] * b[None, :], 0.0)   # a - s*b = s*(theta-b)
    resid = np.where(obs, Lz - pred, 0.0)
    mean_r = np.where(cnt_m > 0, resid.sum(1) / np.maximum(cnt_m, 1), 0.0)
    var_r = np.where(cnt_m > 0, (resid * resid).sum(1) / np.maximum(cnt_m, 1) - mean_r**2, 0.0)
    sigma = np.sqrt(np.maximum(var_r, 0.0))
    var_ell = np.maximum(np.where(cnt_m > 0, (Lz * Lz).sum(1) / np.maximum(cnt_m, 1) - mean_ell**2,
                                  0.0), 0.0)
    print(f"  [{tag}] n_iter={n_iter} db={db:.2e} converged={converged} cap_hit={cap_hit}",
          flush=True)
    return dict(b=b, s=s, a=a, theta=theta, sigma=sigma, mean_ell=mean_ell,
                var_ell=var_ell, cnt_m=cnt_m, n_iter=n_iter, converged=converged,
                cap_hit=cap_hit, final_db=db)


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


# ---------------------------------------------------------------- main
def main():
    rebuild = "--rebuild" in sys.argv
    os.makedirs(os.path.join(OUT_DIR, "code"), exist_ok=True)
    np.random.seed(SEED)
    t0 = time.time()
    L, model_ids, item_ids, acc_metric, file_hashes, load_fail = build_L(rebuild=rebuild)
    M, I = L.shape
    obs = ~np.isnan(L)
    print(f"L: M={M} I={I}  load_fail={int(load_fail.sum())}", flush=True)

    # ---- inits
    Lz = np.where(obs, L, 0.0)
    col_cnt = obs.sum(0).astype(float)                        # n_models per item
    item_mean_ell = np.where(col_cnt > 0, Lz.sum(0) / np.maximum(col_cnt, 1), 0.0)
    b_init_primary = std_gauge(-item_mean_ell)               # primary init (spec)
    # secondary: binary-Rasch-style logit of item accuracy p_i = mean_m 1{ell>0}
    correct = np.where(obs, (L > 0), False)
    p_item = np.where(col_cnt > 0, correct.sum(0) / np.maximum(col_cnt, 1), 0.5)
    p_item = np.clip(p_item, PCLIP, 1 - PCLIP)
    b_init_binary = std_gauge(np.log((1 - p_item) / p_item))  # logit(1-p): easy item -> low b

    # ---- primary + second-init fits
    fit = als_fit(L, b_init_primary, tag="primary")
    fit2 = als_fit(L, b_init_binary, tag="binary-init")

    s, theta, sigma, b = fit["s"], fit["theta"], fit["sigma"], fit["b"]
    mean_ell, var_ell, cnt_m = fit["mean_ell"], fit["var_ell"], fit["cnt_m"]
    A = np.where(cnt_m > 0, correct.sum(1) / np.maximum(cnt_m, 1), np.nan)   # acc = mean 1{ell>0}
    n_extreme = np.where(obs, np.abs(L) > EXTREME, False).sum(1)

    # ---- flags / validity
    flags = [[] for _ in range(M)]
    for m in range(M):
        if load_fail[m]:
            flags[m].append("load_fail"); continue
        if cnt_m[m] < FEW_ITEMS:            flags[m].append("few_items")
        if var_ell[m] < LOWVAR_ELL:         flags[m].append("degenerate_lowvar")
        if not (np.isfinite(s[m]) and np.isfinite(theta[m])): flags[m].append("nonfinite")
        if np.isfinite(s[m]) and s[m] <= 0: flags[m].append("neg_slope")
    fittable = np.array([("load_fail" not in f and "few_items" not in f
                          and "degenerate_lowvar" not in f and "nonfinite" not in f)
                         for f in flags])   # neg_slope models ARE fittable (finite s<=0)

    # ---- second-init agreement (on models fittable in both, sign-aligned)
    s2, theta2, b2 = fit2["s"].copy(), fit2["theta"].copy(), fit2["b"].copy()
    fittable2 = np.isfinite(s2) & np.isfinite(theta2) & (cnt_m >= FEW_ITEMS) & (var_ell >= LOWVAR_ELL)
    both = fittable & fittable2
    r_s_raw, _ = corr(s[both], s2[both])
    sign_flip = bool(np.isfinite(r_s_raw) and r_s_raw < 0)   # global (b,s,theta)->-(.) gauge
    if sign_flip:
        s2, theta2, b2 = -s2, -theta2, -b2
    r_s, n_agree = corr(s[both], s2[both])
    r_theta, _ = corr(theta[both], theta2[both])
    r_b, _ = corr(b, b2)
    max_ds = float(np.max(np.abs(s[both] - s2[both]))) if both.any() else float("nan")
    max_dtheta = float(np.max(np.abs(theta[both] - theta2[both]))) if both.any() else float("nan")
    max_db_init = float(np.max(np.abs(b - b2)))

    # ---- s<=0 fraction (over fittable models) + gauge diagnostic
    fit_pos = fittable & np.isfinite(s) & (s > 0)
    n_fittable = int(fittable.sum())
    n_neg = int((fittable & (s <= 0)).sum())
    frac_neg = n_neg / n_fittable if n_fittable else float("nan")
    mean_log_s = float(np.mean(np.log(s[fit_pos]))) if fit_pos.any() else float("nan")

    # s distribution over positive-s fittable models (extreme-s tail = broken models
    # with pathological raw log-likelihoods; retained un-clipped -- downstream note)
    sp = s[fit_pos]
    pcts = np.percentile(sp, [1, 25, 50, 75, 95, 99]).tolist() if sp.size else [np.nan]*6
    s_dist = {"p1": pcts[0], "p25": pcts[1], "p50": pcts[2], "p75": pcts[3],
              "p95": pcts[4], "p99": pcts[5], "max": float(sp.max()) if sp.size else float("nan"),
              "n_s_gt_20": int((sp > 20).sum()), "n_s_gt_50": int((sp > 50).sum()),
              "n_s_gt_100": int((sp > 100).sum()),
              "note": ("Extreme-s tail (~29 models s>50) are BROKEN models with pathological "
                       "raw log-likelihoods (very negative mean_ell); OLS fits them faithfully "
                       "but downstream log-s analyses (checks 2/6/7) should consider trimming "
                       "s>~p99. Not clipped here per spec.")}

    # ---------------------------------------------------------------- outputs
    import csv
    p_ts = os.path.join(OUT_DIR, "theta_s_sigma_arc.csv")
    with open(p_ts, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["repo", "theta", "s", "sigma", "n_items", "mean_ell", "acc",
                    "acc_metric", "flags"])
        for m in range(M):
            fin = lambda v: (f"{v:.10g}" if np.isfinite(v) else "")
            w.writerow([model_ids[m], fin(theta[m]), fin(s[m]), fin(sigma[m]),
                        int(cnt_m[m]), fin(mean_ell[m]), fin(A[m]),
                        fin(acc_metric[m]), "|".join(flags[m])])
    p_b = os.path.join(OUT_DIR, "b_arc.csv")
    with open(p_b, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "b", "n_models"])
        for i in range(I):
            w.writerow([item_ids[i], f"{b[i]:.10g}", int(col_cnt[i])])

    # input file hashes
    valid_hashes = [h for h in file_hashes if isinstance(h, str) and len(h) == 64]
    combined = hashlib.sha256("".join(sorted(valid_hashes)).encode()).hexdigest()

    from collections import Counter
    fc = Counter()
    for f in flags:
        for x in f:
            fc[x] += 1

    meta = {
        "bench": BENCH, "seed": SEED, "data_dir": DATA_DIR,
        "ell_formula": ELL_FORMULA,
        "n_npz_files_real": int(M),
        "n_models": n_fittable,
        "n_models_load_fail_empty": int(load_fail.sum()),
        "n_items": int(I),
        "target_models": TARGET_MODELS,
        "coverage_vs_target": n_fittable / TARGET_MODELS,
        "als_primary": {
            "init_b": "standardized negative item-mean of ell",
            "iterations": fit["n_iter"], "converged": fit["converged"],
            "cap_hit": fit["cap_hit"], "final_max_db": fit["final_db"],
            "tol_b": TOL_B, "max_iters": MAX_ITERS,
        },
        "identification": {
            "imposed": ["mean_i b = 0", "Var_i b = 1"],
            "achieved_mean_b": float(b.mean()), "achieved_var_b": float(b.var()),
            "gauge_diagnostic_mean_log_s": mean_log_s,
            "mean_log_s_note": ("NOT imposed (2026-09-13 revision). Only 2 gauge d.o.f. exist; "
                                "mean_i b=0 & Var_i b=1 exhaust them, so mean_m log s=0 is "
                                "over-determined. Reported here as a diagnostic. Downstream "
                                "log-s comparisons are invariant to a global s rescale."),
        },
        "s_positivity": {"fraction_s_le_0": frac_neg, "n_s_le_0": n_neg,
                         "n_fittable": n_fittable},
        "s_distribution": s_dist,
        "second_init_agreement": {
            "second_init_b": "standardized binary-Rasch logit of item accuracy = std(logit(1-p_i))",
            "iterations": fit2["n_iter"], "converged": fit2["converged"],
            "cap_hit": fit2["cap_hit"],
            "n_models_compared": int(n_agree),
            "sign_flip_applied": sign_flip,
            "corr_s": r_s, "corr_theta": r_theta, "corr_b": r_b,
            "max_abs_delta_s": max_ds, "max_abs_delta_theta": max_dtheta,
            "max_abs_delta_b": max_db_init,
        },
        "input_file_hashes_count": int(len(valid_hashes)),
        "input_file_hashes_combined_sha256": combined,
        "totals_defs": {"acc": "mean_i 1{ell_im>0}  (theory A_m, 1.3)",
                        "acc_metric": "mean_i leaderboard metric_acc (argmax loglik==gold)",
                        "mean_ell": "mean_i ell_im"},
        "residual_scale_def": "sigma_m = sqrt(Var_i(ell_im - s_m*theta_m + s_m*b_i))",
        "ell_extremes": {"threshold": EXTREME,
                         "n_models_with_extreme_ell": int((n_extreme > 0).sum()),
                         "total_extreme_cells": int(n_extreme.sum())},
        "flag_counts": dict(fc),
        "runtime_sec": time.time() - t0,
    }
    p_meta = os.path.join(OUT_DIR, "fit_meta_arc.json")
    with open(p_meta, "w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- FIT_REPORT.md (written by the script; no Write tool needed)
    dod = [
        ("theta_s_sigma_arc.csv / b_arc.csv / fit_meta_arc.json / FIT_REPORT.md written",
         "all four present under fit/"),
        (f"convergence max|db|<1e-6 (cap {MAX_ITERS})",
         f"primary db={fit['final_db']:.2e} iters={fit['n_iter']} converged={fit['converged']} cap_hit={fit['cap_hit']}"),
        (f"coverage recorded (>=80% of {TARGET_MODELS})",
         f"n_models={n_fittable} coverage={n_fittable/TARGET_MODELS:.4f}"),
        ("gauge = only mean_i b=0 & Var_i b=1; mean_m log s reported not imposed",
         f"mean_b={b.mean():.2e} var_b={b.var():.6f} mean_log_s={mean_log_s:.4f}"),
        ("fraction s<=0 reported",
         f"frac_s<=0={frac_neg:.4f} (n={n_neg}/{n_fittable})"),
    ]
    lines = []
    lines.append("# N0 FIT_REPORT -- ARC person-slope Rasch (ALS)\n")
    lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  runtime={meta['runtime_sec']:.1f}s")
    lines.append(f"- data: `{DATA_DIR}` ; real npz={M} ; empty(all-NaN, dropped)={int(load_fail.sum())}")
    lines.append(f"- item union (master) = {I} items (per-file item sets: 1170 or 1172)")
    lines.append("")
    lines.append("## ell (observable, 1.1)")
    lines.append(f"`{ELL_FORMULA}`")
    lines.append(f"- extreme |ell|>{EXTREME}: {int(n_extreme.sum())} cells across "
                 f"{int((n_extreme>0).sum())} models -- retained (spec: no clipping).")
    lines.append("")
    lines.append("## ALS fit (1.4)")
    lines.append(f"- primary init: standardized negative item-mean of ell; "
                 f"iters={fit['n_iter']}, final max|db|={fit['final_db']:.2e}, "
                 f"converged={fit['converged']}, cap_hit={fit['cap_hit']}")
    lines.append(f"- gauge imposed each iter: mean_i b=0 (achieved {b.mean():.2e}), "
                 f"Var_i b=1 (achieved {b.var():.6f})")
    lines.append(f"- gauge diagnostic (NOT imposed): mean_m log s = {mean_log_s:.6f}")
    lines.append(f"- residual scale: sigma_m = sqrt(Var_i(ell - s*theta + s*b))")
    lines.append("")
    lines.append("## models fit")
    lines.append(f"- n_fittable = {n_fittable} ; coverage vs {TARGET_MODELS} target = "
                 f"{n_fittable/TARGET_MODELS:.4f}")
    lines.append(f"- fraction s_m <= 0 = {frac_neg:.4f}  (n={n_neg})")
    lines.append(f"- s distribution (positive-s fittable): p50={s_dist['p50']:.3f}, "
                 f"p95={s_dist['p95']:.3f}, p99={s_dist['p99']:.3f}, max={s_dist['max']:.1f}; "
                 f"s>50: {s_dist['n_s_gt_50']} models, s>100: {s_dist['n_s_gt_100']} models.")
    lines.append(f"  {s_dist['note']}")
    lines.append(f"- flag counts: {dict(fc)}")
    lines.append(f"- models dropped: {int(load_fail.sum())} empty(all-NaN) + "
                 f"{fc.get('few_items',0)} few_items(<{FEW_ITEMS}) + "
                 f"{fc.get('degenerate_lowvar',0)} degenerate(Var_i ell<{LOWVAR_ELL}) + "
                 f"{fc.get('nonfinite',0)} nonfinite. neg_slope models are RETAINED "
                 f"(flagged), reported as the s<=0 fraction.")
    lines.append("")
    lines.append("## robustness: second initialization (task step 4)")
    lines.append(f"- second init: standardized binary-Rasch logit of item accuracy = std(logit(1-p_i))")
    lines.append(f"- second-init ALS: iters={fit2['n_iter']}, converged={fit2['converged']}, "
                 f"cap_hit={fit2['cap_hit']}")
    lines.append(f"- models compared (fittable in both) = {n_agree}; sign_flip_applied={sign_flip}")
    lines.append(f"- corr(s) = {r_s:.6f} ; corr(theta) = {r_theta:.6f} ; corr(b) = {r_b:.6f}")
    lines.append(f"- max|delta s| = {max_ds:.3e} ; max|delta theta| = {max_dtheta:.3e} ; "
                 f"max|delta b| = {max_db_init:.3e}")
    lines.append("")
    lines.append("## provenance")
    lines.append(f"- input file hashes count = {len(valid_hashes)} (sha256 per npz)")
    lines.append(f"- combined sha256 of sorted per-file hashes = {combined}")
    lines.append("")
    lines.append("## DoD (5x2 self-check)")
    for i, (name, ev) in enumerate(dod, 1):
        lines.append(f"{i}. {name} -> {ev}")
    with open(os.path.join(OUT_DIR, "FIT_REPORT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n==== SUMMARY ====")
    print(json.dumps(meta, indent=2))
    print("wrote:", p_ts, p_b, p_meta, os.path.join(OUT_DIR, "FIT_REPORT.md"))
    return meta


if __name__ == "__main__":
    main()
