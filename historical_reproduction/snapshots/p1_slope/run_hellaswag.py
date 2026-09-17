#!/usr/bin/env python3
"""
N0h -- thin wrapper: run the FROZEN N0 ARC estimator (fit_person_slope_rasch.py)
on HellaSwag, WITHOUT modifying the estimator file.

Why a wrapper: fit_person_slope_rasch.py hard-codes BENCH/DATA_DIR/OUT_DIR/CACHE as
module globals AND writes output filenames literally as "theta_s_sigma_arc.csv" etc.
inside main(); running it with a changed DATA_DIR would OVERWRITE the frozen ARC
outputs. So we:
  1) import the frozen module by path (its ALS math is used verbatim, byte-untouched),
  2) monkeypatch ONLY the module globals BENCH / DATA_DIR / OUT_DIR / CACHE / TARGET_MODELS,
     pointing OUT_DIR at a scratch dir so main() writes its 4 files there (never into fit/),
  3) call mod.main() verbatim -> produces (theta,s,sigma), b, meta on HS data in scratch,
  4) copy the 3 data outputs into fit/ under *_hellaswag names (ARC outputs never touched),
  5) append an HS section to fit/FIT_REPORT.md (mirroring the ARC section fields),
  6) cross-bench: join ARC vs HS on repo -> Pearson/Spearman of theta, log s, sigma
     -> fit/crossbench_theta_s_arc_hs.csv + a table appended to FIT_REPORT.md.

CPU only. No GPU. No network.
"""
import os, sys, json, shutil, time, importlib.util
import numpy as np

ROOT      = "."
FIT_DIR   = os.path.join(ROOT, "Imports/theory_rescue_20260913/fit")
CODE      = os.path.join(FIT_DIR, "code", "fit_person_slope_rasch.py")
HS_DATA   = os.path.join(ROOT, "data_peritem_v1_20260913/hellaswag")
SCRATCH   = "runs/hs_fit_scratch"
HS_TARGET = 6748   # coverage denominator per N0h task brief

os.makedirs(os.path.join(SCRATCH, "code"), exist_ok=True)

# ---- 1) import frozen estimator by path (verbatim) -------------------------
spec = importlib.util.spec_from_file_location("fit_person_slope_rasch", CODE)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ---- 2) monkeypatch ONLY paths/labels (not the ALS math) -------------------
mod.BENCH         = "hellaswag"
mod.DATA_DIR      = HS_DATA
mod.OUT_DIR       = SCRATCH                                             # main() writes here
mod.CACHE         = os.path.join(SCRATCH, "code", "_L_cache_hellaswag.npz")
mod.TARGET_MODELS = HS_TARGET
# sanity: confirm the frozen file on disk is unchanged is checked outside (sha256)

# ---- 3) run main() verbatim -> scratch -------------------------------------
t0 = time.time()
meta = mod.main()   # writes theta_s_sigma_arc.csv, b_arc.csv, fit_meta_arc.json, FIT_REPORT.md in SCRATCH
print(f"\n[wrapper] main() done in {time.time()-t0:.1f}s")

# ---- 4) copy 3 data outputs into fit/ under *_hellaswag names --------------
copies = [
    ("theta_s_sigma_arc.csv", "theta_s_sigma_hellaswag.csv"),
    ("b_arc.csv",             "b_hellaswag.csv"),
    ("fit_meta_arc.json",     "fit_meta_hellaswag.json"),
]
for src, dst in copies:
    s = os.path.join(SCRATCH, src)
    d = os.path.join(FIT_DIR, dst)
    assert os.path.exists(s), f"missing scratch output {s}"
    shutil.copyfile(s, d)
    print(f"[wrapper] copied {src} -> fit/{dst}")
# NB: scratch/FIT_REPORT.md is ARC-titled boilerplate; discarded (we author HS section below).

# reload the copied HS meta (file system is truth)
with open(os.path.join(FIT_DIR, "fit_meta_hellaswag.json")) as fh:
    hm = json.load(fh)
print(f"[wrapper] HS coverage n_models={hm['n_models']} / target {hm['target_models']} "
      f"= {hm['coverage_vs_target']:.4f}")

# ---- 5) b distribution (task asks for item-difficulty b distribution) ------
import csv
bvals = []
with open(os.path.join(FIT_DIR, "b_hellaswag.csv")) as fh:
    r = csv.reader(fh); next(r)
    for row in r:
        try: bvals.append(float(row[1]))
        except: pass
bvals = np.array(bvals)
b_dist = {"n_items": int(bvals.size), "min": float(bvals.min()), "p1": float(np.percentile(bvals,1)),
          "p50": float(np.percentile(bvals,50)), "p99": float(np.percentile(bvals,99)),
          "max": float(bvals.max())}
print(f"[wrapper] b dist: {b_dist}")

# ---- 6) cross-bench: join ARC vs HS on repo --------------------------------
def load_theta(path):
    d = {}
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            def f(k):
                v = row[k]
                return float(v) if v not in ("", None) else np.nan
            d[row["repo"]] = (f("theta"), f("s"), f("sigma"), row["flags"])
    return d

arc = load_theta(os.path.join(FIT_DIR, "theta_s_sigma_arc.csv"))
hs  = load_theta(os.path.join(FIT_DIR, "theta_s_sigma_hellaswag.csv"))
common = sorted(set(arc) & set(hs))
print(f"[wrapper] repos: ARC={len(arc)} HS={len(hs)} common={len(common)}")

rows = []
for repo in common:
    ta, sa, siga, fa = arc[repo]
    th, sh, sigh, fh_ = hs[repo]
    rows.append((repo, ta, th, sa, sh, siga, sigh, fa, fh_))

# write per-model paired CSV
cb_path = os.path.join(FIT_DIR, "crossbench_theta_s_arc_hs.csv")
with open(cb_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["repo", "theta_arc", "theta_hs", "s_arc", "s_hs", "sigma_arc", "sigma_hs",
                "flags_arc", "flags_hs"])
    for row in rows:
        repo, ta, th, sa, sh, siga, sigh, fa, fh_ = row
        g = lambda v: (f"{v:.10g}" if np.isfinite(v) else "")
        w.writerow([repo, g(ta), g(th), g(sa), g(sh), g(siga), g(sigh), fa, fh_])
print(f"[wrapper] wrote {cb_path}  ({len(rows)} paired models)")

# correlations
from scipy.stats import pearsonr, spearmanr
ta = np.array([r[1] for r in rows]); th = np.array([r[2] for r in rows])
sa = np.array([r[3] for r in rows]); sh = np.array([r[4] for r in rows])
siga = np.array([r[5] for r in rows]); sigh = np.array([r[6] for r in rows])

def corr_pair(x, y, need_pos=False):
    m = np.isfinite(x) & np.isfinite(y)
    if need_pos:
        m &= (x > 0) & (y > 0)
    xx, yy = x[m], y[m]
    if xx.size < 3:
        return dict(n=int(xx.size), pearson=float("nan"), spearman=float("nan"))
    pr = pearsonr(xx, yy); sr = spearmanr(xx, yy)
    return dict(n=int(xx.size), pearson=float(pr.statistic), spearman=float(sr.statistic))

cb = {
    "theta":  corr_pair(ta, th),
    "log_s":  corr_pair(np.log(np.where(sa > 0, sa, np.nan)),
                        np.log(np.where(sh > 0, sh, np.nan))),  # log defined only for s>0 both
    "sigma":  corr_pair(siga, sigh),
    "n_common_repos": len(common),
}
# also a p99-trimmed log s variant (meta note: broken models with s>p99 pollute log s)
p99a = hm["s_distribution"]["p99"]
with open(os.path.join(FIT_DIR, "fit_meta_arc.json")) as fh:
    am = json.load(fh)
p99arc = am["s_distribution"]["p99"]
mtrim = (sa > 0) & (sh > 0) & (sa <= p99arc) & (sh <= p99a)
if mtrim.sum() >= 3:
    prt = pearsonr(np.log(sa[mtrim]), np.log(sh[mtrim]))
    srt = spearmanr(np.log(sa[mtrim]), np.log(sh[mtrim]))
    cb["log_s_p99trim"] = dict(n=int(mtrim.sum()), pearson=float(prt.statistic),
                               spearman=float(srt.statistic),
                               arc_p99=p99arc, hs_p99=p99a)
print("[wrapper] crossbench:", json.dumps(cb, indent=2))

# ---- 7) append HS section + crossbench table to fit/FIT_REPORT.md ----------
REPORT = os.path.join(FIT_DIR, "FIT_REPORT.md")
existing = open(REPORT).read() if os.path.exists(REPORT) else ""
if "## HELLASWAG (N0h)" in existing:
    print("[wrapper] HS section already present in FIT_REPORT.md -- NOT appending again.")
else:
    sd = hm["s_distribution"]
    ap = hm["als_primary"]; sec = hm["second_init_agreement"]; ident = hm["identification"]
    L = []
    L.append("")
    L.append("---")
    L.append("")
    L.append("# N0h FIT_REPORT -- HELLASWAG person-slope Rasch (ALS)")
    L.append("")
    L.append("## HELLASWAG (N0h)")
    L.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  runtime={hm['runtime_sec']:.1f}s")
    L.append(f"- code: FROZEN estimator `fit_person_slope_rasch.py` run verbatim via "
             f"`code/run_hellaswag.py` (module globals BENCH/DATA_DIR/OUT_DIR/CACHE/TARGET_MODELS "
             f"monkeypatched; estimator file byte-unchanged).")
    L.append(f"- data: `{hm['data_dir']}` ; real npz={hm['n_npz_files_real']} ; "
             f"empty(all-NaN, dropped)={hm['n_models_load_fail_empty']}")
    L.append(f"- item union (master) = {hm['n_items']} items")
    L.append("")
    L.append("### ell (observable, 1.1)")
    L.append(f"`{hm['ell_formula']}`")
    L.append(f"- extreme |ell|>{hm['ell_extremes']['threshold']}: "
             f"{hm['ell_extremes']['total_extreme_cells']} cells across "
             f"{hm['ell_extremes']['n_models_with_extreme_ell']} models -- retained (spec: no clipping).")
    L.append("")
    L.append("### ALS fit (1.4)")
    L.append(f"- primary init: {ap['init_b']}; iters={ap['iterations']}, "
             f"final max|db|={ap['final_max_db']:.2e}, converged={ap['converged']}, "
             f"cap_hit={ap['cap_hit']}")
    L.append(f"- gauge imposed each iter: mean_i b=0 (achieved {ident['achieved_mean_b']:.2e}), "
             f"Var_i b=1 (achieved {ident['achieved_var_b']:.6f})")
    L.append(f"- gauge diagnostic (NOT imposed): mean_m log s = "
             f"{ident['gauge_diagnostic_mean_log_s']:.6f}")
    L.append(f"- residual scale: {hm['residual_scale_def']}")
    L.append("")
    L.append("### models fit")
    L.append(f"- n_fittable = {hm['n_models']} ; coverage vs {hm['target_models']} target = "
             f"{hm['coverage_vs_target']:.4f}")
    L.append(f"- fraction s_m <= 0 = {hm['s_positivity']['fraction_s_le_0']:.4f}  "
             f"(n={hm['s_positivity']['n_s_le_0']})")
    L.append(f"- s distribution (positive-s fittable): p50={sd['p50']:.3f}, p95={sd['p95']:.3f}, "
             f"p99={sd['p99']:.3f}, max={sd['max']:.1f}; s>50: {sd['n_s_gt_50']} models, "
             f"s>100: {sd['n_s_gt_100']} models.")
    L.append(f"  {sd['note']}")
    L.append(f"- item-difficulty b distribution (gauge mean0/var1): min={b_dist['min']:.3f}, "
             f"p1={b_dist['p1']:.3f}, p50={b_dist['p50']:.3f}, p99={b_dist['p99']:.3f}, "
             f"max={b_dist['max']:.3f}  (n_items={b_dist['n_items']})")
    L.append(f"- flag counts: {hm['flag_counts']}")
    L.append("")
    L.append("### robustness: second initialization")
    L.append(f"- second init: {sec['second_init_b']}")
    L.append(f"- second-init ALS: iters={sec['iterations']}, converged={sec['converged']}, "
             f"cap_hit={sec['cap_hit']}")
    L.append(f"- models compared (fittable in both) = {sec['n_models_compared']}; "
             f"sign_flip_applied={sec['sign_flip_applied']}")
    L.append(f"- corr(s) = {sec['corr_s']:.6f} ; corr(theta) = {sec['corr_theta']:.6f} ; "
             f"corr(b) = {sec['corr_b']:.6f}")
    L.append(f"- max|delta s| = {sec['max_abs_delta_s']:.3e} ; "
             f"max|delta theta| = {sec['max_abs_delta_theta']:.3e} ; "
             f"max|delta b| = {sec['max_abs_delta_b']:.3e}")
    L.append("")
    L.append("### provenance")
    L.append(f"- input file hashes count = {hm['input_file_hashes_count']} (sha256 per npz)")
    L.append(f"- combined sha256 of sorted per-file hashes = "
             f"{hm['input_file_hashes_combined_sha256']}")
    L.append("")
    L.append("## CROSS-BENCH  ARC vs HELLASWAG  (feeds N1c)")
    L.append(f"- matched on `repo`; common repos in both fits = {cb['n_common_repos']}")
    L.append(f"- file: `fit/crossbench_theta_s_arc_hs.csv` (per-model paired theta/s/sigma)")
    L.append("")
    L.append("| quantity | n models | Pearson r | Spearman rho |")
    L.append("|---|---|---|---|")
    L.append(f"| theta_arc vs theta_hs | {cb['theta']['n']} | {cb['theta']['pearson']:.4f} | "
             f"{cb['theta']['spearman']:.4f} |")
    L.append(f"| log s_arc vs log s_hs (s>0 both) | {cb['log_s']['n']} | "
             f"{cb['log_s']['pearson']:.4f} | {cb['log_s']['spearman']:.4f} |")
    if "log_s_p99trim" in cb:
        t = cb["log_s_p99trim"]
        L.append(f"| log s (trimmed s<=p99: arc<= {t['arc_p99']:.2f}, hs<= {t['hs_p99']:.2f}) | "
                 f"{t['n']} | {t['pearson']:.4f} | {t['spearman']:.4f} |")
    L.append(f"| sigma_arc vs sigma_hs | {cb['sigma']['n']} | {cb['sigma']['pearson']:.4f} | "
             f"{cb['sigma']['spearman']:.4f} |")
    L.append("")
    with open(REPORT, "a") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[wrapper] appended HS + cross-bench section to {REPORT}")

# dump a small machine summary for the report
with open(os.path.join(FIT_DIR, "code", "run_hellaswag_summary.json"), "w") as fh:
    json.dump({"coverage": hm["coverage_vs_target"], "n_models": hm["n_models"],
               "target": hm["target_models"], "b_dist": b_dist, "crossbench": cb}, fh, indent=2)
print("[wrapper] DONE")
