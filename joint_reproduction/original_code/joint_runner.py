#!/usr/bin/env python3
"""FULLIRT-RUNTIME-EXEC - joint-only runner for the full analysis panel (CUDA).

Imports the ORIGINAL `fit_joint` / `beta_irt` modules VERBATIM from
Imports/eirt_cl_reproducible_20260611/logprob_irt_compare_20260614 (no rewrite, no copy, hashes
recorded; the imported module's __file__ is asserted to be that original path). Only part (B) of
m_irsl_joint.py is run: the joint dual-latent fit. The unreported (A) IRSL Beta-IRT fits are NOT
run; `fit_joint` re-seeds internally (torch.manual_seed(seed); np.random.seed(seed)), so a B-only
call reproduces the joint vectors of the original driver given identical inputs.

Derivation, verbatim from m_irsl_joint.py:
    BIN  = matrix BIN ; LP_mean/LP_sum = matrix arrays ; CONF = matrix CONF
    RESP = exp(LP_mean).clip(1e-6, 1-1e-6)          (Beta channel response; NaN preserved)
    tok  = LP_sum / LP_mean ; Lt = nanmedian(tok, axis=1) ; Lt -= nanmean(Lt)
Config: 2500 epochs; ARC lr=0.05 grad_clip=None; HellaSwag lr=0.02 grad_clip=5.0; seed=0;
phi=10.0; clamp_eps=1e-6; float64 (fit_joint casts internally); device CUDA.

Outputs: joint_vectors_<bench>.npz — ALL original `fit_joint`-returned arrays are persisted
(theta_A, theta_C, gamma_m, a, b, alpha, z, loss_history) plus the model vectors (acc, meanY,
meanCONF, gauge, tok_len, signed) and mode/fit_epochs — + a run JSON with ACTUAL config, loss,
item-parameter hashes, device, timing, peak memory and input sha256.  `loss_history` is the FULL
per-epoch numeric trajectory (float64, one value per epoch, 2500 for a full run) persisted in the
result NPZ itself (no scientific math change; the fit code is imported verbatim and untouched).
Every fit output is checked finite before the run is reported as completed; a non-finite array
leaves the outputs on disk and fails the run loudly.

Guards (fail closed):
  * full runs require the public complete-panel matrix identity and fixed 2500 epochs.
  * --pilot is capped at 5 epochs, must write a path containing 'pilot', and is stamped
    PILOT_ONLY so it can never be mistaken for (or reused as) a 2500-epoch fit.
  * outputs are never overwritten; CPU device is only allowed for --pilot with <= 3 epochs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time

import numpy as np

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT_MAX_EPOCHS = 5
BENCH_CONFIG = {
    "arc": {"n_epochs": 2500, "lr": 0.05, "grad_clip": None},
    "hellaswag": {"n_epochs": 2500, "lr": 0.02, "grad_clip": 5.0},
}
FIT_DEFAULTS = {"seed": 0, "phi": 10.0, "clamp_eps": 1e-6, "dtype": "float64"}
JOINT_KEYS = ("models", "theta_A", "theta_C", "gamma_m", "acc", "meanY", "meanCONF",
              "gauge", "tok_len", "signed", "a", "b", "alpha", "z")
ITEM_PARAM_KEYS = ("a", "b", "alpha", "z")
MODEL_PARAM_KEYS = ("theta_A", "theta_C", "gamma_m")


class RunnerError(RuntimeError):
    """Raised on any gate/validity violation. Never auto-approves, never overwrites."""


def project_root(start: str) -> str:
    p = os.path.abspath(start)
    for _ in range(12):
        if os.path.isdir(os.path.join(p, "Imports")):
            return p
        if os.path.dirname(p) == p:
            break
        p = os.path.dirname(p)
    raise RunnerError(f"project root (dir containing 'Imports') not found above {start}")


PROJECT_ROOT = os.path.dirname(HERE)
ORIG_DIR = HERE
JOINT_PY = os.path.join(ORIG_DIR, "joint_irt.py")
BETA_PY = os.path.join(ORIG_DIR, "beta_irt.py")
ORIGINAL_SHA256 = {
    "joint_irt.py": "9bb4eca87b3a8023719297c3d9dab00f2dca1c43a3d2844b3358716fd5c46924",
    "beta_irt.py": "ae523341f08fbb864d976aedbc00d63f36d2f71063b9606779b1f0cc933c0225",
}


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def load_original_fit():
    """Import ORIGINAL fit_joint/beta_irt verbatim from their original directory."""
    for name, path in (("joint_irt.py", JOINT_PY), ("beta_irt.py", BETA_PY)):
        if not os.path.isfile(path):
            raise RunnerError(f"original module missing: {path}")
        got = sha256_file(path)
        if got != ORIGINAL_SHA256[name]:
            raise RunnerError(f"{name} sha256 mismatch (got {got}, expected {ORIGINAL_SHA256[name]})")
    sys.path.insert(0, ORIG_DIR)
    import beta_irt  # noqa: PLC0415
    import joint_irt  # noqa: PLC0415
    if os.path.realpath(joint_irt.__file__) != os.path.realpath(JOINT_PY):
        raise RunnerError(f"joint_irt imported from {joint_irt.__file__}, expected {JOINT_PY}")
    if os.path.realpath(beta_irt.__file__) != os.path.realpath(BETA_PY):
        raise RunnerError(f"beta_irt imported from {beta_irt.__file__}, expected {BETA_PY}")
    if not hasattr(joint_irt, "fit_joint") or not callable(joint_irt.fit_joint):
        raise RunnerError("original joint_irt.fit_joint not found")
    if getattr(joint_irt, "beta_nll", None) is not getattr(beta_irt, "beta_nll", object()):
        raise RunnerError("joint_irt is not using beta_irt.beta_nll verbatim")
    return joint_irt, beta_irt


def derive_inputs(matrix_path: str) -> dict:
    """Load + validate a built matrix and derive (BIN, RESP, Lt) exactly as m_irsl_joint.py."""
    if not os.path.isfile(matrix_path):
        raise RunnerError(f"matrix not found: {matrix_path}")
    with np.load(matrix_path, allow_pickle=False) as z:
        missing = [k for k in ("items", "models", "BIN", "LP_mean", "LP_sum", "CONF") if k not in z.files]
        if missing:
            raise RunnerError(f"{matrix_path}: required key(s) absent: {missing}")
        items = [str(x) for x in z["items"]]
        models = [str(x) for x in z["models"]]
        BIN = z["BIN"].astype(np.float64)
        LPm = z["LP_mean"].astype(np.float64)
        LPs = z["LP_sum"].astype(np.float64)
        CONF = z["CONF"].astype(np.float64)
    if not (BIN.shape == LPm.shape == LPs.shape == CONF.shape):
        raise RunnerError(f"matrix shape mismatch: {BIN.shape} {LPm.shape} {LPs.shape} {CONF.shape}")
    if BIN.shape != (len(items), len(models)):
        raise RunnerError(f"matrix shape {BIN.shape} != (items={len(items)}, models={len(models)})")
    if len(set(models)) != len(models) or len(set(items)) != len(items):
        raise RunnerError("duplicate model or item ids in matrix")
    if BIN[~np.isnan(BIN)].size and not np.isin(BIN[~np.isnan(BIN)], (0.0, 1.0)).all():
        raise RunnerError("BIN contains values outside {0,1,NaN}")
    # NaN (missing) is preserved end-to-end; no imputation, no metric substitution.
    RESP = np.exp(LPm).clip(1e-6, 1 - 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        tok = LPs / LPm
    Lt = np.nanmedian(tok, axis=1)
    Lt = Lt - np.nanmean(Lt)
    if not np.isfinite(Lt).all():
        raise RunnerError(f"Lt has {int((~np.isfinite(Lt)).sum())} non-finite entries (item token median degenerate)")
    n1 = int(np.sum(~np.isnan(BIN)))
    n2 = int(np.sum(~np.isnan(RESP)))
    if n1 == 0 or n2 == 0:
        raise RunnerError(f"empty channel mask: n1={n1} (BIN), n2={n2} (RESP)")
    conf = np.nanmean(RESP, 0)
    acc = np.nanmean(BIN, 0)
    vectors = {
        "models": np.array(models), "acc": acc, "meanY": conf, "meanCONF": np.nanmean(CONF, 0),
        "gauge": np.nanmean(LPm, 0), "tok_len": np.nanmean(tok, 0), "signed": conf - acc,
    }
    return {"items": items, "models": models, "BIN": BIN, "RESP": RESP, "Lt": Lt, "vectors": vectors,
            "n1": n1, "n2": n2, "conf": CONF,
            "matrix_sha256": sha256_file(matrix_path),
            "array_sha256": {k: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                             for k, a in (("BIN", BIN), ("LP_mean", LPm), ("LP_sum", LPs), ("CONF", CONF))},
            "shape": list(BIN.shape)}


def check_input_contract(path: str | None, *, bench: str, matrix_sha256: str, epochs: int) -> dict:
    """Validate the public complete-panel matrix and fixed epoch count."""
    path = path or os.path.join(HERE, '..', 'input_contract', 'CONTRACT.json')
    with open(path, encoding='utf-8') as stream:
        contract = json.load(stream)
    if bench not in contract['benches'] or epochs != contract['rules']['epochs']:
        raise RunnerError('Full-fit benchmark or epoch contract mismatch')
    if matrix_sha256 != contract['benches'][bench]['matrix_archived_sha256']:
        raise RunnerError('Full-fit matrix identity mismatch; use the public cold wrapper for reconstructed matrices')
    return {'contract_sha256':sha256_file(path),'bench':bench,'epochs':epochs}


def enforce_output_finiteness(jt: dict, *, n_models: int, n_items: int) -> dict:
    """Enforce that EVERY original fit-returned array is finite and correctly shaped.

    Support for later numerical verification: returns shapes + sha256 of each persisted array.
    Raises RunnerError (outputs stay on disk for inspection) on any non-finite/missing/mis-shaped
    array; fit math is never altered.
    """
    problems: dict[str, str] = {}
    info: dict[str, dict] = {}
    for k in MODEL_PARAM_KEYS:
        if k not in jt:
            problems[k] = "missing from fit output"
            continue
        arr = np.asarray(jt[k], dtype=np.float64)
        if arr.shape != (n_models,):
            problems[k] = f"shape {arr.shape} != ({n_models},)"
        elif not np.isfinite(arr).all():
            problems[k] = f"{int((~np.isfinite(arr)).sum())} non-finite entries"
        info[k] = {"shape": list(arr.shape), "dtype": "float64",
                   "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
                   "mean": float(np.mean(arr)) if arr.size else None,
                   "min": float(np.min(arr)) if arr.size else None,
                   "max": float(np.max(arr)) if arr.size else None}
    for k in ITEM_PARAM_KEYS:
        if k not in jt:
            problems[k] = "missing from fit output"
            continue
        arr = np.asarray(jt[k], dtype=np.float64)
        if arr.shape != (n_items,):
            problems[k] = f"shape {arr.shape} != ({n_items},)"
        elif not np.isfinite(arr).all():
            problems[k] = f"{int((~np.isfinite(arr)).sum())} non-finite entries"
        info[k] = {"shape": list(arr.shape), "dtype": "float64",
                   "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
                   "mean": float(np.mean(arr)) if arr.size else None,
                   "min": float(np.min(arr)) if arr.size else None,
                   "max": float(np.max(arr)) if arr.size else None}
    loss = np.asarray(jt.get("loss_history", []), dtype=np.float64)
    if loss.size == 0:
        problems["loss_history"] = "empty"
    elif not np.isfinite(loss).all():
        problems["loss_history"] = f"{int((~np.isfinite(loss)).sum())} non-finite entries"
    info["loss_history"] = {"shape": list(loss.shape), "dtype": "float64",
                            "sha256": hashlib.sha256(np.ascontiguousarray(loss).tobytes()).hexdigest(),
                            "first": float(loss[0]) if loss.size else None,
                            "last": float(loss[-1]) if loss.size else None,
                            "min": float(np.min(loss)) if loss.size else None}
    if problems:
        raise RunnerError(f"fit output is not finite/valid: {problems} - outputs kept for inspection")
    return {"all_finite": True, "n_models": n_models, "n_items": n_items, "arrays": info}


def run(bench: str, matrix_path: str, out_npz: str, out_json: str, *, pilot: bool = False,
        epochs: int | None = None, input_contract: str | None = None, device: str = "cuda",
        verbose: bool = False, validate_only: bool = False) -> dict:
    if bench not in BENCH_CONFIG:
        raise RunnerError(f"bench must be one of {tuple(BENCH_CONFIG)}, got {bench!r}")
    cfg = BENCH_CONFIG[bench]
    n_epochs = int(epochs if epochs is not None else (PILOT_MAX_EPOCHS if pilot else cfg["n_epochs"]))
    is_pilot = bool(pilot)
    if is_pilot and n_epochs > PILOT_MAX_EPOCHS:
        raise RunnerError(f"--pilot is capped at {PILOT_MAX_EPOCHS} epochs, got {n_epochs}")
    if not is_pilot and n_epochs != cfg["n_epochs"]:
        raise RunnerError(f"full run epochs must be {cfg['n_epochs']} (got {n_epochs}) - "
                          "any shorter fit must be labelled --pilot with separate outputs")
    tag_present = "pilot" in os.path.basename(out_npz).lower()
    if is_pilot != tag_present or ("pilot" in os.path.basename(out_json).lower()) != is_pilot:
        raise RunnerError("pilot outputs must be named *pilot*; full outputs must not be "
                          "(keeps a 2500-epoch artifact from ever masquerading as a pilot or vice versa)")
    for p in (out_npz, out_json):
        if os.path.exists(p):
            raise RunnerError(f"refusing to overwrite existing output: {p}")
    if device == "cpu":
        if not is_pilot or n_epochs > 3:
            raise RunnerError("CPU device is bounded to --pilot with <= 3 epochs (mechanics only); "
                              "the 2500-epoch joint fit is CUDA-only")
    import torch  # noqa: PLC0415  (import after path/env gates so gates fail without touching CUDA)
    if device != "cpu" and not torch.cuda.is_available():
        raise RunnerError("CUDA requested but torch.cuda.is_available() is False")
    inputs = derive_inputs(matrix_path)
    appr = None
    if is_pilot:
        appr = {"status": "PILOT_ONLY"}
    else:
        appr = {"status": "INPUTS_VALIDATED", **check_input_contract(input_contract, bench=bench,
                                                       matrix_sha256=inputs["matrix_sha256"],
                                                       epochs=n_epochs)}
    job = {
        "bench": bench, "mode": "pilot" if is_pilot else "full", "panel": "full analysis panel",
        "n_models": len(inputs["models"]), "n_items": len(inputs["items"]),
        "matrix": os.path.abspath(matrix_path), "matrix_sha256": inputs["matrix_sha256"],
        "matrix_array_sha256": inputs["array_sha256"], "matrix_shape_items_models": inputs["shape"],
        "config": {"n_epochs": n_epochs, "lr": cfg["lr"], "grad_clip": cfg["grad_clip"],
                   **FIT_DEFAULTS, "device": device, "response": "exp(LP_mean).clip(1e-6,1-1e-6)",
                   "Lt": "nanmedian(LP_sum/LP_mean, axis=1) centered by nanmean",
                   "channel": "joint only: C=BIN (Eq1), Y=RESP (Eq2); no IRSL (A) beta fits run",
                   "torch": torch.__version__, "python": platform.python_version(),
                   "torch_num_threads": torch.get_num_threads(),
                   "cuda_device": torch.cuda.get_device_name(0) if device != "cpu" else None},
        "input_masks": {"n1_BIN_observed": inputs["n1"], "n2_RESP_observed": inputs["n2"],
                        "NaN_preserved": True},
        "input_validation": appr, "original_modules": {"joint_irt.py": sha256_file(JOINT_PY),
                                              "beta_irt.py": sha256_file(BETA_PY)},
        "out_npz": os.path.abspath(out_npz), "out_json": os.path.abspath(out_json),
    }
    if is_pilot:
        job["PILOT_ONLY"] = "<=5 epochs for mechanics/timing only; NOT a 2500-epoch result and never reusable as one"
    if validate_only:
        job["status"] = "VALIDATED_NO_FIT"
        json.dump(job, open(out_json, "w", encoding="utf-8"), indent=1, sort_keys=True, default=str)
        return job
    joint_irt, _ = load_original_fit()
    torch.manual_seed(FIT_DEFAULTS["seed"])
    np.random.seed(FIT_DEFAULTS["seed"])
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    jt = joint_irt.fit_joint(inputs["BIN"].T, inputs["RESP"].T, inputs["Lt"], n_epochs=n_epochs,
                             lr=cfg["lr"], grad_clip=cfg["grad_clip"], phi=FIT_DEFAULTS["phi"],
                             clamp_eps=FIT_DEFAULTS["clamp_eps"], device=device,
                             seed=FIT_DEFAULTS["seed"], verbose=verbose)
    wall = time.time() - t0
    # Full numeric loss trajectory: persisted in the result NPZ (float64, one value per epoch).
    loss = [float(x) for x in jt["loss_history"]]
    loss_history = np.asarray(loss, dtype=np.float64)
    vectors = dict(inputs["vectors"])
    vectors.update({"theta_A": jt["theta_A"], "theta_C": jt["theta_C"], "gamma_m": jt["gamma_m"],
                    "a": jt["a"], "b": jt["b"], "alpha": jt["alpha"], "z": jt["z"],
                    "mode": np.array(job["mode"]), "fit_epochs": np.array(n_epochs)})
    np.savez(out_npz, **{k: vectors[k] for k in JOINT_KEYS}, mode=vectors["mode"],
             fit_epochs=vectors["fit_epochs"], loss_history=loss_history)
    # Every fit-returned array must be finite and correctly shaped (ALSO the item parameters a/b/
    # alpha/z and the full loss trajectory); a bad array leaves the outputs on disk and fails loudly.
    out_finite = enforce_output_finiteness(jt, n_models=len(inputs["models"]),
                                           n_items=len(inputs["items"]))
    job.update({
        "status": "PILOT_COMPLETED" if is_pilot else "FULL_COMPLETED",
        "out_npz_sha256": sha256_file(out_npz),
        "npz_keys": list(JOINT_KEYS) + ["mode", "fit_epochs", "loss_history"],
        "item_parameters": {k: out_finite["arrays"][k] for k in ITEM_PARAM_KEYS},
        "model_latents": {k: out_finite["arrays"][k] for k in MODEL_PARAM_KEYS},
        "output_finiteness": {"all_finite": out_finite["all_finite"],
                              "loss_history_n": int(loss_history.size)},
        "loss": {"first": loss[0], "last": loss[-1], "min": float(np.min(loss)),
                 "history_sha256": hashlib.sha256(loss_history.tobytes()).hexdigest(),
                 "n_recorded": len(loss), "npz_key": "loss_history", "npz_dtype": "float64",
                 "npz_shape": list(loss_history.shape)},
        "timing": {"wall_s": round(wall, 3), "s_per_epoch": round(wall / max(n_epochs, 1), 6),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                   "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "peak_memory": {"cuda_max_allocated_bytes": (int(torch.cuda.max_memory_allocated())
                                                     if device != "cpu" else None),
                        "cuda_max_reserved_bytes": (int(torch.cuda.max_memory_reserved())
                                                    if device != "cpu" else None),
                        "cuda_total_bytes": (int(torch.cuda.get_device_properties(0).total_memory)
                                             if device != "cpu" else None),
                        "process_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
        "vectors_sanity": {"theta_A_mean": float(np.mean(jt["theta_A"])),
                           "theta_C_mean": float(np.mean(jt["theta_C"])),
                           "gamma_m_mean": float(np.mean(jt["gamma_m"])),
                           "a_mean": float(np.mean(jt["a"])), "b_mean": float(np.mean(jt["b"])),
                           "alpha_mean": float(np.mean(jt["alpha"])),
                           "z_mean": float(np.mean(jt["z"])),
                           "all_finite": bool(np.isfinite(jt["theta_A"]).all()
                                              and np.isfinite(jt["theta_C"]).all()
                                              and np.isfinite(jt["gamma_m"]).all()
                                              and all(np.isfinite(jt[k]).all()
                                                      for k in ITEM_PARAM_KEYS))},
    })
    json.dump(job, open(out_json, "w", encoding="utf-8"), indent=1, sort_keys=True, default=str)
    return job


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, choices=tuple(BENCH_CONFIG))
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out-npz", required=True, help="pilot: name must contain 'pilot'")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--pilot", action="store_true", help=f"<= {PILOT_MAX_EPOCHS} epochs, separate outputs")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--input-contract", default=None, help="public complete-panel input contract")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--validate-only", action="store_true", help="gate + input derivation, no fit")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    try:
        job = run(a.bench, a.matrix, a.out_npz, a.out_json, pilot=a.pilot, epochs=a.epochs,
                  input_contract=a.input_contract, device=a.device, verbose=a.verbose, validate_only=a.validate_only)
    except RunnerError as e:
        print(f"RunnerError: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: job[k] for k in ("status", "bench", "mode", "n_models", "n_items")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
