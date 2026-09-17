"""Independent V-SEAL audit.

This verifier is deliberately dormant until PREDICTION_SEAL.json exists.  It
never opens targets/targets.csv and does not import the training implementation
or its selection helpers; all package and tie decisions are recomputed here.
"""
from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import digest, readjson, rows, sethash  # noqa: E402


PLAN_SHA = "PUBLIC_PROTOCOL_SHA256"
TOL = 1e-12


def fail(check, detail):
    raise AssertionError(f"{check}: {detail}")


def tie_key(c):
    # Frozen registry tie order: ridge first, then larger lambda; HGB uses
    # fewer leaves, larger L2, and smaller learning rate.
    if c["family"] == "ridge":
        return (0, -float(c["lambda"]))
    return (1, int(c["max_leaf_nodes"]), -float(c["l2_regularization"]),
            float(c["learning_rate"]))


def choose(configs, losses, eligible=None):
    ids = list(range(len(configs)))
    if eligible is not None:
        ids = [i for i in ids if eligible[i]]
    if not ids:
        fail("selection", "empty eligible configuration set")
    m = min(float(losses[i]) for i in ids)
    tied = [i for i in ids if float(losses[i]) <= m + TOL]
    return min(tied, key=lambda i: tie_key(configs[i]))


def expected_columns(pr):
    # train.outer writes each package endpoint, then its three algorithms; the
    # field names themselves are group_algorithm_endpoint.
    return ([f"{g}_{a}_{e}" for g in pr["feature_groups"]
             for e in pr["endpoints"] for a in ("ridge", "hgb", "best")]
            + [f"{g}_best_{e}" for g in pr["procedures"]
               for e in pr["endpoints"]])


def check_outer_rows(pr, seal):
    cols = expected_columns(pr)
    if len(cols) != 190 or len(set(cols)) != 190:
        fail("columns", f"registry expands to {len(cols)} columns")
    cohort = rows(ROOT / "evaluation_cohort.csv")
    expected = {r["model_id"]: r["org_id"] for r in cohort}
    all_rows, outer_rows = [], {}
    for k in range(5):
        p = ROOT / "predictions" / f"outer_{k}.csv"
        if not p.exists(): fail("outer_rows", f"missing {p.name}")
        rr = rows(p)
        with p.open(newline="") as fh:
            fields = next(csv.reader(fh))
        expected_fields = ["model_id", "org_id", "outer_fold", "target"] + cols
        if fields != expected_fields:
            fail("columns", f"outer {k} header set mismatch")
        if not rr: fail("outer_rows", f"empty outer {k}")
        outer_rows[k] = rr; all_rows.extend(rr)
        frozen_test = readjson(ROOT / "cohort_contract.json")["fixed_rows"][f"features/outer_{k}/test.csv"]["model_ids"]
        if [r["model_id"] for r in rr] != frozen_test:
            fail("outer_rows", f"outer {k} test IDs differ from frozen contract")
        for r in rr:
            if int(r["outer_fold"]) != k: fail("outer_rows", "fold label mismatch")
            if any(not np.isfinite(float(r[c])) or not 0 <= float(r[c]) <= 1 for c in cols):
                fail("outer_rows", f"nonfinite/out-of-range prediction {r['model_id']}")
    got = {r["model_id"]: r["org_id"] for r in all_rows}
    if len(all_rows) != 6666 or len(got) != len(all_rows) or got != expected:
        fail("cohort", f"rows={len(all_rows)} unique={len(got)} expected=6666")
    if int(seal["n_models"]) != 6666 or int(seal["n_orgs"]) != 1324:
        fail("seal_counts", repr({k: seal.get(k) for k in ("n_models", "n_orgs")}))
    return cols, expected, outer_rows, all_rows


def check_selection(pr, outer_rows):
    configs = pr["candidate_configurations"]
    groups, endpoints = pr["feature_groups"], pr["endpoints"]
    order = list(groups)
    for k in range(5):
        log = readjson(ROOT / "predictions" / f"outer_{k}_selection.json")
        if len(log.get("packages", [])) != 62 or len(log.get("procedures", [])) != 4:
            fail("selection_shape", f"outer {k}")
        packages = {(x["group"], x["endpoint"]): x for x in log["packages"]}
        if len(packages) != 62: fail("selection_shape", f"duplicate package outer {k}")
        for (g, e), x in packages.items():
            losses = np.asarray(x["inner_losses"], float)
            if losses.shape != (17,) or not np.isfinite(losses).all():
                fail("inner_losses", f"{k}/{g}/{e}")
            fam = {"ridge": [i for i,c in enumerate(configs) if c["family"] == "ridge"],
                   "hgb": [i for i,c in enumerate(configs) if c["family"] == "hgb"]}
            for f, ix in fam.items():
                # choose() returns an index local to this family; translate it
                # back to the frozen registry index.
                want = ix[choose([configs[i] for i in ix], losses[ix])]
                if int(x["family_selected_indices"][f]) != want:
                    fail("family_selection", f"{k}/{g}/{e}/{f}")
            want = choose(configs, losses)
            if int(x["best_index"]) != want or x["best_configuration"] != configs[want]["id"]:
                fail("global_selection", f"{k}/{g}/{e}")
            if abs(float(x["selected_actual_inner_loss"]) - losses[want]) > 1e-10:
                fail("selected_loss", f"{k}/{g}/{e}")
            # A global-best final fit is needed exactly when it differs from
            # the winning configuration of its own family.
            famidx = int(x["family_selected_indices"][configs[want]["family"]])
            if bool(x["deployment_extra_fit"]) != (famidx != want):
                fail("deployment_fit", f"{k}/{g}/{e}")
            for ent in list(x.get("estimators", {}).values()) + [x["best_estimator"]]:
                p = ROOT / ent["path"]
                if not p.exists() or digest(p) != ent["sha256"]:
                    fail("estimator_hash", ent.get("path"))
        # Independently recompute source eligibility and package tie choice.
        missing = log.get("missing_contexts", [])
        if len(missing) != 5:
            fail("missing_contexts", f"outer {k}: {len(missing)}")
        metric_ids = list(pr["metric_ids"])
        direct = {}
        for entry in missing:
            context = entry["context"]
            if context.endswith("final_train"):
                fn = "train.csv"
            else:
                bits = context.split("/")
                if len(bits) != 3 or not bits[1].startswith("inner"):
                    fail("missing_contexts", f"invalid context {context}")
                if bits[2] not in ("train", "validation"):
                    fail("missing_contexts", f"invalid split {context}")
                fn = f"inner{bits[1][5:]}_" + ("train.csv" if bits[2] == "train" else "valid.csv")
            data_rows = rows(ROOT / "features" / f"outer_{k}" / fn)
            direct[context] = {m: not any(np.isfinite(float(r[m])) for r in data_rows) for m in metric_ids}
            if entry.get("all_missing") != direct[context]:
                fail("missing_contexts", f"logged mask differs from source rows {context}")
        expected_eval = {m: not any(direct[x["context"]][m] for x in missing)
                         for m in pr["metric_ids"]}
        if log.get("source_evaluable") != expected_eval:
            fail("source_evaluable", f"outer {k}")
        expected_joint = not any(all(bool(x.get("all_missing", {}).get(m, False)) for m in pr["structure_ids"])
                                 for x in missing)
        if bool(log.get("joint_evaluable")) != expected_joint:
            fail("joint_evaluable", f"outer {k}")
        # Procedures must use the selected package's already-sealed best column.
        byid = {r["model_id"]: r for r in outer_rows[k]}
        for q in log["procedures"]:
            g, e = q["chosen_package"], q["endpoint"]
            if not q.get("no_additional_fit", False): fail("procedure_fit", f"outer {k}")
            if g not in groups or e not in endpoints: fail("procedure_key", repr(q))
            names = list(pr["procedures"][q["procedure"]])
            eligible = {n: (n in ("B0", "Bd") or expected_eval[n[2:]]) for n in names}
            actual = {n: float(packages[n, e]["selected_actual_inner_loss"]) for n in names}
            want_pkg = min((n for n in names if eligible[n]), key=lambda n: (actual[n], names.index(n)))
            # Explicit epsilon tie handling, with baseline then registry order.
            mn = min(actual[n] for n in names if eligible[n])
            want_pkg = next(n for n in names if eligible[n] and actual[n] <= mn + TOL)
            if g != want_pkg or q.get("eligible_packages") != eligible or q.get("package_actual_inner_losses") != actual:
                fail("procedure_selection", f"outer {k}/{q['procedure']}/{e}")
            if q["best_estimator"] != packages[g, e]["best_estimator"]:
                fail("procedure_estimator_reuse", f"{k}/{q['procedure']}/{e}")
            for r in byid.values():
                if abs(float(r[f'{q["procedure"]}_best_{e}']) - float(r[f'{g}_best_{e}'])) > 1e-12:
                    fail("procedure_prediction_reuse", f"{k}/{q['procedure']}/{e}")


def check_recommendations(pr, all_rows):
    source = {r["model_id"]: r for r in rows(ROOT / "evaluation_cohort.csv")}
    byorg = {}
    for r in all_rows: byorg.setdefault(r["org_id"], []).append(r)
    methods = list(pr["feature_groups"]) + list(pr["procedures"])
    want_c, want_r = [], []
    for org, rr in sorted(byorg.items()):
        top = max(float(source[r["model_id"]]["A"]) for r in rr)
        cand = sorted([r for r in rr if float(source[r["model_id"]]["A"]) >= top - .02], key=lambda r:r["model_id"])
        if len(cand) < 3: continue
        for r in cand: want_c.append((org,r["model_id"],int(r["outer_fold"]),float(source[r["model_id"]]["A"]),len(cand)))
        for m in methods:
            b = min(cand, key=lambda r:(float(r[f"{m}_best_Y_sel"]), r["model_id"]))
            want_r.append((org,int(b["outer_fold"]),m,b["model_id"],float(b[f"{m}_best_Y_sel"]),len(cand)))
    c = rows(ROOT/"candidates.csv"); r = rows(ROOT/"recommendations.csv")
    gotc = [(x["org_id"],x["model_id"],int(x["outer_fold"]),float(x["source_A"]),int(x["n_candidates"])) for x in c]
    gotr = [(x["org_id"],int(x["outer_fold"]),x["method"],x["model_id"],float(x["predicted_R50"]),int(x["n_candidates"])) for x in r]
    if gotc != want_c or gotr != want_r or len({x["org_id"] for x in c}) != int(readjson(ROOT/"PREDICTION_SEAL.json")["n_case_orgs"]):
        fail("recommendations", f"candidate rows {len(c)} / recommendation rows {len(r)}")
    if len(methods) != 33: fail("recommendation_methods", len(methods))


def check_access(seal):
    records = []
    for p in sorted((ROOT/"targets").glob("access_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip(): records.append(json.loads(line))
    training = [x for x in records if x.get("kind") == "training"]
    if len(training) < 45: fail("target_access", f"training records={len(training)}")
    if any(x.get("kind") != "training" for x in records):
        fail("target_access", "non-training access before V-SEAL")
    expected = set()
    for k in range(5):
        expected.add((k,f"outer{k}/final_train"))
        expected.update((k,f"outer{k}/inner{j}/{s}") for j in range(4) for s in ("train","validation"))
    got = {(int(x["outer_fold"]),x["context"]) for x in training}
    if got != expected: fail("target_scope", f"contexts missing={expected-got} extra={got-expected}")
    for x in training:
        k,ctx=int(x["outer_fold"]),x["context"]
        fn = "train.csv" if ctx.endswith("final_train") else (f"inner{ctx.split('/')[1][5:]}_" + ("train.csv" if ctx.endswith("/train") else "valid.csv"))
        fixed = readjson(ROOT/"cohort_contract.json")["fixed_rows"][f"features/outer_{k}/{fn}"]
        if x["models_sha256"] != sethash(fixed["model_ids"]) or x["orgs_sha256"] != sethash(fixed["org_ids"]):
            fail("target_scope", f"hash mismatch {k}/{ctx}")
    return {"n_training_records": len(training),
            "n_unique_training_contexts": len(got),
            "duplicate_attempts": len(training) - len(got)}


def main():
    seal_path = ROOT / "PREDICTION_SEAL.json"
    if not seal_path.exists():
        print("V-SEAL_NOT_READY: PREDICTION_SEAL.json absent")
        return 2
    seal = readjson(seal_path); pr = readjson(ROOT/"protocol.json")
    if digest(ROOT/"protocol.json") and digest(Path(pr["plan_path"])) != PLAN_SHA:
        fail("plan", "plan SHA mismatch")
    for n,h in seal.get("artifacts", {}).items():
        if digest(ROOT/n) != h: fail("seal_artifact", n)
    with (ROOT / "predictions_oof.csv").open(newline="") as fh:
        if next(csv.reader(fh)) != ["model_id", "org_id", "outer_fold", "target"] + expected_columns(pr):
            fail("columns", "predictions_oof header mismatch")
    for key,path in (("CODE_RELEASE_sha256","CODE_RELEASE.json"),("source_release_sha256","SOURCE_RELEASE.json"),("E1_bundle_sha256","E1_BUNDLE.json")):
        if seal.get(key) != digest(ROOT/path): fail("seal_binding", key)
    code_release = readjson(ROOT/"CODE_RELEASE.json")
    source_release = readjson(ROOT/"SOURCE_RELEASE.json")
    if code_release.get("E1_bundle_sha256") != digest(ROOT/"E1_BUNDLE.json"):
        fail("input_binding", "CODE_RELEASE does not bind E1 bundle")
    if source_release.get("features_ready_sha256") != digest(ROOT/"features/FEATURES_READY.json"):
        fail("input_binding", "SOURCE_RELEASE does not bind FEATURES_READY")
    cols, expected, outer, all_rows = check_outer_rows(pr, seal)
    check_selection(pr, outer)
    check_recommendations(pr, all_rows)
    access = check_access(seal)
    report = {"verdict":"ACCEPT","node":"V-SEAL","independent":True,"target_values_read":False,
              "checks":{"190_columns":True,"cohort_6666_1324":True,"selection_and_reuse":True,
                         "recommendations_33":True,"target_access_45_scoped":True},
              "target_access":access,
              "seal_sha256":digest(seal_path),"artifacts":seal["artifacts"],
              "code_release_sha256":seal["CODE_RELEASE_sha256"],"source_release_sha256":seal["source_release_sha256"],
              "E1_bundle_sha256":seal["E1_bundle_sha256"],
              "code_sha256":code_release.get("code_sha256", {}),
              "features_ready_sha256":source_release.get("features_ready_sha256")}
    out=ROOT/"reviews/SEAL_REVIEW.json";out.write_text(json.dumps(report,indent=2)+"\n")
    (ROOT/"reviews/SEAL_REVIEW.md").write_text("# V-SEAL Independent Review\n\nVerdict: **ACCEPT**. Target values were not read.\n\n"+json.dumps(report["checks"],indent=2)+"\n")
    print(json.dumps(report,indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
