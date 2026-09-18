# Recompute the study from item-level responses

Run from the repository root. Use new output directories: the pipeline stops rather than overwriting completed runs or accepting mismatched inputs. The main analysis and the supporting analyses use the same documented item-identity exclusions.

## 1. Set up the environments

Use Python 3.12 for the core and historical prediction pipelines:

```bash
python3 -m venv .venv-core
.venv-core/bin/python -m pip install -r environment/requirements-core.lock
.venv-core/bin/python -m pip check
.venv-core/bin/python tools/check_required_inputs.py --mode public
```

Acquisition and joint fitting have separate dependency specifications in `environment/`; see [environment details](05_ENVIRONMENT.md). The joint full fits use a CUDA GPU and the documented 2,500-epoch settings. Core regressions use single-threaded numerical libraries, which the driver sets for its subprocesses.

## 2. Retrieve and verify responses

In the acquisition environment:

```bash
python tools/reharvest_peritem.py --bench arc --include-supplemental \
  --write-raw "$PWD/runs/acquired" --out "$PWD/runs/arc-acquisition-log" \
  --cache-dir /tmp/llm-overconfidence-hf-cache
python tools/reharvest_peritem.py --bench hellaswag \
  --write-raw "$PWD/runs/acquired" --out "$PWD/runs/hs-acquisition-log" \
  --cache-dir /tmp/llm-overconfidence-hf-cache
python tools/make_corrected_raw_view.py --raw "$PWD/runs/acquired" \
  --input-identity semantic --out "$PWD/runs/corrected-core"
```

The downloader reconstructs all 13,494 core records plus the two additional ARC records for the broader joint analysis. Every array and scalar is checked against the bundled semantic fingerprints before output is written. The filtering step creates the 13,494-file core view by excluding the specified ambiguous item IDs. The joint builder consumes the acquired records directly and applies the same exclusions itself.

A small acquisition check can select individual files with `--npz`, but a sample is insufficient for the complete analysis. Source versions, parquet paths, retrieval outcomes, and checksums are recorded. Missing or changed sources produce errors; there is no automatic substitution of a different revision. Raw responses are downloaded separately and are not part of the small release attachment.

## 3. Generate source features and fit the main comparison

```bash
CORE_WORK="$PWD/runs/reproduction/core"
CORE_RAW="$PWD/runs/corrected-core"
CORE_EXP="$CORE_WORK/confidence_comparison/risk_metrics_v1"

.venv-core/bin/python core_reproduction/run_core.py raw \
  --public-mode --corrected --raw "$CORE_RAW" --work "$CORE_WORK"
.venv-core/bin/python core_reproduction/run_core.py legacy \
  --public-mode --corrected --raw "$CORE_RAW" --work "$CORE_WORK"
.venv-core/bin/python core_reproduction/run_core.py source \
  --public-mode --corrected --raw "$CORE_RAW" --work "$CORE_WORK"
.venv-core/bin/python independent_verification/release_core.py source \
  --public-mode --core "$CORE_EXP" --migration-root core_reproduction
.venv-core/bin/python core_reproduction/run_core.py train \
  --public-mode --corrected --raw "$CORE_RAW" --work "$CORE_WORK"
.venv-core/bin/python independent_verification/release_core.py seal \
  --public-mode --core "$CORE_EXP" --migration-root core_reproduction
.venv-core/bin/python core_reproduction/run_core.py evaluate \
  --public-mode --corrected --raw "$CORE_RAW" --work "$CORE_WORK"
.venv-core/bin/python tools/export_candidate_matrix.py --experiment "$CORE_EXP"
```

The nested work path is intentional: an archived module evaluates an ancestor-path expression, so very shallow work directories such as `/tmp/work` are unsuitable.

The raw stage reconstructs the fixed candidate model axis and the item halves after exclusions. The legacy stage recomputes the conventional summaries, reference fits, and context-specific eligibility. The source stage constructs the complete indicator library. The independent source check recomputes source quantities, checks all 50 feature tables against the reference identities, and verifies that target labels were not used in source construction.

Training uses five outer organization folds and four inner folds, followed by prediction sealing. The seal check verifies held-out membership, configuration selection, prediction identities, and permitted target access before scoring. In public mode the runtime copy of the archived seal verifier is bound to the public protocol hash and to the documented public cohort (6,701 models from 1,330 organizations; the archived file hard-codes the original private cohort of 6,666 / 1,324); both substitutions are recorded in `reviews/PLAN_ADAPTER.json` and the archived verifier itself is left byte-identical. Evaluation computes all 31 primary comparisons with simultaneous intervals, 31 secondary contrasts, descriptive model selection, and the recorded computation-cost measurements. The full fit count for the reference run is 21,080 inner fits and 620 final fits; configurations are fixed in the bundled protocol.

These commands use portable input identities and the public scientific protocol. Hash-bound numerical checks remain active, including checks of the declared path adaptations. A failed check stops the affected branch; correct its upstream cause before continuing.

## 4. Recompute supporting analyses

The supporting slope, five-summary and expanded-summary analyses are documented in [the supporting-comparison reproduction guide](06_HISTORICAL_REPRODUCTION.md). They reuse the freshly generated source inputs where appropriate and have their own source, prediction, and result checks. Their interval conventions remain distinct from the main 31-comparison simultaneous intervals.

For the full-panel joint item-response analysis, use the joint environment:

```bash
python joint_reproduction/cold.py build --bench arc \
  --raw-root "$PWD/runs/acquired" --out "$PWD/runs/joint/arc-input"
python joint_reproduction/cold.py build --bench hellaswag \
  --raw-root "$PWD/runs/acquired" --out "$PWD/runs/joint/hs-input"
python joint_reproduction/cold.py fit --bench arc \
  --matrix "$PWD/runs/joint/arc-input/matrix_arc.npz" --out "$PWD/runs/joint/arc-fit"
python joint_reproduction/cold.py fit --bench hellaswag \
  --matrix "$PWD/runs/joint/hs-input/matrix_hellaswag.npz" --out "$PWD/runs/joint/hs-fit"
python joint_reproduction/analyze.py \
  --arc-fit "$PWD/runs/joint/arc-fit" --hellaswag-fit "$PWD/runs/joint/hs-fit" \
  --out "$PWD/runs/joint/statistics"
```

The builder checks the complete eligible populations and matrix identities. Fitting uses the specified mathematical model and fixed settings. `--validate-only` checks fit inputs without optimization; `--pilot` runs a short resource check whose outputs cannot supply the full-fit statistics. The analyzer's supplied-fit mode consumes the new fitted vectors. Omitting both fit arguments instead recomputes statistics from the bundled reference fits; these are separate reproduction paths. See [joint-analysis details](../joint_reproduction/README.md).

## 5. Generate analysis assets from the new results

Follow the [figure and table guide](02_QUICKSTART_ANALYSIS_ASSETS.md), passing `--experiment "$CORE_EXP"` instead of the bundled result directory. The reference tables provide comparison targets; freshly computed outputs remain in the new run tree. Input masks, cohort scope, seeds, estimators, comparison families, and support rules must stay fixed when reproducing the reported analysis.
