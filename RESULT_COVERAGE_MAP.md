# Result coverage map

Study evidence and public computational paths; numerical reference files are authoritative.

Scientific analyses and scoped public reproduction checks independently verified.

## T1 — Signed mean confidence–accuracy gap and mean-preserving construction

Algebra and possibility example; no claim of empirical prevalence or latent-construct absence.

Code: `historical_reproduction/code/check_algebra.py`, `analysis_assets/make_revision_figures.py`.

References: `confidence_risk_audit_20260914/theory/PROPOSITIONS.md`.

## T2 — Core source: 6706 candidate models, 1168 ARC items; 6701 evaluated models from 1330 organizations

Source quantities independently checked. Public raw arrays and all 50 feature tables reproduced exactly; the independent source audit checked 125 contexts and 55 reference sets without reading target labels.

Code: `core_reproduction/run_core.py`, `core_reproduction/corrected_mode.py`.

References: `core_reproduction/corrected_inputs/CORRECTED_CORE_ARRAYS.identity.json`, `core_reproduction/corrected_inputs/CORRECTED_FEATURE_IDENTITIES.json`, `confidence_comparison/risk_metrics_v1/evaluation_cohort.csv`.

## T3 — All 31 primary and 31 secondary comparisons; 190 predictive effects

The numerical results were independently checked. Between-easiness-bin confidence-bias variance meets the expanded-baseline criterion; easiness-weighted confident-error mass does not meet that criterion. All comparisons remain visible. Two bounded public final-fit checks reproduced held-out predictions exactly; these integration checks do not cover a complete nested search.

Code: `core_reproduction/run_core.py`, `core_reproduction/original_inputs/confidence_comparison/risk_metrics_v1/code/evaluate.py`.

References: `confidence_comparison/risk_metrics_v1/results/PRIMARY_RESULTS.json`, `confidence_comparison/risk_metrics_v1/results/SECONDARY_RESULTS.json`, `confidence_comparison/risk_metrics_v1/CANDIDATE_MATRIX.csv`.

## T4 — Descriptor availability and source repeatability

Source summaries from the analysis; reference-population and cohort dependent.

Code: `core_reproduction/original_inputs/confidence_comparison/risk_metrics_v1/code/assemble_features.py`.

References: `confidence_comparison/risk_metrics_v1/features/source_quality.json`.

## T5 — Descriptive model selection

Paired case-organization descriptive intervals; no binary decision-success threshold or measured deployment benefit.

Code: `core_reproduction/original_inputs/confidence_comparison/risk_metrics_v1/code/evaluate.py`.

References: `confidence_comparison/risk_metrics_v1/results/selection_results.json`.

## T6 — Exploratory slope and five-summary/expanded-summary comparisons

Analyses and their public runtime implementations independently verified. These are retrospective exploratory analyses with the documented task overlap.

Code: `historical_reproduction/`.

References: `reference_originals/corrected_historical/initial/REFERENCE_MANIFEST.json`, `reference_originals/corrected_historical/expanded/REFERENCE_MANIFEST.json`, `reference_originals/corrected_historical/slope_check2.json`.

## T7 — Full-panel joint item-response supporting analysis: 6717 ARC, 6720 HellaSwag, 6716 shared models

Full-panel supporting analysis. Full fits and statistics independently checked. Both public cold matrices reconstructed exactly; public bounded pilot/validation checks are separate from full fitting.

Code: `joint_reproduction/cold.py`, `joint_reproduction/analyze.py`.

References: `joint_reproduction/ARTIFACTS.json`, `joint_reproduction/input_contract/CONTRACT.json`, `joint_reproduction/reference_results/analyze_joint_report.json`.

## T8 — Standalone analysis figures and tables

Standalone figure and table renderers consume the numerical reference results. The asset checker validates generated outputs and their inputs; scientific-result verification is documented separately.

Code: `analysis_assets/make_revision_figures.py`, `analysis_assets/make_supplement_tables.py`, `analysis_assets/check_assets.py`, `tools/export_candidate_matrix.py`.

References: `analysis_assets/FIGURE_GENERATION.json`, `analysis_assets/corrected_cohort_metadata.csv`, `analysis_assets/supplement_results_table.tex`, `analysis_assets/supplement_stability_table.tex`, `analysis_assets/supplement_secondary_table.tex`, `analysis_assets/supplement_selection_table.tex`.

## T9 — Computational cost accounting

Measured event totals and all 9,900 warm measurements were independently checked. Complete acquisition-to-results cold time remains unmeasured; the disk count is a total result-tree snapshot, not a new-byte delta.

Code: `core_reproduction/original_inputs/confidence_comparison/risk_metrics_v1/code/cost_report.py`, `core_reproduction/original_inputs/confidence_comparison/risk_metrics_v1/code/warm_costs.py`.

References: `confidence_comparison/risk_metrics_v1/results/cost_report.json`, `confidence_comparison/risk_metrics_v1/cost/warm_summary.json`.
