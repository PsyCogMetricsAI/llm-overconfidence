# Rebuild standalone analysis figures and tables

This path uses the bundled numerical results. It requires no raw response downloads or model fitting.

## Environment

Use Python 3.12 and run from the repository root:

```bash
python3 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r environment/requirements-figures.lock
.venv-figures/bin/python -m pip check
```

## Regenerate and check

Generate a separate set of assets from the bundled experiment results and model metadata:

```bash
.venv-figures/bin/python analysis_assets/make_revision_figures.py \
  --experiment confidence_comparison/risk_metrics_v1 \
  --metadata-source external_inputs/_model_type_labels.csv \
  --output-dir runs/analysis-assets
.venv-figures/bin/python analysis_assets/make_supplement_tables.py \
  --experiment confidence_comparison/risk_metrics_v1 \
  --output-dir runs/analysis-assets
.venv-figures/bin/python analysis_assets/check_assets.py \
  --experiment confidence_comparison/risk_metrics_v1 \
  --assets-dir runs/analysis-assets
```

The figure command writes the conceptual example and comparison plot as standalone PDF and PNG files, plus cohort metadata and generation provenance. The table command writes four `.tex` table snippets. Generating and checking snippet text requires no LaTeX installation; typesetting the snippets separately would require LaTeX.

Check the bundled table snippets and the complete bundled asset set separately:

```bash
.venv-figures/bin/python analysis_assets/make_supplement_tables.py \
  --experiment confidence_comparison/risk_metrics_v1 \
  --output-dir analysis_assets --check
.venv-figures/bin/python analysis_assets/check_assets.py \
  --experiment confidence_comparison/risk_metrics_v1 \
  --assets-dir analysis_assets
```

The asset checker regenerates into an isolated temporary directory. It verifies all 31 primary comparison transcriptions and classifications, compares four table snippets and two PNG figures byte-for-byte, and checks cohort metadata and provenance. PDF checks cover presence and file headers; timestamped PDF bytes are not compared. These checks complement the independent scientific checks in the [full reproduction workflow](03_FULL_COLD_REPRODUCTION.md).

To render a fresh computation, pass its experiment directory to each command and use a separate output directory for that run. Pass the output directory as `--assets-dir` to the checker. Keep the same model metadata and analysis definitions used for the reported results.
