# LLM Overconfidence Is Not a Measurable, Comparable Trait

Code, data-retrieval tools, reference results, figures, and table-generation tools for the paper by Yuehan Wang, Zhiye Jin, Xuequn Wang, and Yibai Li.

## Research question

Average confidence minus accuracy describes a model's mean confidence bias. It leaves unspecified which answers are wrong and where confident mistakes occur. We ask whether source-task descriptions of difficulty-related confidence and error structure help predict a model's risk on another benchmark after accounting for conventional performance, calibration, and error summaries.

## Study and findings

The main study uses ARC-Challenge as the source and HellaSwag as the target, with 6,701 models from 1,330 uploading organizations. It compares 20 source descriptors using nested organization-held-out regression. The target is each model's error rate among its most-confident half of answers. The regressors learn the source-to-target mapping from cached response records.

Between-easiness-bin variance of confidence bias reduces organization-weighted mean squared error in predicting model-level selective risk by **13.889%** relative to the expanded baseline and meets the predefined support rule with simultaneous intervals across all 31 comparisons. Easiness-weighted confident-error mass gives a **6.676%** point reduction in prediction mean squared error against that baseline, but its simultaneous interval includes zero. The separate model-selection analysis is inconclusive for decision benefit.

The evaluation is retrospective, and both benchmarks were examined during earlier exploratory work. The findings concern cross-benchmark model-level risk prediction.

## Repository structure

```text
.
├── analysis_assets/                 # Standalone figures, table snippets, and renderers
├── confidence_comparison/           # Main experiment's reference results
├── core_reproduction/               # Source descriptors, nested prediction, and statistical analysis
├── historical_reproduction/         # Supporting slope and summary-prediction analyses
├── joint_reproduction/              # Full-panel joint item-response analysis
├── independent_verification/        # Input, prediction, and result checks
├── upstream_metadata/               # Source revisions, item exclusions, and data fingerprints
├── reference_originals/             # Compact numerical references used by reproduction checks
├── external_inputs/                 # Model metadata
├── environment/                     # Dependency locks
├── references/                      # Bibliography and literature records
├── docs/                            # Reproduction instructions and data provenance
├── tools/                           # Data retrieval, processing, and package checks
└── manifest/                        # File sizes and checksums
```

## Reproduce the study

1. **Rebuild figures and tables:** install the figure dependencies and run the renderers against the bundled results. This needs no raw downloads or model fitting. See the [quick-start guide](docs/02_QUICKSTART_ANALYSIS_ASSETS.md).
2. **Recompute from responses:** retrieve the pinned item-level data, verify its fingerprints, apply the documented item-identity exclusions, and run the source-feature, prediction, and statistical stages. Independent checks separate these stages. See the [full reproduction guide](docs/03_FULL_COLD_REPRODUCTION.md).
3. **Recompute supporting analyses:** follow the supporting-comparison and joint-analysis instructions linked from that guide.

The repository contains code, reference results, and standalone analysis assets. Item-level responses are retrieved separately from their upstream repositories. Availability depends on those sources; the downloader reports missing or changed inputs instead of substituting them.

## Citation and licensing

Author and citation metadata are provided in [CITATION.cff](CITATION.cff).

Original code is released under the [MIT license](LICENSE). Upstream datasets and third-party material retain their own terms; see [license scope](LICENSE_SCOPE.md).
