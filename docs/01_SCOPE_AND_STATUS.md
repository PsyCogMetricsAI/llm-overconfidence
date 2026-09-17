# Scope of the reproducibility materials

The repository contains original code under the MIT license, dependency locks, reference results, standalone figures and table snippets, data fingerprints, and retrieval tools. Large item-level response files are obtained separately from public Open LLM Leaderboard v1 detail repositories.

## Analyses covered

| Analysis | Role in the study | Reproduction path |
|---|---|---|
| Mean confidence bias and constructed example | Establish what average confidence minus accuracy retains and omits | Bundled analytical checks and figure generator |
| Twenty source descriptors and 31 comparisons | Evaluate additional prediction information for HellaSwag model-level risk | `core_reproduction/` |
| Source-half coverage and repeatability | Describe availability and within-source stability of the indicators | Source-feature diagnostics in the core workflow |
| Model selection | Examine whether prediction gains improve the realized risk of selected models | Descriptive analysis in the core evaluation |
| Response-slope correlation | Supporting cross-benchmark stability analysis | `historical_reproduction/` |
| Five-summary and expanded-summary prediction designs | Supporting exploratory comparisons | `historical_reproduction/` |
| Full-panel joint item-response model | Supporting latent-parameter and gauge analysis | `joint_reproduction/` |

The supervised core uses a fixed 6,706-model candidate universe and organization splits. Qualification yields 6,701 evaluated models from 1,330 organizations. The broader joint analysis uses 6,717 ARC models and 6,720 HellaSwag models, with 6,716 in common; it is a distinct population.

## Data identity and acquisition

The fixed collection inventory contains 7,055 candidate repositories. Collection was attempted for both benchmarks, yielding 6,748 records per benchmark; the remaining cases are accounted for in the acquisition records. The core archive has 6,746 ARC files and 6,748 HellaSwag files; two additional ARC records enter the broader joint analysis.

The upstream question/option mapping contains ambiguous canonical item identities. The analysis excludes all 4 specified ARC item IDs and 22 specified HellaSwag item IDs under an outcome-independent rule. Remaining response values are retained unchanged. The item split uses the documented hash rule after these exclusions. The exclusion list, source identities, and numerical checks are supplied with the data-processing code.

The downloader uses pinned source revisions and verifies every reconstructed response array and metadata field against the recorded fingerprints. Missing repositories, unavailable paths, and mismatched values are reported explicitly. A revision pin identifies a retrieval source; it does not guarantee its continued availability or grant redistribution rights.

## Interpretation

The study is retrospective: both benchmarks overlap earlier exploratory analyses. Organization-held-out prediction tests transfer across uploading organizations, which are not verified independent model lineages. The core endpoint is each model's selective error rate, and prediction accuracy is measured by organization-weighted mean squared error. The joint model's raw latent parameters depend on its scale conventions, as documented in `joint_reproduction/README.md`.

The [quick-start guide](02_QUICKSTART_ANALYSIS_ASSETS.md) rebuilds standalone analysis assets from the reference results. The [full reproduction guide](03_FULL_COLD_REPRODUCTION.md) describes input retrieval, fresh computation, and independent checks. Reference outputs are comparisons for a rerun, not substitutes for its computed outputs.
