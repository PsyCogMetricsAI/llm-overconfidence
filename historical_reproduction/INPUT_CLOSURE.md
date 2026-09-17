# Analysis inputs and computations

| Reported branch | Computational input | Computation |
|---|---|---|
| Slope Pearson/Spearman and uploader jackknife | Raw item records after exclusions, immutable uploader mapping | Primary and secondary ALS fits for both benchmarks; paired log-slope statistics |
| Five-summary prediction | Raw/source arrays on the fixed 6,706 candidate IDs, item halves after exclusions | Reference fits, explicit-decimal-edge target scorer, nested ridge, 97.5% bootstrap |
| Nine-summary prediction | Freshly computed conventional-summary references, qualification and floor-bin targets | Four-package matched ridge/boosting searches, sealed predictions, 95% bootstrap and selection |
| Supporting algebra | Symbolic identities and constructed examples | Algebra/numerical checks; no empirical training |

Source qualification yields 6,701 models and 1,330 organizations for both prediction studies. Comparison-only canonical identities and result JSONs are bundled separately from executable computation. The checker validates feature identities, membership, targets, phase evidence and artifacts. Each run requires its own source, prediction and result checks.

The full-panel joint analysis uses its separate public workflow. Original joint-model code is retained as provenance.
