# Supporting analyses

This directory contains scientific code and portable orchestration for the slope, five-summary and nine-summary analyses in Supplement S6. Each branch has its specified formulas and scoring conventions.

Follow [the supporting-comparison workflow](../docs/06_HISTORICAL_REPRODUCTION.md). It starts from raw data with the specified item exclusions and freshly computed shared core dependencies, runs fits, seals predictions, and checks final statistics. The five-summary protocol is bundled with the runtime.

Comparison-only references are in `../reference_originals/corrected_historical/`. In public cold validation, both prediction studies matched their reference predictions and final statistics. The slope analysis matched all reported raw point estimates and intervals.

The supporting algebra can be checked with `python historical_reproduction/code/check_algebra.py`. It is a theory calculation, not an empirical fit. The full-panel joint item-response model uses its separate public runtime.
