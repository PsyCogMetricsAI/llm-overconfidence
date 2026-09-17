# Core experiment: reproduction from responses

`run_core.py` computes source descriptors, nested predictions and statistical comparisons in a fresh
work directory. Use acquired raw data with the specified item exclusions, the locked core environment,
and `--public-mode --corrected` on every phase. The pipeline checks input and output identities. Full commands are in `../docs/03_FULL_COLD_REPRODUCTION.md`.

The phases are:

1. `raw`: check all 13,494 analysis-input files, reconstruct the 6,706-candidate ARC arrays,
   apply the specified item-half rule on 1,168 retained items and compare every array with its
   reference identity.
2. `legacy`: reconstruct the nine conventional summaries, nested reference populations and
   source qualification; prepare HellaSwag targets. The evaluated cohort has 6,701 models from
   1,330 uploading organizations.
3. `source`: reconstruct all 20 candidate descriptors and 50 feature tables; stop for the
   independent source verifier, which checks formulas and exact reference values.
4. `train`: after source release, execute the fixed nested regression search and seal predictions.
5. `evaluate`: after independent seal release, compute the predefined comparisons, descriptive
   selection analysis and cost records, then run the statistical verifier.

For the two runtime verification steps:

```bash
python independent_verification/release_core.py source --core /data/work/confidence_comparison/risk_metrics_v1 --migration-root core_reproduction --public-mode
python independent_verification/release_core.py seal --core /data/work/confidence_comparison/risk_metrics_v1 --migration-root core_reproduction --public-mode
```

Run these commands from the repository root, after `source` and `train`, respectively. Each gate
checks the current work directory and binds fresh output hashes. The seal check verifies held-out membership, selected configurations, prediction identities and
permitted target access against the public protocol.

`--raw` and `--work` specify reader-controlled locations. A new raw phase requires a fresh work
directory; subsequent phases use that same directory. The environment is checked before raw
construction. CPU fitting uses the fixed single-threaded numerical settings and specified search.
The reproduction workflow uses the five phases listed above.

Numerical reference results are in `../confidence_comparison/risk_metrics_v1/`; they are
comparison evidence, never inputs to supervised fitting. After evaluation, create the paper's
31-row CSV view with `tools/export_candidate_matrix.py --experiment EXPERIMENT_DIRECTORY`.
