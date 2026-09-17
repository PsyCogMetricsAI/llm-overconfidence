# Supporting comparison analyses

These commands recompute the supporting response-slope, five-summary and expanded-summary analyses. All use the input view with ambiguous canonical item identifiers excluded. The five-summary and nine-summary studies use their specified formulas, searches, split rules and distinct interval conventions. Their source candidate set contains 6,706 models; qualification yields 6,701 evaluated models from 1,330 organizations. The slope analysis has its own broader raw-response population.

Install the pinned environment from `core_reproduction/requirements.txt`. First run the public **raw** and **legacy** phases described in `03_FULL_COLD_REPRODUCTION.md`. Set `SHARED` to that fresh work root and `RAW` to the per-item directory after exclusions. The workflow requires the masked inputs and newly computed feature tables. Shared legacy source artifacts must exist before the expanded-summary initialization; it must precede any supervised prediction in that shared legacy tree.

From the public repository root:

```bash
RAW=/absolute/path/to/corrected_raw
SHARED=/absolute/path/to/fresh/core/work
HIST=/absolute/path/to/fresh/historical/work

python historical_reproduction/code/run_slope.py \
  --raw "$RAW" --out "$HIST/slope" --corrected

INITIAL="$HIST/initial/confidence_validity"
python historical_reproduction/code/run_historical_predictions.py initial init \
  --raw "$RAW" --shared "$SHARED" --out "$INITIAL"
python historical_reproduction/code/run_historical_predictions.py initial features \
  --raw "$RAW" --shared "$SHARED" --out "$INITIAL"
python independent_verification/release_historical.py initial source --out "$INITIAL"
python historical_reproduction/code/run_historical_predictions.py initial predict \
  --raw "$RAW" --shared "$SHARED" --out "$INITIAL"
python independent_verification/release_historical.py initial seal --out "$INITIAL"
python historical_reproduction/code/run_historical_predictions.py initial evaluate \
  --raw "$RAW" --shared "$SHARED" --out "$INITIAL"
python independent_verification/release_historical.py initial results --out "$INITIAL"

EXPANDED="$HIST/expanded/confidence_comparison/mainline_v2"
python historical_reproduction/code/run_historical_predictions.py expanded init \
  --raw "$RAW" --shared "$SHARED" --out "$EXPANDED"
python independent_verification/release_historical.py expanded source --out "$EXPANDED"
python historical_reproduction/code/run_historical_predictions.py expanded predict \
  --raw "$RAW" --shared "$SHARED" --out "$EXPANDED"
python independent_verification/release_historical.py expanded seal --out "$EXPANDED"
python historical_reproduction/code/run_historical_predictions.py expanded evaluate \
  --raw "$RAW" --shared "$SHARED" --out "$EXPANDED"
python independent_verification/release_historical.py expanded results --out "$EXPANDED"
```

Use fresh output directories. Subprocesses have a 12-hour cap. For the slope command, the phases `--phase arc`, `--phase hellaswag`, and `--phase evaluate` may also be run separately. The slope reader checks every analysis-input file SHA for each fitted benchmark and rebuilds primary and secondary-initialization ALS fits. Its point estimates and uploader jackknife intervals are compared only after fitting. The slope output and reference contain descriptive raw statistics and the uploader-group definition. Their role is to describe cross-benchmark response-pattern stability.

The five-summary study builds its own reference features and uses explicit-decimal-edge target binning; the expanded-summary study reuses the freshly computed legacy source and **floor-bin** targets, then runs its own nested predictors. Do not interchange their target scorers. Reference caching reduces the five-summary maximum 105 reference requests to 55 distinct reference sets; these are separate from the expanded-summary branch's shared 55 fits. Five-summary supervision performs 720 inner and 20 final ridge fits. Expanded-summary supervision performs 2,720 inner and 80 family-final fits, including its conditional deployment-fit branch.

The checker is a separate executable. Its source receipt binds the current driver/checker, scientific modules, all mandatory feature files and reference caches, source-array identity, cohort qualification, and target identities. Its seal receipt verifies every out-of-fold model, organization, fold, prediction column and training-context request, plus candidates/recommendations for the expanded study. Final outcomes are read only by the scoring service after that seal. The result check independently recalculates organization-weighted losses and seeded paired-bootstrap intervals and compares all reported result fields with the reference outputs.

`reference_originals/corrected_historical/` contains comparison-only result JSONs and canonical value digests. These are never substituted for fresh predictions or features. Digest agreement checks identity; it is not itself evidence that a fit ran. Run-specific `run_events.jsonl`, logs, fitted output files and phase receipts document actual execution. Receipts are automated verification artifacts, not an external scientific endorsement.

The five-summary R50 interval crosses zero (the overall two-endpoint verdict is `CAL_ONLY`). The expanded-summary selected response-parameter package also fails its predefined R50 support criterion. The five-summary 97.5% and expanded-summary 95% intervals remain separate from the main experiment's simultaneous 31-comparison intervals. The full-panel joint model is documented in its separate reproduction workflow.

Public cold validation covers all three analyses. Both prediction studies reproduced all reference predictions and reported statistics, with 45 permitted training-query contexts and one final scoring query per study. The four slope ALS fits reproduced the reported raw statistics exactly.
