# Scientific protocol

Fixed definitions and settings for the public reproduction. Machine-readable contracts accompany the code.

## Fixed definitions (all phases)

- Source/target: ARC-Challenge item records (source) → HellaSwag model-level risk (target).
- Source item halves: the original SHA256(seed|item-ID) rank-mod-2 rule is recomputed on
  the corrected retained item union; the original organization folds are preserved.
  Response-parameter statistics use half `F` only, traditional statistics use all valid items;
  invalid items are masked, never zero-filled.
- Descriptors: per-option probability `p = softmax(L_k/h_k)` over real options only,
  `c = max_k p_k`, `ell = u_gold - logsumexp(u_notgold)`, `y = 1[argmax(u) == gold]` with the
  smallest original option index on exact ties; ties in ranking use the canonical item ID.
- Traditional summaries `X_0 = (A, C, O, REL10, R50, ECE10, Brier, H, Margin)` with ten equal-width
  confidence bins `b = min(floor(10c), 9)`, `REL10 = sum_b (n_b/N)(mean_c - mean_y)^2`,
  `ECE10 = sum_b (n_b/N)|mean_c - mean_y|`, `Brier = mean_i sum_k (p_ik - 1[k=gold_i])^2` (no
  division by K), `H = mean_i -sum_k p_ik log p_ik / log(K_i)`, `Margin` as the top-1 minus top-2
  probability, `R50` = error rate among the `ceil(N/2)` most confident items ordered by `c`
  descending then canonical item ID.
- Structure library: the 20 registered metrics `A1..A5, B1..B7, C1..C4, D1..D4` exactly as
  defined in `formula_registry.json` (formula text, boundary rules, dtype, log base,
  minimum-reference rules and imputation are all carried there).
- Organization folds: five outer folds and four inner folds over uploading organizations, frozen
  in `split_manifest.json` / the archived split functions; no refolding, no new data, no MMLU.
- Random seed: `20260914`; single-threaded numerical libraries (`*_NUM_THREADS=1`).

## Fixed settings (risk-metric phase)

- Endpoints: primary `Y_sel = R50` (HellaSwag), secondary `Y_cal = REL10` (HellaSwag).
- Design: `B0 = [X_0, M]` (29 columns), `Bd = [X_0, imputed Z_std, M]` (41), `P_j = [B0, Z_j]`
  for the 20 registered metrics (30), `Q_j = [Bd, Z_j]` for C/D (42), joint `J = [Bd, Z_str]`
  (49); `M` are the 20 per-metric missing flags. 31 fixed packages, 2 procedures, 31 contrasts.
- Ridge: `lambda in {1e-4 ... 1e4}` (9 configurations), weights `v_m = 1/(G n_g)` summing to 1,
  train-only weighted standardization with `sd <= 1e-12` columns zeroed, unpenalized intercept,
  predictions clipped to `[0, 1]`.
- HGB: `sklearn.ensemble.HistGradientBoostingRegressor` (scikit-learn 1.6.1) with
  `squared_error`, `max_iter=200`, `min_samples_leaf=20`, `max_bins=255`, `early_stopping=False`,
  `random_state=20260914`; 8 configurations over `learning_rate {0.05, 0.1}`, `max_leaf_nodes
  {7, 15}`, `l2_regularization {1, 10}`; sample weights `N_train/(G n_g)`; predictions clipped.
- Selection: per configuration 4 inner fits on the same inner folds; inner loss is the mean of
  per-organization mean squared errors over pooled inner-validation rows; tie rule
  `loss <= minimum + 1e-12`, prefer ridge, then larger lambda, then HGB fewer leaves, larger L2,
  lower learning rate; each package's global-best is selected in that order, baseline first.
  No outer-fold outcome may enter fitting, preprocessing or selection.
- Counts (from `expected_counts`): 31 fixed packages, 2 procedures, 31 contrasts, 190 OOF
  prediction columns, 21,080 inner fits, 620 family final fits, at most 310 additional global-best
  final fits (22,010 fits maximum).
- Primary inference: 2,000 shared paired organization-bootstrap draws,
  `numpy.random.default_rng(20260914)`, with loss floor `1e-12`. For each draw, take the
  maximum absolute standardized deviation `(bootstrap_delta - delta) / se` across the
  31 calculable primary comparisons. The simultaneous interval is `delta +/- critical * se`,
  where `critical` is the 0.95 linear quantile of these maxima. Support requires relative
  improvement `>= 0.05`, simultaneous lower bound `> 0`, and `>= 4/5` positive outer folds.
  Secondary/descriptive intervals use ordinary linear percentiles `[0.025, 0.975]`.
- Selection study: candidate rule `A_m >= max A - 0.02` with at least 3 candidates, at least 30
  case organizations and 3 outer folds for `DESCRIPTIVE_ONLY` status. The study uses
  2,000-draw paired case-organization bootstrap with unadjusted descriptive intervals and
  no binary decision-success threshold; insufficient cases stay `INSUFFICIENT_CASES`.
- Forbidden (from `protocol.json`): outer-test `Y` in fitting/preprocessing/selection,
  candidate-specific cohort filtering, target columns as source features, new inference or
  dataset, post-outcome formula or threshold changes.

## Fixed settings (comparison/legacy phase)

- Feature groups `B = (A, C, O, REL10, R50)`; `C = B + (ECE10, Brier, H, Margin)`;
  `L = C + (theta)`; `F = L + (log_s, log_sigma)`.
- 16 family cells (B/C/L/F × ridge/HGB × Y_sel/Y_cal) and 8 deployment cells
  (B/C/L/F × best × Y_sel/Y_cal), all on the same OOF models and the same per-context rows.
- Candidate configurations: 9 ridge lambdas and 8 HGB combinations as listed above.
- Selection: inner loss pooled over the four inner folds; tie rule as above; family-best refit on
  all qualified outer-train rows; the deployment model is the global-best configuration
  (refit separately when it differs from the family-selected one); no new search.
- Inference: institution-unit paired bootstrap, 2,000 draws, seed `20260914`, percentile
  `[0.025, 0.975]` linear, fixed OOF predictions, no refitting; primary contrast `F_best` vs
  `C_best` on `Y_sel`; support `I >= 0.05`, lower bound `> 0`, and `>= 4/5` positive folds;
  historical adaptive selection is not corrected and is disclosed.

The machine-readable protocol, formula, access, cohort and split contracts under
`original_inputs/confidence_comparison/` accompany these definitions.
