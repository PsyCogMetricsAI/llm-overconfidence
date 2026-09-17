# Joint Dual-Latent IRT: Identification, Constraints, Gradients

**Date:** 2026-06-14
**Author role:** S0 Generator (symbolic / numerical derivation, CPU)
**Evidence file:** `verify_identification.py` (sympy symbolic invariance + numpy rank / null-space + finite-difference gradient check). Every numeric claim below is reproduced by `python3 verify_identification.py`; the relevant Part is cited inline.
**Existing conventions matched:** `beta_irt.py` uses `mu = sigmoid(alpha_i*(theta_m + z_i))`, `phi=10`, additive `theta+z`. The candidate Eq2 is exactly this with two added model-side terms (`theta^C_m`, inside the same envelope) and a length term.

---

## 0. TL;DR verdict

1. **Identifiable after constraints?** YES — but **only with a non-trivial fix to the length term and only because Eq1 (the 2PL on correctness) anchors `theta^A` independently.** The candidate model *as written* (global `gamma`, free per-item `z_i`, no mean-zero constraints) has an **8-dimensional unidentified gauge** in Eq2 (Part B). The global `gamma` is **provably not identifiable** (collinear with `z_i`).
2. **`theta^C` clean from `theta^A`?** YES, *conditional on Eq1 anchoring `theta^A`*. `theta^A` and `theta^C` enter Eq2 with the **identical** coefficient `alpha_i`, so Eq2 alone sees only their sum `g_m = theta^A_m + theta^C_m` and **cannot split them** (Part A-a'', Part D). The split is rescued *entirely* by Eq1, which contains `theta^A` but **not** `theta^C`. This is the load-bearing assumption — flagged in §7.
3. **`theta^C` clean from length?** Only if length control is done right. The **global `gamma` is useless** (absorbed by `z_i`); the real risk is **per-model verbosity**, which **masquerades as overconfidence** if uncontrolled (Part E: a model that is purely length-sensitive with zero true overconfidence is recovered with `theta^C` correlating **+1.000** with its length sensitivity). Fix: **per-model `gamma_m`** length slope; it removes the leak and does **not** bias a genuine `theta^C` (Part F: recovery corr +1.000, variance-inflation factor 1.003).

---

## 1. Joint log-likelihood

Index models `m = 1..M`, items `i = 1..I`. Two observed channels per cell `(i,m)`:

- correctness `c_{im} ∈ {0,1}` (Bernoulli, Eq1),
- confidence `y_{im} = exp(logprob of the correct answer) ∈ (0,1)` (Beta, Eq2).

**Linear predictors** (the sigmoid is injective, so identification questions reduce to the arguments `eta`):

```
eta1_{im} = a_i · (theta^A_m − b_i)                              (Eq1, 2PL logit)
eta2_{im} = alpha_i · (theta^A_m + theta^C_m + z_i) + gamma·L̃_i  (Eq2, Beta logit)
p_{im}  = sigmoid(eta1_{im})            (Bernoulli success prob)
mu_{im} = sigmoid(eta2_{im})            (Beta mean), phi fixed = 10
```

**Bernoulli term** (one cell):

```
ℓ1_{im} = c_{im} log p_{im} + (1 − c_{im}) log(1 − p_{im})
```

**Beta term** (one cell), with `A = mu·phi`, `B = (1−mu)·phi`:

```
ℓ2_{im} = (A − 1) log y_{im} + (B − 1) log(1 − y_{im})
          − [ lnΓ(A) + lnΓ(B) − lnΓ(A + B) ]
```

(The bracket is `ln Beta(A,B)`; matches `beta_nll` in `beta_irt.py:30-33`.)

**Joint log-likelihood** (independence across cells; missing cells masked):

```
LL(Θ) = Σ_{i,m∈obs1} ℓ1_{im}  +  Σ_{i,m∈obs2} ℓ2_{im}
```

Joint NLL `= −LL`. Parameter set
`Θ = { theta^A_m, theta^C_m ; a_i, b_i, alpha_i, z_i ; gamma }`, with `phi` fixed,
`L̃_i` a **known per-item constant** (centered length of the correct answer; same for all models).

> Note on the coupling that makes this hard: `theta^A_m` is the **only** parameter shared by both sums. Eq1 is a clean 2PL in `theta^A` alone. Everything that distinguishes `theta^C` from `theta^A` must come through this asymmetry — `theta^A` is in Eq1, `theta^C` is not.

---

## 2. Identification: complete gauge inventory

A *gauge freedom* is a parameter transform that leaves **every** observable prediction (`p_{im}` and `mu_{im}` for all `i,m`) unchanged. Such a direction is unidentified: the likelihood is flat along it. We enumerate them symbolically (Part A, sympy: each candidate substituted into both `eta1` and `eta2`, simplified to confirm `delta == 0`) and confirm the count by null-space rank of the Eq2 design (Part B).

### 2.1 The location gauges (shift freedoms)

All three of the following leave **both** equations literally invariant (Part A: `delta = 0` for both):

| Gauge | Transform | Both-invariant? (Part A) |
|---|---|---|
| **(a) global A↔C shift** | `theta^A_m += c`, `theta^C_m −= c`, `b_i += c` | **YES** |
| **(a′) ability location** | `theta^A_m += c`, `z_i −= c (∀i)`, `b_i += c` | **YES** |
| **(a″) overconfidence location** | `theta^C_m += c`, `z_i −= c (∀i)` | **YES** |

The deep cause: in `eta2`, the three terms `alpha_i·theta^A_m`, `alpha_i·theta^C_m`, `alpha_i·z_i` all carry the **same** per-item coefficient `alpha_i`. Any constant moved among `theta^A`, `theta^C`, and a uniform shift of `z` is invisible to Eq2. Eq1 only pins the *combination* `theta^A − b`, so a matching shift of `b` hides the `theta^A` move there too.

**Consequence:** `theta^A` and `theta^C` each have a free additive location, and there is one shared shift between them. These are removed by **mean-zero constraints**:

```
mean_m theta^A_m = 0        (kills theta^A location; gauges a′)
mean_m theta^C_m = 0        (kills theta^C location; gauge a″)
mean_i z_i = 0  (recommended; or fix one z_i)   (removes z's uniform DOF)
```

With `theta^A` independently centered, gauge (a) — the A↔C swap — is also removed: you cannot shift `c` from `theta^A` into `theta^C` while keeping **both** means at zero.

### 2.2 ★ The length collinearity (risk b — CONFIRMED FATAL for global gamma)

**Claim:** the global length coefficient `gamma` is **not identifiable**: `gamma·L̃_i` is fully absorbed by the free per-item parameter `z_i`.

**Proof (Part A-b, A-b2, sympy):** `gamma·L̃_i` and `alpha_i·z_i` are *both* per-item additive constants in `eta2`. The transform

```
gamma → gamma + d ,  z_i → z_i − d·L̃_i / alpha_i   (∀i)
```

leaves `eta2` unchanged (`delta = 0`), and touches nothing in `eta1`. Equivalently, set `gamma → 0` and fold its effect entirely into `z`: `z_i → z_i + gamma·L̃_i / alpha_i` (Part A-b2, also `delta = 0`). So **any value of `gamma` is observationally equivalent** — the data cannot pin it. The global length term is a redundant reparameterization of the item base, **not** a usable length control.

**Numerical confirmation (Part B):** the vector "`gamma += 1`, `z_i −= L̃_i/alpha_i`" lies in the null space of the Eq2 design (residual 5.6e-17).

> **This directly answers risk (b)(i):** because `z_i` is a free per-item parameter, per-item length variation is *already* fully soaked up at the item level. After fitting, `z_i` has implicitly absorbed `L̃_i`; the model-level `theta^C` is therefore **clean of any *item-level* length effect by construction**. A global `gamma` adds nothing.

**So what is the real length risk?** Not item length (already controlled by `z_i`), but **model × length interaction** — a *verbose* model that assigns systematically higher logprob mass on long correct answers. That is a **per-model** length slope, which varies across `m` and is therefore *not* absorbable by item-level `z_i`. This is risk (b)(ii), addressed in §4.

### 2.3 Scale gauge (risk c)

Standard 2PL has a scale freedom `theta^A *= s, a_i /= s, b_i *= s` that leaves `eta1` invariant. **Does it leak into Eq2?** **No — and this is good news.** Part A-c (sympy): under this transform `eta2`'s delta is `alpha_i·theta^A_m·(s−1) ≠ 0`. Eq2 contains a *bare* `theta^A` (coefficient `alpha_i`, not tied to `a_i`), so rescaling `theta^A` changes `mu`. **Eq2 pins the scale of `theta^A`.** Therefore we do **not** need a separate variance constraint on `theta^A`; the unit is set by the Beta channel. (`alpha_i` and `phi` carry the Eq2 discrimination/precision scale, exactly as in the existing `beta_irt.py` baseline; `phi` is fixed = 10, which removes the Beta precision freedom.)

### 2.4 Null-space count (Part B, M=6, I=8)

Eq2 linear-predictor design over the additive block
`p = [theta^A (M), theta^C (M), z (I), gamma (1)]`:

- **Unconstrained:** rank 13 / 21, **nullity = 8.** Named gauges confirmed in null: A↔C shift (G1), C/z shift (G2), A/z shift (G2b), gamma/z collinearity (G3) — all residual ≤ 5.6e-17.
- **After the fix** (`theta^A` fixed by Eq1; center `theta^C`; center `z`): **nullity = 0 → IDENTIFIED.**

---

## 3. Minimal constraint set (recommended)

| # | Constraint | Removes |
|---|---|---|
| C1 | `mean_m theta^A_m = 0` | `theta^A` location (gauge a′) |
| C2 | `mean_m theta^C_m = 0` | `theta^C` location (gauge a″) and, jointly with C1, the A↔C swap (gauge a) |
| C3 | `mean_i z_i = 0` (or fix one `z_i = 0`) | uniform `z` DOF |
| C4 | **Drop global `gamma`** | length/`z` collinearity (gauge b) — `gamma` is unidentifiable, see §2.2 |
| C5 | `phi` fixed = 10 | Beta precision scale (already in `beta_irt.py`) |
| — | scale of `theta^A` | **no constraint needed** — pinned by Eq2's bare `theta^A` (§2.3) |

Identifiability does **not** require `a_i > 0` / `alpha_i > 0`, but those sign conventions (already in `beta_irt.py`) are recommended to fix the reflection symmetry of each discrimination.

---

## 4. Recommended final model form (with corrected length control)

**Drop the global `gamma`.** Add a **per-model length slope `gamma_m`** to capture model×length interaction (verbosity), which item-level `z_i` cannot absorb because it varies across models:

```
Eq1:  P(c_{im}=1) = sigmoid( a_i · (theta^A_m − b_i) )

Eq2:  y_{im} ~ Beta(mu·phi, (1−mu)·phi),  phi = 10
      mu_{im} = sigmoid( alpha_i · (theta^A_m + theta^C_m + z_i) + gamma_m · L̃_i )
```

with constraints C1, C2, C3, C5 **plus**:

| # | Constraint | Removes |
|---|---|---|
| C6 | `mean_m gamma_m = 0` | the constant component of `gamma_m`, which is collinear with the (now removed) global `gamma` / with `z` |

**Why `gamma_m` is identifiable where `gamma` was not:** `gamma_m·L̃_i` *varies across models* (the coefficient is per-`m`, multiplying a fixed `L̃_i` profile). Item-level `z_i` is per-item, constant across models, so it cannot mimic a model-varying length effect. Part A's absorption argument (which killed global `gamma`) does not apply.

**Caveat — `gamma_m` vs `theta^C` separability (risk b-ii answered):** for a single model, `theta^C_m` multiplies the item vector `alpha_i` and `gamma_m` multiplies the item vector `L̃_i`. They are separable as long as `L̃` is not collinear with `alpha`. The cost of adding `gamma_m` is a variance-inflation factor on `theta^C`:

```
VIF(theta^C_m) = 1 / (1 − corr(alpha, L̃)^2)
```

- **Does omitting `gamma_m` bias `theta^C`?** YES, badly. Part E: with true `theta^C = 0` for everyone but varying per-model length sensitivity, a model fit **without** `gamma_m` recovers `theta^C` correlating **+1.000** with the (unmodeled) length sensitivity — pure verbosity is mislabeled overconfidence.
- **Does adding `gamma_m` bias a genuine `theta^C`?** NO. Part F: with both effects truly present, the joint fit recovers `theta^C` at corr **+1.000** (max err 1e-15); `corr(alpha, L̃) ≈ −0.05 → VIF ≈ 1.003`, i.e. negligible variance cost. So `gamma_m` is safe to include and costly to omit.

> **Practical guard:** before fitting, report `corr(L̃_i, alpha_i)`. If it is large (say > 0.6, VIF > 1.5), `theta^C` and `gamma_m` are weakly separated and `theta^C` estimates will be noisy — surface this as a data-quality flag, not a silent result.

> **Scope of the +1.000 recoveries (Parts E/F).** These hold in the **known-item-params linear-residual regime** (item parameters and the Eq1 anchor treated as given; the per-model fit is then a 2-column linear regression of the Eq2 residual on `[alpha, L̃]`). They establish *identifiability* — that the length confound is structurally removable and `theta^C` is structurally clean once `gamma_m` is included. Full joint nonlinear MLE on noisy data will degrade gracefully relative to this ceiling, governed by the stated VIF and by sample size (M models × I items).

**Constrained null-space check with `gamma_m` (Part B, with_gamma_m=True):** unconstrained nullity 9; after C1/C2/C3 the **residual nullity is 1**, traced to the constant component of `gamma_m` (Part B gauge G4b: `gamma_m(all)+=1, gamma−=1` is in the null). **C6 (`mean_m gamma_m = 0`) removes exactly this**, giving full identification. The non-constant gauge G4 (`gamma_m` vs `theta^C`) is *not* in the null (residual 2.26) — confirming the two are structurally distinct.

---

## 5. theta^C uniqueness: "overconfidence = residual"

**Proposition.** Under C1–C6, with item parameters identified, `theta^C_m` is uniquely identified as the Eq2 model effect **minus** the Eq1-anchored ability:

```
theta^C_m = g_m − theta^A_m,   where g_m := (Eq2 per-model effect) = theta^A_m + theta^C_m
```

**Argument (Part D, exact numerics).**
1. Eq2 sees `theta^A` and `theta^C` **only through their sum** `g_m` (identical coefficient `alpha_i`). With ≥2 items of distinct `alpha_i` (and `z`, `gamma_m` identified), `g_m` is identified by Eq2. But the *split* `g_m = theta^A_m + theta^C_m` is a 1-parameter family — Part D confirms a shift `theta^A+c / theta^C−c` gives identical `g_m` (max diff 4e-16). **Eq2 alone cannot separate them.**
2. **Eq1 breaks the tie:** Eq1 is a 2PL containing `theta^A` but **not** `theta^C`. It identifies `theta^A` up to location+scale; C1 fixes location, and §2.3 shows Eq2 fixes the scale. So `theta^A_m` is pinned.
3. Then `theta^C_m = g_m − theta^A_m` is the unique residual. Part D: with a perfect Eq1 anchor, `theta^C` is recovered exactly (max err 2.2e-16). The interpretation is literal: **`theta^C_m` is the confidence a model shows in Eq2 beyond what its Eq1-measured ability explains** — relative over- (or under-) confidence, centered to mean 0 across the model panel.

> **Scope of the 2.2e-16 number.** Part D verifies the *algebra* of `theta^C = g_m − theta^A_m` under a **perfect** Eq1 anchor (`theta^A_hat := theta^A_true`). It establishes that the residual is *uniquely determined* once `theta^A` is pinned and centering is imposed — i.e. there is no remaining gauge. It does **not** measure empirical recoverability on noisy real data, which depends on (i) how well Eq1 estimates `theta^A` and (ii) the construct-validity assumption in step 2. Read it as "the split is mathematically well-posed," not "recovery is guaranteed in practice."

**Honest caveat (the one decision point).** Step 2 *requires* that Eq1's `theta^A` and Eq2's `theta^A` are **the same latent on the same scale**. The model imposes this by sharing the symbol. If, empirically, the ability that drives correctness is not exactly the ability that drives correct-answer logprob (e.g. a model is calibrated differently than it is accurate), then the shared-`theta^A` assumption is *misspecified*, and `theta^C` will absorb that misspecification rather than pure overconfidence. This is not an identification failure (the math is clean) but a **construct-validity** risk — see §7. It is the same class of risk that sank the earlier "logprob θ ≈ mean-logprob gauge" line (memory: logprob-g = −0.974): if `theta^A` from Eq1 is near-collinear with mean logprob, then `theta^C` is near-zero by construction and carries little independent signal.

---

## 6. Gradients (score) of the joint NLL

Let `NLL = −LL`. Per-cell intermediate quantities (recommended model, §4):

```
p_{im}  = sigmoid(eta1_{im});   eta1_{im} = a_i (theta^A_m − b_i)
mu_{im} = sigmoid(eta2_{im});   eta2_{im} = alpha_i (theta^A_m + theta^C_m + z_i) + gamma_m L̃_i
A = mu·phi,  B = (1−mu)·phi
```

**Channel derivatives w.r.t. their logits** (the two reusable building blocks):

```
∂NLL1/∂eta1_{im} = p_{im} − c_{im}                                     (Bernoulli)
∂NLL2/∂eta2_{im} = [ −phi·( ln y_{im} − ln(1−y_{im}) − (ψ(A) − ψ(B)) ) ] · mu_{im}(1−mu_{im})
                  =: d2_{im}        (ψ = digamma; the bracket is ∂NLL2/∂mu, then ×mu(1−mu))
```

**Scores** (sum over the relevant index; `D1_{im} := p_{im} − c_{im}`):

```
∂NLL/∂theta^A_m  = Σ_i [ D1_{im}·a_i ]  +  Σ_i [ d2_{im}·alpha_i ]      (BOTH channels)
∂NLL/∂theta^C_m  = Σ_i [ d2_{im}·alpha_i ]                              (Eq2 only)
∂NLL/∂gamma_m    = Σ_i [ d2_{im}·L̃_i ]                                 (Eq2 only)
∂NLL/∂a_i        = Σ_m [ D1_{im}·(theta^A_m − b_i) ]                    (Eq1)
∂NLL/∂b_i        = Σ_m [ D1_{im}·(−a_i) ]                               (Eq1)
∂NLL/∂alpha_i    = Σ_m [ d2_{im}·(theta^A_m + theta^C_m + z_i) ]        (Eq2)
∂NLL/∂z_i        = Σ_m [ d2_{im}·alpha_i ]                              (Eq2)
```

Constraints C1/C2/C3/C6 are imposed either by parameterizing each centered block as deviations (drop one DOF) or by projecting the gradient onto the mean-zero subspace (subtract the block mean of the score) before each optimizer step.

**Symbolic vs numerical check (Part C):** analytic scores above vs central finite differences (`eps = 1e-6`) on a random instance (M=4, I=5, 29 free params incl. a global `gamma` as the test stand-in for `gamma_m`'s identical algebra):

```
max |analytic − numeric| = 5.83e-09     max relative error = 1.12e-08
per-block max abs err: thA 5.1e-9, thC 5.8e-9, a 1.5e-9, b 1.5e-9,
                       alpha 3.6e-9, z 5.4e-9, gamma 6.3e-10
>>> GRADIENT CHECK PASS  (threshold 1e-4)
```

(`∂NLL/∂gamma` in Part C is algebraically identical to one summand of `∂NLL/∂gamma_m`, so the per-model length score is verified by the same check.)

---

## 7. Decision points for the human (blockers / construct-validity)

1. **[Resolved by math] Global `gamma` is unidentifiable.** Do not include a global length coefficient. Use per-model `gamma_m` (C6) or rely on item-level `z_i` (which already cleans item length). — *no human input needed, but the model spec must change.*

2. **[★ Real decision] Shared-`theta^A` construct validity.** Identification of `theta^C` as "overconfidence residual" is mathematically clean **only if** the latent driving correctness (Eq1) is the *same* latent on the *same scale* as the one driving correct-answer logprob (Eq2). The model *assumes* this by sharing the symbol. If they diverge empirically, `theta^C` silently absorbs ability-misspecification, not overconfidence. **Recommend:** before trusting `theta^C`, check (i) `corr(theta^A_Eq1-only, theta^A_Eq2-only)` from separate fits — if not high, the shared latent is questionable; (ii) `corr(theta^A, mean-logprob)` — if near 1, `theta^C` will be near-zero by construction (the gauge/−0.974 trap from prior work). This is the single most important caveat and a candidate stop-and-confirm before production.

3. **[Decision] `corr(L̃_i, alpha_i)` data check.** `theta^C` and `gamma_m` separate cleanly only when length is not collinear with Eq2 discrimination. Report this correlation; if large (VIF > ~1.5), `theta^C` is poorly determined and the result should carry a wide-uncertainty flag.

4. **[Minor] Missingness / sparsity.** Identification proofs assume each model answers ≥2 items with distinct `alpha_i`, and the panel is connected. With heavy missingness the constrained rank should be re-checked on the actual mask (the Part B null-space check can be rerun with the real design).

---

## 8. Reproduction

```
python3 verify_identification.py
```

- **Part A** sympy invariance of `eta1`, `eta2` under 6 candidate transforms (§2.1–2.3).
- **Part B** numpy rank / null-space of the Eq2 design, named gauges, constrained rank (§2.4, §4).
- **Part C** analytic vs finite-difference joint-NLL gradient (§6).
- **Part D** `theta^C = g_m − theta^A_m` uniqueness (§5).
- **Part E** verbosity-masquerades-as-overconfidence demo (§4).
- **Part F** `gamma_m` is bias-free for genuine `theta^C`, VIF cost (§4).

All summary lines: A all location gauges both-invariant (scale not); B constrained nullity 0 (no `gamma_m`) and 0 after C6 (with `gamma_m`); C grad PASS 5.8e-9; D unique 2.2e-16; E leak corr +1.000 → 5e-17 with `gamma_m`; F recovery corr +1.000, VIF 1.003.
