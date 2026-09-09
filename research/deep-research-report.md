# Leakage-safe modeling strategies for a causal, cold-start-heavy forecasting task

## Problem framing from the attached artifacts

The dataset is a row-level regression task with unique `(code, sub_code, sub_category, horizon, ts_index)` per row, with `id` defined as a concatenation of those fields. fileciteturn0file0 The train/test split is a strict forward split in `ts_index` (train `1..3601`, test `3602..4376`). fileciteturn0file0

The evaluation metric (“Skill Score”) is a *weighted, clipped* normalized MSE converted to a square-root skill:

- `denom = Σ w * y^2`
- `ratio = Σ w * (y − ŷ)^2 / denom`
- `clipped = clip(ratio, 0, 1)`
- `score = sqrt(1 − clipped)` fileciteturn0file0

So:
- Perfect predictions give score `1`.
- Predicting all zeros gives score ≈ `0` (because the numerator ≈ denom).
- Being worse than zero is clipped (no negative score). fileciteturn0file0

Two structural facts from the audit dominate modeling choices:

- **Heavy cold-start at `(code, sub_code)`**: ~**89.1% of test rows** are on `(code, sub_code)` keys unseen in train. fileciteturn0file1  
- **Full coverage at `(code, sub_category, horizon)`**: that level has **0 unseen test keys**, making it the only “guaranteed coverage” categorical resolution. fileciteturn0file1

Your current baseline family is consistent with those constraints: a **smoothed stable grouped mean** at `(code, sub_category, horizon)` is the defensible always-available fallback. fileciteturn0file0 The baseline’s exact concatenated-fold proxy score is about **0.11195**, with materially worse performance on cold-start rows (about **0.1451**) and clipped-to-zero behavior on some folds/horizons. fileciteturn0file2turn0file0

The best current offline model is a **small, clipped residual correction on top of that baseline** (a ridge “tri-slice” residual overlay), reaching **0.1128626342**. fileciteturn0file0 A strict out-of-fold normalized residual replacement branch underperformed (best around **0.11016**). fileciteturn0file5turn0file0

Finally, the artifacts explicitly flag two operational constraints:

- **Sequential rule**: prediction and *test preprocessing* must be strictly causal, disallowing any use of future test information. fileciteturn0file0  
- **Leakage hotspot**: community discussion flagged **`feature_al`** as potentially target-reconstructive; it should be treated as “high risk” until proven safe. fileciteturn0file0

## Ranked modeling ideas most applicable to this setting

What “wins” here tends to be (a) *hierarchical shrinkage that is robust under massive cold-start*, plus (b) *very conservative overlays* that cannot blow up the clipped metric. The ranking below reflects that.

1. Hierarchical empirical Bayes offsets for warm-start `(code, sub_code)` with partial pooling to `(code, sub_category, horizon)`

**Why it fits this data shape**  
You have full coverage at `(code, sub_category, horizon)` but only a small warm-start fraction at `(code, sub_code)`. The right move is to treat sub-code information as a **random effect** (offset) that is *learned when available* and otherwise shrinks to zero, i.e., back to the parent mean. This is the classic partial pooling / shrinkage setup emphasized in multilevel modeling. citeturn5search1turn5search6

A very practical instantiation is the normal-means empirical Bayes form, where the posterior mean becomes a **data-dependent shrinkage** of each group mean toward the global/parent mean. citeturn3search4turn3search6

**Leakage risks**  
High if you compute offsets using future `ts_index` relative to the prediction row (or fold). The correct implementation is:
- For CV: compute offsets using only the training time window within each fold.
- For test-time: offsets are computed from full training only (no test aggregation), then looked up sequentially. fileciteturn0file0

**Expected upside**  
Moderate-to-high given the current pipeline: it directly targets the ~11% warm-start rows while being guaranteed harmless on cold-start (offset = 0). Upside is larger if warm-start rows carry disproportionate `w*y^2` “energy” in the metric’s denominator (worth checking), but even without that, this is one of the few improvements that is both *safe* and *structurally correct* for the hierarchy you actually have. fileciteturn0file1turn0file0

**Implementation cost**  
Low. Pure `pandas` groupby aggregates + a shrinkage formula (or `numba` for speed). No heavy modeling libraries required.

2. Mixture-of-experts gating between “coarse always-safe” and “fine warm-start” experts

**Why it fits this data shape**  
With 89% cold-start, you want an architecture that has:
- an expert that is *always defined* (your current baseline at `(code, sub_category, horizon)`), and
- one or more experts that are only trusted when there is evidence (e.g., warm-start sub-code effects, or slice-specific residual models).

This is exactly the intent of mixture-of-experts: multiple specialized predictors blended by a gating function. citeturn1search1turn1search2

In this competition, the gate does not need to be a neural net; a simple function of **history count, recency, and stability** is often stronger and safer.

**Leakage risks**  
The gate can leak if it uses any statistic computed from the full test set (even test features) because the rule requires sequential preprocessing. fileciteturn0file0  
Safe gating features include:
- counts and variances computed from training only,
- time-since-last-observation in training,
- sub-code counts accumulated only from *past* test rows (if you maintain state sequentially).

**Expected upside**  
High, because it’s a general framework that can incorporate *several* “small conservative overlays” while keeping cold-start robust. It also matches the clipped metric: you can design gates to be conservative, minimizing risk of making squared error worse than the already-barely-better-than-zero baseline. fileciteturn0file0turn0file2

**Implementation cost**  
Low-to-medium depending on whether the gate is hand-designed (low) or learned via OOF optimization (medium).

3. Stacked blending of multiple leakage-safe baselines (multi-resolution + multi-timescale) with constrained weights

**Why it fits this data shape**  
When the target is close to unpredictable and skill scores are small (your baseline is ~0.112), forecast **combination** is a strong strategy because it reduces variance and exploits weak, complementary signals. The forecasting literature’s core result is that combined forecasts often beat individual forecasts in MSE. citeturn8search0turn8search3

A very competition-practical version is to build a set of baselines, all computable causally without test aggregation, such as:
- long-window smoothed mean at `(code, sub_category, horizon)` (your current),
- EWMA mean at the same level with several half-lives,
- a “global zero” baseline (ŷ=0),
- coarse means like `(code, horizon)` or `(sub_category, horizon)` (all train-only),
then learn stacking weights on OOF predictions with strong regularization and optional non-negativity constraints.

This is conceptually aligned with hierarchical “optimal combination” ideas in forecasting (though you’re not reconciling sums here). citeturn4search4turn4search5

**Leakage risks**  
Stacking can leak if weights are learned on in-fold predictions. You need:
- foldwise generation of each baseline strictly from fold-train,
- stacking learned only on OOF predictions. citeturn6search0

**Expected upside**  
Moderate. This tends to deliver steady but not huge uplifts; however, it’s one of the most reliable ways to get “free” improvements when individual signals are weak and noisy (which your skill score suggests). fileciteturn0file2turn0file0

**Implementation cost**  
Medium. Engineering multiple baselines + OOF stacking pipeline, but all in standard Python.

4. Dynamic state-space / exponential-smoothing mean model at `(code, sub_category, horizon)` (with shrinkage back to long-run mean)

**Why it fits this data shape**  
Your baseline uses “stable all-history” means because recent-window ablations regressed. fileciteturn0file0 But the fold diagnostics show clipped-to-zero folds and horizon slices, which is often a sign of **regime drift** or time-varying mean where a static mean can be actively harmful. fileciteturn0file2

Exponential smoothing and its state-space formulation give a causal way to estimate a time-varying level (and optionally trend/seasonality) and forecast it forward, updating only using past observations. citeturn2search6turn6search4

The key is to **shrink the dynamic level back toward the long-run group mean** when evidence is weak, rather than replacing the baseline outright.

**Leakage risks**  
Low if implemented correctly: state-space updates must use only `(y,w)` up to time `t`. The main pitfall is accidentally fitting smoothing parameters using future data within a fold. Use rolling-origin CV. citeturn6search0

**Expected upside**  
Potentially high specifically on the folds/slices that clip to 0 under static means. Even small reductions in squared error can move a slice from “clipped” to “positive,” which is disproportionately valuable in this metric. fileciteturn0file2turn0file0

**Implementation cost**  
Medium. Feasible because there are only 460 `(code, sub_category, horizon)` groups with full test coverage. fileciteturn0file1

5. Ordered/causal target-statistics encoding for categorical interactions + conservative residual model

**Why it fits this data shape**  
High-cardinality categorical interactions are naturally informative in these problems, but naive target encoding is a leakage trap. The CatBoost paper explains how target statistics (smoothed means) can leak by using the current row’s target, and motivates ordered (history-only) computation to avoid conditional shift. citeturn9view0turn0search1

Even if you don’t use CatBoost directly, you can adopt the same principle:
- compute mean encodings for keys like `(code, sub_category, horizon)`, `(code, horizon)`, `(sub_category, horizon)`,
- and for warm-start keys `(code, sub_code)` or `(code, sub_code, horizon)`,
but always in a fold-OOF, time-ordered way.

**Leakage risks**  
High if you compute encodings on the full dataset or in random KFold. Must be:
- forward-split by `ts_index`,
- encoding for a row must not include its target (or future targets),
- no test-set aggregation (sequential preprocessing rule). fileciteturn0file0turn9view0

Also, your artifacts flag `feature_al` as potentially target-reconstructive; a residual model that relies on it may win CV but fail the rules or private LB. fileciteturn0file0

**Expected upside**  
Moderate. This is especially useful if you can find a small subset of features/slices with stable signal (as your “tri-slice ridge residual” suggests). fileciteturn0file0

**Implementation cost**  
Medium. Either:
- implement ordered encodings yourself, or
- use a library that supports leakage-aware encodings (still requires care for sequential constraints).

6. Horizon-conditioned multi-task residual modeling (share strength across horizons instead of separate models)

**Why it fits this data shape**  
Horizon is categorical (`[1,3,10,25]`) and explicitly “not a numeric offset.” fileciteturn0file0 Still, horizons are related tasks: the mapping from features → predictability often changes smoothly with horizon even if “horizon” is a class label.

Instead of training four totally separate residual models, train a single model with:
- one-hot horizon + interactions (or horizon-specific coefficients with a group penalty),
- strong regularization to prevent overfit.

This matches “cross-learning” findings reported in analyses of Kaggle forecasting competitions, where global/cross-series learning and ensembles are often strong. citeturn7search6turn7search7

**Leakage risks**  
Low if you stay within forward CV and avoid target leakage in encodings.

**Expected upside**  
Small-to-moderate, but it tends to be robust because it reduces variance in coefficient estimates and lets weak signals accumulate across horizons.

**Implementation cost**  
Low. Multi-output ridge or a single ridge with horizon interactions is straightforward.

7. Systematic “slice specialist” residual experts with automatic eligibility rules

**Why it fits this data shape**  
Your best model already behaves like “specialists”: a clipped ridge correction trained for selected sub-categories/horizons performs best. fileciteturn0file0 A scalable generalization is:
- train a small residual expert per `(sub_category, horizon)` (only 20 combos),
- but only *activate* an expert where OOF evidence is positive and stable.

This is a disciplined way to expand the “triweak” idea without making broad corrections that hurt clipped folds.

**Leakage risks**  
Moderate: the biggest risk is “expert selection on validation” turning into overfitting. You need:
- nested evaluation or a fixed policy based on earlier folds,
- and conservative activation thresholds.

**Expected upside**  
Moderate. It’s aligned with the empirical observation that “feature-only models are weak overall,” but may have pockets of signal.

**Implementation cost**  
Medium. Mostly pipeline complexity, not math.

8. Robust regression (Huber / t-like) for residual overlays + strict magnitude control

**Why it fits this data shape**  
The metric is squared-error-based but clipped; catastrophic mistakes disproportionately destroy score (pushing SSE/denom above 1 and clipping to 0). fileciteturn0file0 Robust losses (Huber-like) can reduce the influence of extreme residuals when fitting overlays, making them safer to apply.

Separately, clip overlay magnitude by a quantile calibrated on OOF residuals (you already do this in spirit). fileciteturn0file0

**Leakage risks**  
Low, assuming proper fold discipline. The main risk is tuning clip thresholds on the same folds you report.

**Expected upside**  
Small-to-moderate, but often very reliable in “weak signal + heavy tails” regimes.

**Implementation cost**  
Low (scikit-learn already supports robust regressors).

9. Post-hoc shrinkage / “temperature scaling” of any correction to maximize clipped skill

**Why it fits this data shape**  
Because the metric is monotone in squared error (until clipping), a very common competition trick is to fit a model, then learn a single scalar (or a few scalars per slice) that shrinks predictions toward a safer baseline, chosen to improve the held-out metric.

The forecast-combination literature explicitly discusses shrinking combining weights toward central tendencies for robustness. citeturn8search2turn8search3

In your setting, the simplest version is:
- ŷ = ŷ_base + λ * correction,
with λ learned on OOF to maximize the exact metric.

**Leakage risks**  
Moderate: λ must be learned on OOF predictions (not in-fold).

**Expected upside**  
Small but high ROI: this can turn a “sometimes good, sometimes harmful” correction into a consistently positive one.

**Implementation cost**  
Low. One-dimensional search per slice.

10. Embedding-based neural residual model (categorical embeddings + numeric features), gated and shrinked

**Why it fits this data shape**  
Neural embeddings are a standard way to model entity-like categorical variables, but your cold-start rate at `(code, sub_code)` is extreme. That means embeddings for most test sub_codes are unseen, so the model must rely on higher-level categories and numeric features anyway.

This is only worth trying as a *residual overlay* behind a conservative gate, not as a primary model.

**Leakage risks**  
Mostly pipeline-related (folding, sequential preprocessing). The bigger practical risk is overfitting to CV quirks, especially if any feature is target-proxy (e.g., `feature_al`). fileciteturn0file0

**Expected upside**  
Uncertain; likely low unless you discover stable non-linear interactions in selected slices.

**Implementation cost**  
High relative to the likely gain.

## Warm-start overlays on seen `(code, sub_code)` keys

Warm-start opportunities exist (97 overlapping `(code, sub_code)` keys), but they cover only ~10.9% of test rows, so overlays must be **very cheap** and **very safe**. fileciteturn0file1 The most effective overlays are ones that:  
(1) revert exactly to the baseline when history is weak, and (2) never create large signed drift that can push SSE above the zero baseline.

A practical template is a **three-level shrinked offset** added on top of your existing `(code, sub_category, horizon)` baseline:

Let:
- `b(i) = baseline(code, sub_category, horizon, ts_index)` (your smoothed stable mean)
- residual `r = y − b`

Define three nested offsets estimated on training only (causally within fold):

1) **Sub-code global offset within code**:  
`δ₁(code, sub_code)` = shrinked mean of `r` over all rows of the pair.

2) **Sub-code × sub-category offset**:  
`δ₂(code, sub_code, sub_category)` = shrinked mean of `(r − δ₁)`.

3) **Full fine offset**:  
`δ₃(code, sub_code, sub_category, horizon)` = shrinked mean of `(r − δ₁ − δ₂)`.

Prediction becomes:
`ŷ = b + g₁*δ₁ + g₂*δ₂ + g₃*δ₃`,  
where each gate `gk ∈ [0,1]` is a monotone function of training evidence for that level.

### Recommended gating and shrinkage for warm-start

Use an empirical Bayes / James–Stein-style weight:  
`shrink_weight = n / (n + k)`  
where `n` is an effective weighted sample size and `k` is a prior-strength hyperparameter estimated per slice (or tuned). This is the same structure that arises in empirical Bayes shrinkage of means. citeturn3search4turn3search6

Concrete, competition-friendly gates:

- **Count gate**: `g = n_eff / (n_eff + k)` (automatic).
- **Recency gate**: multiply by `exp(−Δt/τ)` where `Δt` is time since last train observation for that key; this protects against stale sub_codes.
- **Stability gate**: multiply by `I[|δ̂| < clip_level]` or a smooth version (e.g., `1 / (1 + (|δ̂|/c)^p)`), where `clip_level` is an OOF-calibrated quantile of offset magnitudes.

Then apply a final clip: `δ_total = clip(δ_total, ±Q)` per `(sub_category, horizon)` because the metric clips hard at SSE/denom ≥ 1 and your artifacts already show that broad unshrunk corrections are not justified while some folds clip. fileciteturn0file0turn0file2

### How to keep this strictly causal at test time

- Precompute `(δ₁, δ₂, δ₃)` from training only.
- At test time, for each row, do dictionary lookups by key and apply gates using precomputed `n_eff`, `Δt`, and stored stats.
- Do **not** compute any per-test-key means, variances, or centering using the test set; the rules require sequential preprocessing. fileciteturn0file0

### Where warm-start overlays are most likely to pay

Given your diagnostics emphasize gains in horizons 1 and 3 and certain sub-categories (e.g., `DPPUO5X2` focus in the log), prioritize learning/tuning `k, τ, clip` per `(sub_category, horizon)` rather than globally. fileciteturn0file0

## Hierarchical shrinkage and mixture-of-experts recommendations

This section gives implementable “templates” you can drop into your existing baseline + correction design.

### Empirical Bayes shrinkage for offsets

A canonical EB derivation is:

- Observed group mean `\bar r_g` has variance ≈ `σ² / n_g`.
- Group true effect `θ_g ~ Normal(0, τ²)`.
- Posterior mean is a convex combination of `\bar r_g` and 0 with weight `τ² / (τ² + σ²/n_g)`, which is equivalent to `n_g / (n_g + k)` after reparameterization. citeturn3search6turn3search4

In practice you can:
- estimate `(σ², τ²)` per `(code, sub_category, horizon)` or per `(sub_category, horizon)` using method-of-moments, or
- tune `k` directly on OOF.

This approach is aligned with the partial pooling logic described in multilevel modeling references: complete pooling (all offsets 0) vs no pooling (raw group means) vs partial pooling (shrinked). citeturn5search1turn5search6

**Key practical detail for your metric:**  
Because score is `sqrt(1 - SSE/denom)` with clipping, *overconfident offsets* can be catastrophic. Use conservative priors (larger `k`) and per-slice clipping calibrated on OOF.

### Mixture-of-experts design that is simple enough for Kaggle-style constraints

A high-ROI mixture structure for your setting:

- Expert A (always): `ŷ_A = baseline(code, sub_category, horizon, ts_index)`
- Expert B (warm-start): `ŷ_B = ŷ_A + δ̂(code, sub_code, …)` (shrinked EB offsets)
- Expert C (slice residual): `ŷ_C = ŷ_A + ML_residual(x)` but only for validated slices

Final prediction:
`ŷ = w_A*ŷ_A + w_B*ŷ_B + w_C*ŷ_C`,  
with weights summing to 1.

Weights can be computed as:
- `w_B ∝ evidence_strength(sub_code)` (e.g., `n_eff/(n_eff+k)`),
- `w_C ∝ slice_trust(sub_category,horizon)` (OOF gain, stability),
- `w_A` is the remainder.

The mixture-of-experts notion—specialized experts combined by input-dependent weights—goes back to classic work by entity["people","Robert A. Jacobs","mixture of experts author"] and entity["people","Michael I. Jordan","machine learning researcher"] and colleagues. citeturn1search1turn1search2

**Make it leakage-safe and sequential**
- Only use gating features that are computed from training or from past test rows.
- Do not compute any global test statistics (mean/std by test groups).
- Learn any gate parameters (like `k`, thresholds, or slice eligibility rules) on OOF only. fileciteturn0file0

### Forecast-combination-inspired stacking as a special case

If you treat `(ŷ_A, ŷ_B, ŷ_C, …)` as candidate forecasts and learn weights, you are doing forecast combination. The fundamental result that combined forecasts can reduce MSE traces back to entity["people","C. W. J. Granger","economist forecast combination"] and entity["people","J. M. Bates","economist forecast combination"]. citeturn8search0 The broader review literature (e.g., Clemen) emphasizes that even simple combinations can be robust. citeturn8search3

In your clipped-skill context, stacking should be:
- strongly regularized (ridge),
- optionally constrained (non-negative weights, sum-to-one),
- and learned on OOF predictions only.

## Are transformed-target residuals promising here?

You asked specifically about **asinh**, **relative residual**, and **multiplicative correction**. The best answer depends on (a) target sign distribution, (b) heteroskedasticity, and (c) whether the predictable component lives in *levels* or *ratios*. Your artifacts already provide one strong clue: a strict OOF-normalized residual branch underperformed your incumbent residual correction. fileciteturn0file5turn0file0 That doesn’t eliminate transforms, but it suggests “normalize everything” is not automatically beneficial.

### asinh transforms

The inverse hyperbolic sine (asinh) transform is widely used as a “log-like” transform that can handle zeros and negatives, and is discussed in econometrics contexts as a robust way to reduce the influence of extreme dependent-variable values. citeturn3search3turn3search2

**Why it might help**
- If `y_target` is heavy-tailed with both signs, asinh can stabilize optimization for residual models and reduce sensitivity to extreme outliers. citeturn3search3turn3search0
- If your overlays occasionally “go wild,” fitting in asinh-space can produce inherently conservative corrections.

**Why it might hurt**
- Your metric is fundamentally weighted squared error on the original scale (through `Σ w (y-ŷ)^2`). fileciteturn0file0  
  Compressing extremes during training can reduce attention on high-`|y|` rows that contribute more to both numerator and denominator (`w*y^2`), potentially reducing true skill.

**Recommendation**
- asinh is most promising *not* as a wholesale target transform for the main model, but as a way to model **residual corrections** where the main risk is outlier-driven overfitting.
- If you try it, use it in a *delta form*, such as modeling `asinh(y) − asinh(baseline)` and then inverting via `sinh(asinh(baseline) + Δ̂)`, with a final amplitude clip calibrated on OOF.

### Relative residuals (additive-to-relative reparameterization)

A relative residual like `r_rel = (y − b) / (|b| + ε)` is attractive when error scales with the baseline magnitude.

**Why it might help**
- If conditional variance grows with scale, relative residuals can reduce heteroskedasticity and make a linear residual model behave better.

**Why it might not (given your evidence)**
- Your strict OOF-normalized residual attempt—conceptually similar to “scale residuals then learn”—did not beat the incumbent. fileciteturn0file5turn0file0  
- If `b` is often near 0 (common in finance-like series), relative residuals can become unstable unless ε is carefully chosen and the correction is aggressively clipped.

**Recommendation**
- Treat “relative residual” as a niche option: only activate it in groups/slices where `|b|` is reliably away from 0 and where OOF shows clear wins.

### Multiplicative corrections

A multiplicative correction is `ŷ = b * (1 + m)` or `ŷ = b * s`, with `m` or `log s` predicted.

**Why it might help**
- It naturally preserves sign when `b` is predictive and the main error is scale.
- It also tends to be conservative if you constrain `s` near 1.

**Why it might hurt**
- If `b` crosses zero or is frequently small, multiplicative adjustments become unstable or meaningless.
- Your metric and baseline both suggest the predictable component is weak; multiplicative scaling can amplify noise unless heavily regularized.

**Recommendation**
- Multiplicative corrections are most promising as a *warm-start-only* overlay, where sub_code-specific behavior might be a stable scaling of the parent baseline. Use EB shrinkage on `log(|y|+ε) − log(|b|+ε)` or on `asinh(y) − asinh(b)` to avoid sign/zero issues. citeturn3search3turn3search2

## Do-next branches to try first

These are ordered by expected ROI under your constraints (cold-start dominance, clipped metric, leakage sensitivity), and by how well they complement what already works (smoothed hierarchical mean + small clipped residual).

### Branch to try first: EB warm-start offsets + conservative gating

Implement the three-level warm-start offset described above (`δ₁/δ₂/δ₃`) with:
- shrink weights `n/(n+k)` learned per `(sub_category, horizon)`,
- optional recency decay from training,
- hard clipping per slice.

This is the cleanest “structural” improvement because it cannot harm cold-start rows (offset=0) and should be stable under strict sequential test rules. fileciteturn0file1turn0file0

### Branch to try next: multi-baseline stacking with strong regularization

Construct 6–12 leakage-safe baseline predictors (multi-resolution, multi-half-life) and stack them:
- learn weights on OOF predictions only (rolling-origin / forward CV),
- constrain or shrink weights for robustness.

This directly leverages forecast-combination evidence and is a classic way to get incremental gains when individual signals are weak. citeturn8search0turn8search3turn6search0

### Branch to try third: dynamic (state-space) parent-group mean + shrink back to stable mean

For each `(code, sub_category, horizon)` series, fit a local-level exponential smoothing / Kalman-filter mean estimate, but blend it with the stable long-run mean using an EB-style shrink (so you don’t repeat the “recent-window regression” failure mode). fileciteturn0file0turn2search6

Target this branch especially at the horizons/folds where your current baselines clip to 0, because moving those slices above the clipping boundary is disproportionately valuable. fileciteturn0file2turn0file0