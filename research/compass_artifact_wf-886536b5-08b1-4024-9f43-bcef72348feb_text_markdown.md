# Modeling strategies for heavy cold-start hierarchical forecasting

**The grouped mean at the (code, sub_category, horizon) level is provably the right foundation for this competition's 89% cold-start problem — but systematic improvements of 10–30% are available through hierarchical shrinkage, target normalization, and warm-start overlays.** Evidence from M5, Predict Future Sales, and Corporación Favorita competitions confirms that hierarchical fallback means dominate when most test entities are unseen, and that incremental gains come from empirical Bayes shrinkage, group-normalized residual modeling, and careful LightGBM feature engineering using only coarser-level aggregates. The critical insight: for the 89% cold-start rows, no method can reliably beat the group mean — but for the 11% warm-start rows and for refining the group mean itself, substantial gains are available. The clipped metric means your downside risk is bounded: any group where the model underperforms the baseline simply scores 0.

---

## Deliverable 1: Ranked list of 10 modeling ideas

The following ideas are ranked by expected improvement over the grouped-mean baseline, weighted by applicability to this specific data shape (89% cold-start, hierarchical groups, causal processing, weighted clipped skill-score).

**1. Hierarchical empirical Bayes shrinkage of group means.** Apply James-Stein / EB shrinkage at the (code, sub_category, horizon) level, shrinking noisy group means toward the global or code-level mean. Groups with fewer observations get pulled more toward the population. For cold-start sub_codes, prediction equals the shrunk coarser-level mean. Evidence: Efron & Morris (1973, 1975) showed **10–30% MSE reduction** over raw group means. Amazon's 2022 CIKM paper demonstrated EB for cold-start product search with **+13.5% new-item impressions**. Computationally trivial — O(k) for k groups.

**2. Group-normalized target residual modeling with LightGBM.** Transform targets to `(y − group_mean) / group_std` before training LightGBM on coarse-level features (code, sub_category, horizon, interactions). The model learns systematic corrections to group means. Wikipedia Web Traffic gold-medal solution used exactly this pattern: log1p followed by per-series standardization. Corporación Favorita top solutions used log1p normalization. This directly addresses the heterogeneous scale problem and allows LightGBM to focus on relative deviations.

**3. Multi-level pooled LightGBM ensemble (M5 1st-place pattern).** Train separate LightGBM models at different hierarchical pooling levels — one model per code, one per (code, sub_category), one global — then simple-average predictions. The M5 1st-place winner trained **220 models** across 3 pooling levels with both recursive and direct forecasting, and the equal-weighted average was the final submission. This provides ensemble diversity that naturally handles cold-start through coarser models.

**4. Warm-start exponential smoothing overlay with Bayesian blending.** For the ~11% of test rows where (code, sub_code) is observed, compute sub_code-level exponential smoothing forecasts and blend with group-level predictions using credibility weight `w = n / (n + k)`. The actuarial credibility formula is mathematically equivalent to empirical Bayes shrinkage. The Predict Future Sales competition validated this pattern — sub_code-level lags were the most important features for observed entities.

**5. Hierarchical target encoding as multi-level features.** Provide LightGBM with separate target-encoded features at each hierarchy level: (code), (code, sub_category), (code, sub_category, horizon), and (code, sub_code) where available. The "Target-Thresh" approach from Carey (2024) on NAICS hierarchical codes showed that setting unseen fine-level codes to NULL and letting the tree model learn the fallback was **especially resistant to overfitting**. Use CatBoost-style ordered encoding for leakage safety.

**6. Horizon-specific models with separate tuning.** Train separate models per horizon (or horizon bucket h∈{1,3} vs h∈{6,12,...}). The M5 winner used weekly horizon-specific models. Your worst slices are at **h=1 and h=3**, suggesting short horizons have fundamentally different signal-to-noise ratios. Separate tuning allows more aggressive smoothing for noisy short horizons and less smoothing where longer-horizon patterns are detectable.

**7. Sample weight alignment with competition metric.** Set LightGBM sample weights proportional to `group_metric_weight / group_baseline_energy²`, aligning the L2 loss gradient with the weighted skill-score metric. Post-process: for any group where the model prediction yields a worse score than the group mean, replace with the group mean (free improvement under clipping). Competition evidence from M5 shows that sample weights and proper feature engineering mattered far more than custom loss functions.

**8. Entity embeddings with stochastic sub_code dropout.** Train a neural network with embedding layers for code, sub_category, and sub_code. During training, randomly mask sub_code to an "unknown" token with probability 0.3–0.5. Carey (2024) demonstrated on NAICS hierarchical codes that this stochastic regularization allowed **unseen-code performance to approach known-code performance**. The model learns to degrade gracefully from sub_code-specific to group-level prediction.

**9. PyMC hierarchical Bayesian model with nested partial pooling.** A three-level Bayesian model (global → code,sub_category,horizon → code,sub_code,sub_category,horizon) with partial pooling naturally provides uncertainty-weighted predictions and principled cold-start fallback. The TSB-HB paper (arXiv:2511.12749) demonstrated this for intermittent demand with **lowest RMSE/RMSSE** among baselines. Non-centered parameterization with NUTS sampling runs in **30–120 seconds** for ~460 groups.

**10. Matrix factorization with metadata regression for warm-start.** Xie et al. (arXiv:1710.08473) proposed a unified framework combining high-dimensional regression from group features (for cold-start) with matrix factorization on residuals (for warm-start). The regression component ≈ sophisticated grouped mean; the MF component captures latent structure among observed sub_codes. Primarily benefits the 11% warm-start rows.

---

## Deliverable 2: Detailed analysis of each idea

### Idea 1 — Hierarchical empirical Bayes shrinkage

**Why it fits:** Your 460 (code, sub_category, horizon) groups have varying sample sizes, and the grouped-mean baseline treats all group means as equally reliable regardless of sample size. EB shrinkage directly corrects this by pulling noisy group means toward the population. With ~89% cold-start, the coarse-level estimate IS the prediction for most rows — making even a 5% improvement in coarse-level accuracy a large absolute gain.

**Leakage risks:** Minimal. The shrinkage is computed from training-set group statistics. Under sequential folds, recompute hyperparameters (τ², μ) on each fold's training portion only. The method is inherently causal since it uses only historical aggregates.

**Expected upside:** **5–15% improvement in overall score**, concentrated on groups with small sample sizes where raw means are noisy. The gain scales with the ratio of within-group to between-group variance.

**Implementation cost:** **Low.** Pure numpy code, <1ms runtime, ~20 lines of code. Drop-in via `category_encoders.JamesSteinEncoder`.

### Idea 2 — Group-normalized target residual modeling

**Why it fits:** Your groups span heterogeneous scales, and the energy-denominator weighting means high-variance groups dominate the metric. By normalizing targets to `(y − μ_g) / σ_g`, LightGBM operates on comparable-scale residuals across all groups, preventing large-scale groups from dominating the gradient. The model then learns systematic biases: "for code X at horizon 1, the group mean tends to overpredict by 8%."

**Leakage risks:** Moderate — `μ_g` and `σ_g` must be computed from training data only, not from the fold being predicted. Use expanding-window group statistics. For sequential folds, recompute group stats per fold.

**Expected upside:** **10–25% improvement.** This is the single highest-impact technique for tree-based models in multi-scale settings, per evidence from M5 and Wikipedia competitions.

**Implementation cost:** **Low-Medium.** Requires careful causal group-stat computation but standard pandas/numpy code.

### Idea 3 — Multi-level pooled LightGBM ensemble

**Why it fits:** The M5 1st-place validation is strong. Different pooling levels capture different granularities of pattern: code-level models learn broad trends, (code, sub_category)-level models capture category-specific behavior, and finer models add detail. The simple average is robust and hard to beat with learned weights.

**Leakage risks:** Low if each model respects the sequential fold structure. Each LightGBM model is trained on its fold's training data only.

**Expected upside:** **5–15% over a single model.** The diversity benefit is well-documented: the M5 winner's 6-model average was stronger than any single model. Your gain will depend on how diverse the models actually are.

**Implementation cost:** **Medium.** Requires training multiple LightGBM models and managing the pooling-level feature engineering for each.

### Idea 4 — Warm-start exponential smoothing overlay

**Why it fits:** The 11% warm-start rows have sub_code-level history that is completely ignored by the coarse grouped-mean baseline. Even simple exponential smoothing at the sub_code level captures the entity's own level, which may deviate substantially from the group mean.

**Leakage risks:** Low if ETS uses only past observations (`shift(1)`). The blending weight `w = n/(n+k)` must be tuned on validation data, not test data. Risk: with very few observations per sub_code (97 overlapping pairs, potentially few rows each), ETS estimates may be noisy.

**Expected upside:** **15–30% improvement on the warm-start slice** (which is ~11% of rows). Overall impact: **~2–4% on aggregate score**, but potentially larger if warm-start rows have higher metric weights.

**Implementation cost:** **Low.** Pandas `ewm` + simple blending formula, ~30 lines of code.

### Idea 5 — Hierarchical target encoding as multi-level features

**Why it fits:** Instead of a single blended target-encoded feature, providing LightGBM with separate encoded features at each hierarchy level lets the tree model learn *when* to trust fine-level vs. coarse-level information. The count feature (number of observations in each group) acts as a reliability indicator. For unseen sub_codes, the fine-level feature is NaN — LightGBM handles NaN natively via best-split routing.

**Leakage risks:** **High if not done carefully.** Target encoding on the training set leaks target information. Mitigate with: (a) CatBoost-style ordered encoding (each row encoded using only preceding rows), (b) leave-one-out with smoothing, or (c) out-of-fold encoding using the sequential validation folds. Never use `fit_transform` on the full training set and then evaluate on a temporal hold-out.

**Expected upside:** **5–15%** when combined with LightGBM. The multi-feature approach consistently outperforms single-blended encoding per Carey (2024).

**Implementation cost:** **Low-Medium.** `category_encoders.TargetEncoder(hierarchy=...)` provides the core functionality, but safe sequential encoding requires custom code.

### Idea 6 — Horizon-specific models

**Why it fits:** Your worst performance is at h=1 and h=3, suggesting fundamentally different dynamics at short horizons. A single model across all horizons is forced to compromise between short-horizon noise and long-horizon signal. Separate models allow tuned hyperparameters: more regularization (deeper shrinkage) for short horizons, more flexibility for longer horizons.

**Leakage risks:** None inherent — each horizon model sees only its own rows.

**Expected upside:** **5–15%**, concentrated on the problematic short horizons. If h=1/h=3 groups have disproportionate metric weight, the overall impact is larger.

**Implementation cost:** **Low.** Simply filter by horizon and train separate LightGBM models. Doubles or triples training time.

### Idea 7 — Sample weight alignment with metric

**Why it fits:** The weighted clipped skill-score creates a non-uniform optimization landscape — some groups matter more than others, and groups below the clipping threshold contribute zero. By setting sample weights proportional to the competition metric weights and inverse to the baseline energy denominator, the L2 loss becomes a smooth proxy for the skill score. The post-processing clip (replace with group mean when model is worse) is a free, risk-free improvement.

**Leakage risks:** None — weights are derived from training-set group properties and the known metric formula.

**Expected upside:** **3–10%** from weight alignment; **1–5%** from post-processing clip. The clip is essentially free insurance.

**Implementation cost:** **Low.** Set `lgb.Dataset(X, y, weight=w)` and add a post-prediction comparison step.

### Idea 8 — Entity embeddings with stochastic dropout

**Why it fits:** Neural networks with entity embeddings can learn rich representations of the hierarchical group structure. The stochastic sub_code dropout during training teaches the model to predict well even when the fine-level code is missing — precisely the cold-start scenario. However, this requires a neural network pipeline.

**Leakage risks:** Low — embeddings are learned from features, not direct target encoding. Validation must be sequential.

**Expected upside:** **10–20%** for the cold-start slice where the stochastic regularization has direct impact. The approach is less proven than EB shrinkage for this exact problem shape.

**Implementation cost:** **High.** Requires PyTorch/Keras neural network training, embedding layer management, careful dropout implementation, and longer training times.

### Idea 9 — PyMC hierarchical Bayesian model

**Why it fits:** A three-level Bayesian model provides principled uncertainty quantification and natural cold-start fallback through the population distribution. The non-centered parameterization handles varying group sizes gracefully. The posterior predictive distribution for unseen groups is exactly the population distribution, informed by all training data.

**Leakage risks:** Low if MCMC is run on training data only per fold.

**Expected upside:** **5–15%** over raw group means; similar to EB for point predictions, but with uncertainty estimates that enable more sophisticated downstream decisions (e.g., when to trust the model vs. default to the group mean).

**Implementation cost:** **Medium-High.** Requires PyMC setup, ~30–120 seconds per fold, and careful handling of predictions for unseen groups. Debugging divergences can be time-consuming.

### Idea 10 — Matrix factorization with metadata regression

**Why it fits:** The framework explicitly handles the cold-start / warm-start split: metadata regression for cold-start (equivalent to a sophisticated grouped mean), MF residuals for warm-start. Primarily valuable for squeezing additional signal from the 97 overlapping sub_code pairs.

**Leakage risks:** Moderate — the MF component must not use future-period columns.

**Expected upside:** **5–10%** on the warm-start slice (~1% on aggregate). The metadata regression component is unlikely to improve over a well-tuned EB grouped mean.

**Implementation cost:** **High.** Requires custom matrix construction, alternating optimization, and careful temporal segmentation.

---

## Deliverable 3: Warm-start overlay recommendations

For the ~11% of test rows where (code, sub_code) is seen in training, you have sub_code-level history that the coarse grouped mean ignores. Three specific overlay strategies:

### Sub_code-level expanding mean with credibility blending

Compute the expanding (cumulative) mean of the target for each seen sub_code, using only observations prior to the prediction point. Blend with the group-level prediction using the actuarial credibility formula: `ŷ = w × sub_code_mean + (1−w) × group_pred`, where `w = n / (n + k)` and `k` is a smoothing hyperparameter tuned on validation. When `n` is small, the prediction collapses to the group mean; when `n` is large, the sub_code's own level dominates. Implementation: `df.groupby('sub_code')['target'].expanding().mean().shift(1)` in pandas. Leakage risk is zero if `shift(1)` is applied. **Tune `k` on your 4 sequential folds** — typical range is 10–100. This is the simplest and most robust warm-start overlay.

### Sub_code-level exponential smoothing with alpha optimization

For sub_codes with temporal ordering, exponential smoothing captures recent trends better than expanding means. Use `df.groupby('sub_code')['target'].ewm(alpha=α, adjust=False).mean().shift(1)`. Optimize `α` using `scipy.optimize.minimize_scalar` on validation MSE, bounded between 0.01 and 0.99. For sub_codes with very few observations (under 3), fall back to the group mean. The ETS forecast becomes one input to the credibility blending formula above. Leakage risk: none with `shift(1)`. **Key caveat**: with only 97 overlapping sub_code pairs, many may have sparse history — aggressive smoothing toward the group mean (small `α`, high `k`) is safer.

### Residual correction conditioned on sub_code deviation

Compute residuals `r = y − group_pred` for each training observation. For each warm-start sub_code, compute the mean residual from its training history: `mean_residual = mean(r | sub_code)`. Apply as a correction: `ŷ_warm = group_pred + shrink(mean_residual)`, where `shrink` is the EB shrinkage formula `(1 − B) × mean_residual` with `B = SE²_residual / (τ²_residual + SE²_residual)`. This captures systematic offsets — e.g., a specific sub_code that consistently runs 15% above its group mean. Leakage risk: the residuals must be computed from training data only, and the group_pred used to compute residuals must NOT be fitted on the same data used to estimate the residual correction. Use separate folds or an expanding-window approach.

### Blending protocol to avoid leakage

The critical anti-leakage rule: **never compute the group mean and the sub_code-level residual from the same data partition.** Specifically:
- Compute group means on fold k's training data
- Compute sub_code residuals (relative to those group means) on the same training data
- Apply both corrections on fold k's validation data
- The sub_code-level correction is additive on top of the group mean, not an independent estimate

This ensures the residual correction captures genuine sub_code-specific signal rather than noise from the group mean estimation.

---

## Deliverable 4: Hierarchical shrinkage and mixture-of-experts methods

### James-Stein shrinkage for your 460 groups

The MSS (Multi-Sample-Size) James-Stein estimator is the simplest high-value method. For each (code, sub_category, horizon) group with sample mean `x_i`, within-group variance `s²_i`, and count `n_i`:

```python
sigma2_i = s2_i / n_i                              # SE² of group mean
tau2 = np.var(group_means, ddof=1) - np.mean(sigma2_i)  # between-group variance
tau2 = max(tau2, 1e-10)                             # prevent negative
B_i = sigma2_i / (tau2 + sigma2_i)                  # shrinkage weight
shrunk_i = (1 - B_i) * x_i + B_i * grand_mean      # shrunk estimate
```

Groups with small `n_i` (high `σ²_i`) get pulled strongly toward the grand mean; groups with large `n_i` retain their observed mean. Efron & Morris showed ~31% MSE reduction in their classic baseball example. **Runtime: <1ms for 460 groups.** The `category_encoders.JamesSteinEncoder` provides a drop-in sklearn API.

### Two-level empirical Bayes with hierarchical prior

Apply EB at two levels sequentially. First, shrink the 460 (code, sub_category, horizon) group means toward the code-level means (or global mean) using the MLE procedure: estimate τ² by maximizing the marginal likelihood `∏ Normal(x_i | μ, τ² + σ²_i)` via `scipy.optimize.minimize_scalar`. Second, for the ~1856 (code, sub_code, sub_category, horizon) groups observed in training, shrink toward their parent (code, sub_category, horizon) EB estimate. For the ~89% unseen fine-level groups, the prediction equals the Level 1 EB estimate directly. This is mathematically equivalent to a two-level hierarchical Bayesian model with point-estimated hyperparameters. **The key advantage over raw grouped means**: noisy groups with 5 observations are not treated the same as stable groups with 500 observations.

### PyMC nested partial pooling model

For a fully Bayesian approach, specify a three-level model with non-centered parameterization:

```python
with pm.Model() as model:
    mu_global = pm.Normal("mu_global", 0, 10)
    sigma_coarse = pm.HalfCauchy("sigma_coarse", 5)
    offset_coarse = pm.Normal("offset_coarse", 0, 1, shape=460)
    mu_coarse = mu_global + sigma_coarse * offset_coarse

    sigma_fine = pm.HalfCauchy("sigma_fine", 2)
    offset_fine = pm.Normal("offset_fine", 0, 1, shape=n_fine_groups)
    mu_fine = mu_coarse[parent_idx] + sigma_fine * offset_fine

    sigma_y = pm.HalfNormal("sigma_y", 5)
    y = pm.Normal("y", mu_fine[obs_idx], sigma_y, observed=y_data)
    idata = pm.sample(2000, tune=2000, target_accept=0.95)
```

For unseen fine-level groups, sample from `Normal(mu_coarse[parent], sigma_fine)` — the posterior predictive uses the informed population distribution. **Runtime: 30–120 seconds with NUTS.** Use `nuts_sampler='numpyro'` for 3–5× speedup via JAX. The TSB-HB paper validated this pattern for intermittent demand forecasting with cold-start.

### Mixture-of-experts with hierarchical gating

A simpler deterministic alternative to full Bayes: train three "experts" (global model, coarse-level model, fine-level model) and a gating function that routes based on data availability:

```python
def hierarchical_moe_predict(row, fine_model, coarse_model, global_model):
    fine_key = (row.code, row.sub_code, row.sub_category, row.horizon)
    coarse_key = (row.code, row.sub_category, row.horizon)
    if fine_key in fine_model:
        n = fine_model[fine_key]['count']
        alpha = n / (n + K_SMOOTH)
        return alpha * fine_model[fine_key]['pred'] + (1-alpha) * coarse_model[coarse_key]['pred']
    else:
        return coarse_model[coarse_key]['pred']
```

This is a learned version of your current fallback but with principled blending. The gating parameter `K_SMOOTH` can be tuned or replaced with a small LightGBM model that learns when to trust the fine-level expert. For this problem shape, the simple credibility-weight gating is likely sufficient — a neural gating network adds complexity without clear benefit when the hierarchy is known and clean.

### Recommendation across methods

For **speed and simplicity**, use James-Stein shrinkage (Idea 1). For **principled uncertainty**, use PyMC partial pooling (Idea 9). For **maximum flexibility with LightGBM**, use hierarchical target encoding at multiple levels as features (Idea 5). All three naturally handle cold-start by falling back to coarser levels. The expected improvement over raw group means is **10–30%** in MSE, with the EB/JS methods requiring minutes of implementation time.

---

## Deliverable 5: Target transform analysis

### Group normalization is the highest-priority transform

The evidence strongly favors **group-wise standardization** as the primary target transform: `y_norm = (y − μ_g) / max(σ_g, ε)` where `μ_g` and `σ_g` are the (code, sub_category, horizon) group mean and standard deviation. This puts all groups on a comparable scale, preventing high-variance groups from dominating the L2 gradient. The Wikipedia Web Traffic gold-medal solution used exactly this pattern. PyTorch Forecasting's `GroupNormalizer` implements it as a first-class API. When combined with LightGBM, the model learns relative corrections ("this code at this horizon overpredicts by 8%") rather than absolute values across wildly different scales.

**Inverse transform**: `ŷ = μ_g + σ_g × model_prediction`. For cold-start rows, `μ_g` and `σ_g` come from the coarser hierarchy level.

### Multiplicative correction factors deserve testing

Predicting the ratio `y / μ_g` (multiplicative correction) is a natural alternative. The baseline prediction is ratio = 1.0 (equivalent to the group mean). LightGBM then learns multiplicative deviations. This works well when the error structure is proportional to the level (heteroscedasticity proportional to mean), which is common in sales and energy data. **Risk**: division by near-zero group means — use a floor `max(μ_g, ε)` with `ε` set to a small percentile of the target distribution.

### asinh transform is a secondary option

The asinh transform `asinh(y) = ln(y + √(y²+1))` handles zeros and negatives gracefully, unlike log. For large values it behaves like log; near zero it's approximately linear. However, **asinh is sensitive to the scale of the input** — applying it directly to raw targets with heterogeneous scales is problematic. The correct approach: `asinh((y − μ_g) / σ_g)`, which first normalizes then compresses extreme residuals. This is valuable if the residual distribution has heavy tails, but if residuals are approximately Gaussian after group normalization, asinh adds little. In practice, group normalization alone captures most of the benefit, and adding asinh on top is an incremental improvement at best.

### Log-ratio targets are powerful but fragile

The transform `log(y / μ_g) = log(y) − log(μ_g)` combines the benefits of log compression and group normalization. The e-commerce forecasting paper (arXiv:1905.13614) explicitly normalized sales by dividing by the product's mean level, noting that "the sales time series are strongly heteroscedastic, and the local variance is strongly correlated with the local mean." However, this **fails when y can be zero or negative** — use `log1p(y / μ_g)` or the asinh variant if zeros are present.

### How transforms interact with the competition metric

The weighted clipped skill-score with energy-denominator weighting means that **reducing error in high-energy-denominator groups has less marginal impact** (they're already close to baseline). The transform doesn't change the metric — it changes what the model optimizes. Group normalization + L2 loss effectively optimizes a weighted MSE where each group contributes proportionally to `1/σ²_g`, which may or may not align with the competition weights. For exact alignment, combine group normalization with sample weights that match the metric's group-weight structure.

### Practical recommendation

Use **additive group standardization** `(y − μ_g) / σ_g` as the default transform. Test **multiplicative ratio** `y / μ_g` as an alternative — try both and select on validation. Apply asinh only if residuals after group normalization show heavy tails. Avoid pure log unless all targets are strictly positive. Always post-process: if the inverse-transformed prediction is worse than the group mean for a given group, replace it with the group mean (free improvement under clipping).

---

## Deliverable 6: Top 3 "do next" modeling branches

### Branch 1: Empirical Bayes shrinkage + group-normalized LightGBM (highest expected impact)

This is the highest-priority branch because it combines two complementary improvements: better group means (EB shrinkage) and a learnable residual model (LightGBM on normalized targets).

**Step 1 — Compute EB-shrunk group means (30 min):**
```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

def eb_shrink(df, group_cols, target_col):
    stats = df.groupby(group_cols)[target_col].agg(['mean','var','count'])
    grand_mean = df[target_col].mean()
    se2 = stats['var'] / stats['count']
    
    def neg_mll(log_tau2):
        tau2 = np.exp(log_tau2)
        return -norm.logpdf(stats['mean'], grand_mean, np.sqrt(tau2 + se2)).sum()
    
    tau2 = np.exp(minimize_scalar(neg_mll, bounds=(-10, 10), method='bounded').x)
    B = se2 / (tau2 + se2)
    stats['eb_mean'] = (1 - B) * stats['mean'] + B * grand_mean
    return stats[['eb_mean']].reset_index()

eb_means = eb_shrink(train, ['code','sub_category','horizon'], 'target')
```

**Step 2 — Group-normalize targets (15 min):**
```python
group_stats = train.groupby(['code','sub_category','horizon'])['target'].agg(['mean','std'])
group_stats.columns = ['g_mean', 'g_std']
group_stats['g_std'] = group_stats['g_std'].clip(lower=1e-6)

train = train.merge(group_stats, on=['code','sub_category','horizon'])
train['y_norm'] = (train['target'] - train['g_mean']) / train['g_std']
```

**Step 3 — Build LightGBM features (1 hr):**
```python
features = ['code_encoded', 'sub_category_encoded', 'horizon',
            'g_mean', 'g_std', 'group_count', 'eb_mean',
            'code_mean', 'code_std',  # code-level aggregates
            'horizon_mean',           # horizon-level aggregate
            'code_x_horizon_mean']    # interaction aggregate
# For warm-start rows, add: 'sub_code_expanding_mean', 'sub_code_count'
# For cold-start rows: these features are NaN (LightGBM handles natively)
```

**Step 4 — Train and evaluate (30 min):**
```python
import lightgbm as lgb
params = {'objective': 'regression', 'metric': 'rmse',
          'learning_rate': 0.05, 'num_leaves': 31,
          'min_data_in_leaf': 20, 'feature_fraction': 0.8}

# Train on y_norm, evaluate on original scale
model = lgb.train(params, lgb.Dataset(X_train, y_norm_train),
                  num_boost_round=1000,
                  valid_sets=[lgb.Dataset(X_val, y_norm_val)],
                  callbacks=[lgb.early_stopping(50)])

# Inverse transform: ŷ = g_mean + g_std * model_pred
y_pred = X_val['g_mean'] + X_val['g_std'] * model.predict(X_val)

# Post-process: clip to EB group mean when model is worse
y_baseline = X_val['eb_mean']
mask_worse = compute_group_score(y_pred) < compute_group_score(y_baseline)
y_pred[mask_worse] = y_baseline[mask_worse]
```

**Expected improvement: 10–25% over raw grouped mean.** Implementation time: ~3 hours.

### Branch 2: Warm-start overlay with ETS blending (medium impact, easy win)

This branch specifically targets the 11% of test rows with known sub_codes, extracting sub_code-level signal ignored by the baseline.

**Step 1 — Compute sub_code-level statistics (20 min):**
```python
# Expanding mean and ETS for each observed sub_code
train_sorted = train.sort_values('ts_index')

sub_code_stats = (train_sorted.groupby(['code','sub_code','sub_category','horizon'])
    ['target'].agg(['mean','std','count']).reset_index())

# Exponential smoothing per sub_code
from scipy.optimize import minimize_scalar

def fit_alpha(series):
    if len(series) < 3:
        return 0.1  # default for sparse series
    def mse(alpha):
        ewm = series.ewm(alpha=alpha, adjust=False).mean().shift(1).dropna()
        return ((series.iloc[1:] - ewm)**2).mean()
    return minimize_scalar(mse, bounds=(0.01, 0.99), method='bounded').x
```

**Step 2 — Credibility blending (15 min):**
```python
K = 30  # tune on validation
def warm_start_blend(row, sub_code_pred, group_pred, sub_code_count):
    w = sub_code_count / (sub_code_count + K)
    return w * sub_code_pred + (1 - w) * group_pred

# Apply to warm-start rows only; cold-start rows keep group_pred
```

**Step 3 — Tune K across validation folds (30 min):**
```python
from scipy.optimize import minimize_scalar

def score_K(K):
    blended = []
    for fold in folds:
        warm_mask = fold.is_warm_start
        blended_fold = fold.group_pred.copy()
        n = fold.sub_code_count[warm_mask]
        w = n / (n + K)
        blended_fold[warm_mask] = w * fold.sub_code_pred[warm_mask] + \
                                   (1-w) * fold.group_pred[warm_mask]
        blended.append(blended_fold)
    return compute_metric(pd.concat(blended))

best_K = minimize_scalar(score_K, bounds=(1, 200), method='bounded').x
```

**Expected improvement: 15–30% on warm-start slice → ~2–4% aggregate.** Implementation time: ~1.5 hours.

### Branch 3: Horizon-specific models with metric-aligned weights (surgical improvement)

This branch targets the worst-performing slices: h=1 and h=3 at specific code/sub_category combinations.

**Step 1 — Analyze per-horizon performance (30 min):**
```python
# Identify which horizons and groups have SS > 0 (improvable)
for h in horizons:
    mask = val['horizon'] == h
    score_h = compute_metric(val[mask])
    print(f"Horizon {h}: score={score_h:.4f}, n={mask.sum()}")
```

**Step 2 — Train horizon-specific models (1 hr):**
```python
models = {}
for h in horizons:
    train_h = train[train['horizon'] == h]
    val_h = val[val['horizon'] == h]
    
    # More aggressive smoothing for short horizons
    params_h = params.copy()
    if h <= 3:
        params_h['min_data_in_leaf'] = 50  # more regularization
        params_h['num_leaves'] = 15        # simpler trees
    
    models[h] = lgb.train(params_h, lgb.Dataset(X_train_h, y_norm_h),
                          num_boost_round=500,
                          valid_sets=[lgb.Dataset(X_val_h, y_norm_val_h)],
                          callbacks=[lgb.early_stopping(30)])
```

**Step 3 — Set metric-aligned sample weights (15 min):**
```python
# Weight each sample by its group's importance in the competition metric
# Higher weight → model focuses more on that group
group_weights = compute_competition_weights(train)  # from metric spec
baseline_energy = train.groupby(group_cols)['target'].transform('var')

sample_weight = group_weights / np.maximum(baseline_energy, 1e-6)
dtrain = lgb.Dataset(X_train, y_train, weight=sample_weight)
```

**Expected improvement: 5–15% on short-horizon slices, 3–8% aggregate.** Implementation time: ~2 hours.

### Sequencing and total timeline

Execute Branch 1 first (it provides the EB-shrunk baseline that Branches 2 and 3 build on). Then Branch 2 (warm-start overlay, fast to add). Then Branch 3 (horizon-specific tuning). Total implementation time: **~7 hours**. After all three branches, blend predictions using `scipy.optimize.minimize` with Nelder-Mead on the concatenated out-of-fold predictions from your 4 sequential folds. The expected aggregate improvement over the current grouped-mean baseline is **15–35%**, with the largest gains coming from EB shrinkage + group normalization (Branch 1) and the warm-start overlay providing reliable incremental improvement on the 11% seen rows.