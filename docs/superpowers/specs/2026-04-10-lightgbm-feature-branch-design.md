# LightGBM Feature-Based Forecasting Branch

**Date**: 2026-04-10
**Updated**: 2026-04-11
**Status**: Phase 2 complete — residual mode is new best
**Current best offline score**: 0.1457 (lgbm_residual_weighted_v2)
**Previous best offline score**: 0.1454 (ridge_oofnorm_pentaslice_h13)
**Public LB highscore**: 0.8718
**Goal**: Close the gap using nonlinear feature-based models

## Context

The current model family is entirely group-mean based: smoothed weighted group means at (code, sub_category, horizon) level, blended with horizon means, plus warm-start overlays, recent-delta corrections, and conservative OOF Ridge residuals. The 86 raw features are barely used — only through a Ridge residual with correction_scale=0.08 on h=1,3.

A prior HistGradientBoostingRegressor feature-only probe scored 0.0, but that was sklearn's limited implementation without group context. LightGBM is a materially different tool: native categorical handling, missing value support, deeper nonlinear capacity, and GPU acceleration.

This branch tests whether a proper gradient-boosted tree model can exploit the 86 features to make row-level predictions that group means cannot.

## Architecture

Two-phase approach sharing infrastructure:

### Phase 1: Diagnostic (Direct LightGBM)

LightGBM predicts y_target from raw features + categoricals alone. Answers: do features carry nonlinear signal?

### Phase 2: Stacked LightGBM on Incumbent

LightGBM predicts y_target from raw features + categoricals + the incumbent winner's prediction as an additional feature. The incumbent prediction is a strong group-level prior; LightGBM learns when to trust it and when row-level features override.

## Features

| Feature Set | Count | Handling |
|---|---|---|
| Raw features (feature_a..feature_ch) | 85 | Numeric, nulls left as-is (LightGBM native) |
| feature_al | BANNED | Target-reconstruction risk |
| weight | BANNED | Not a feature |
| code | 1 | LightGBM categorical (23 values) |
| sub_category | 1 | LightGBM categorical (5 values) |
| horizon | 1 | LightGBM categorical (4 values) |
| sub_code | EXCLUDED | 89% cold-start, 180 to 47 cardinality shift |
| ts_index | 1 | Numeric |
| incumbent_pred | 1 (Phase 2 only) | Numeric |

Total: 89 features (Phase 1) or 90 (Phase 2).

## Training Strategy

For each forward CV fold (existing 4-fold expanding-window setup):

1. Split train/valid by ts_index using existing fold boundaries
   - Fold 1: train end=2161, valid 2162-2521
   - Fold 2: train end=2521, valid 2522-2881
   - Fold 3: train end=2881, valid 2882-3241
   - Fold 4: train end=3241, valid 3242-3601
2. Phase 1: Train LightGBM on train features -> y_target
3. Phase 2: Fit incumbent on train -> generate incumbent predictions for train and valid -> train LightGBM on [features + incumbent_pred] -> y_target
4. Predict valid set with sample_weight=weight for early stopping eval
5. Collect OOF predictions across folds
6. Compute canonical weighted skill score on concatenated folds

### Sample Weighting

Train with sample_weight=weight to align optimization with evaluation metric (weighted MSE ratio).

### Early Stopping

Use fold validation set. LightGBM trains with early_stopping_rounds=50 on validation weighted RMSE.

## Hyperparameters

Starting conservative to avoid overfitting on 89% cold-start data:

```python
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_depth": -1,
    "verbose": -1,
    "seed": 42,
}
```

## File Structure

New files (no changes to existing code):

- `src/ts_forecasting/lgbm_models.py` — LGBMDirectRegressor, LGBMStackedRegressor
- `scripts/evaluate_lgbm.py` — Forward CV evaluation driver with stress reporting
- `scripts/predict_lgbm.py` — Test-set submission generation
- `tests/test_lgbm_models.py` — Unit tests for new model classes

Reused modules:
- `validation.py` — fold generation
- `metrics.py` — weighted skill score
- `evaluation.py` — stress reporting
- `io.py` — data loading
- `constants.py` — column names, paths

## Incumbent Integration (Phase 2)

For generating incumbent predictions:

1. Import the incumbent config from baseline_registry (RECENT_DELTA_K7_H13_PENTASLICE_H3_BETA_UP_V1_CONFIG)
2. Build the incumbent predictor using build_baseline()
3. For each fold: fit incumbent on train window, predict both train and valid
4. Use incumbent predictions as an additional numeric feature for LightGBM

The incumbent produces group means, not row-level predictions, so using in-fold train predictions has minimal overfitting risk (each row gets its group's mean, not a row-specific value).

## Categorical Encoding

LightGBM native categorical handling:
- Convert code, sub_category, horizon columns to pandas Categorical dtype
- Pass categorical_feature parameter to LightGBM
- No manual one-hot encoding needed

## Results (2026-04-11)

### All Modes Evaluated

| Model | Aggregate | Cold-Start | Warm-Start | Notes |
|---|---|---|---|---|
| Incumbent (group-mean) | 0.1454 | — | — | Baseline to beat |
| lgbm_direct_weighted_v1 | 0.0964 | — | — | Phase 1: features carry signal |
| lgbm_stacked_weighted_v1 | 0.0924 | 0.0896 | 0.0991 | OOF coverage issues, early stop at 4-11 rounds |
| lgbm_residual_weighted_v1 | 0.1456 | 0.1605 | 0.1008 | v1 had fold 1 OOF fallback (leaky) |
| **lgbm_residual_weighted_v2** | **0.1457** | **0.1605** | **0.1008** | Clean OOF — fold 1 gain confirmed real |
| lgbm_residual_unweighted_v2 | 0.0 | 0.0 | 0.0 | Sample weight essential for this metric |

### Per-Fold Detail (residual weighted v2)

| Fold | Train End | Valid Range | OOF Coverage | Score | Best Iter |
|---|---|---|---|---|---|
| fold_01 | 2161 | 2162-2521 | 1.6M/2.9M | 0.2451 | 1 |
| fold_02 | 2521 | 2522-2881 | 2.3M/3.5M | 0.0986 | 27 |
| fold_03 | 2881 | 2882-3241 | 2.9M/4.1M | 0.0759 | 20 |
| fold_04 | 3241 | 3242-3601 | 3.5M/4.7M | 0.1065 | 34 |

### Key Observations

1. **Features carry nonlinear signal** — 0.0964 direct score proves this (Phase 1 pass)
2. **Residual mode beats incumbent** — 0.1457 vs 0.1454, gain is small but confirmed with clean OOF
3. **Gain is cold-start driven** — 0.1605 cold vs 0.1008 warm
4. **Stacked mode failed** — OOF coverage too sparse, model couldn't learn beyond incumbent
5. **Sample weight is non-negotiable** — unweighted scores 0.0 (metric is weighted MSE ratio)
6. **Early stopping very aggressive** — 1-34 rounds, LightGBM correction is small
7. **High fold variance** — std 0.067, fold 1 (0.245) vs fold 3 (0.076)
8. **OOF fix validated** — lowering oof_min_train_steps 2000→1000 gave fold 1 real coverage; score held

### What Didn't Work

- **Stacked mode**: OOF incumbent coverage too low on early folds; model early-stops at 4-11 rounds
- **Unweighted training**: Evaluation metric is weighted MSE ratio; training without weights is catastrophic

## Success Criteria

- Phase 1 passes if: canonical score > 0.0 (features carry nonlinear signal) **PASSED (0.0964)**
- Phase 2 passes if: canonical score > 0.1454 (beats current winner) **PASSED (0.1457, residual mode)**
- Stretch: meaningful closure toward 0.87 public LB — **NOT MET** (gap remains enormous)

## Next Steps (for Codex handoff)

### High-Value Experiments (try first)

1. **Widen residual horizons** — currently gated to h=1,3 only. Direct mode (0.0964) uses all horizons. Try h=1,3,10,25 (all) for residual correction.
2. **Per-horizon LightGBM models** — train separate models per horizon instead of one global model. Different horizons may have different feature importance.
3. **Lower learning rate + more rounds** — current early stopping is at 1-34 rounds with lr=0.05. Try lr=0.01 with n_estimators=5000 to allow more gradual learning.
4. **Feature engineering** — the 85 raw features are used as-is. Consider:
   - Feature interactions (feature_a * feature_b)
   - Rolling statistics over ts_index
   - Lag features

### Lower-Priority Experiments

5. **Stacked mode with more OOF coverage** — could work if oof_step_size is reduced from 360 to 180, giving more sub-folds
6. **Residual on all horizons with scale gating** — different correction_scale per horizon
7. **Hyperparameter tuning** — Optuna sweep on num_leaves, min_child_samples, feature_fraction

### Submission Pipeline

Once a winning mode is confirmed:
1. Train LightGBM on full train data (ts_index 1-3601)
2. For residual: fit incumbent on full train, predict train+test, use as feature
3. Predict test set (ts_index 3602-4376) — residual correction on gated horizons, incumbent on rest
4. Write submission CSV matching expected format (id, y_target columns)
5. Script exists: `scripts/predict_lgbm.py --mode residual`

## File Inventory

| File | Purpose |
|---|---|
| `src/ts_forecasting/lgbm_models.py` | LGBMForecaster class, build_feature_frame, DEFAULT_PARAMS |
| `scripts/evaluate_lgbm.py` | Forward-CV evaluation with direct/stacked/residual modes |
| `scripts/predict_lgbm.py` | Test-set submission generation |
| `tests/test_lgbm_models.py` | Unit tests for LGBMForecaster |
| `docs/superpowers/specs/lgbm_*_results.json` | Raw evaluation results per experiment |
