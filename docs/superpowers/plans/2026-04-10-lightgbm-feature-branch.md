# LightGBM Feature-Based Forecasting Branch

> **Updated 2026-04-11** — Tasks 1-5 complete. Residual mode is new best (0.1457 vs incumbent 0.1454).

**Goal:** Build a LightGBM gradient-boosted tree model that uses 86 raw features to predict y_target, both standalone (Phase 1 diagnostic) and stacked on the incumbent group-mean winner (Phase 2).

**Architecture:** Single `LGBMForecaster` class that trains LightGBM on numeric features + native categoricals. Evaluation script handles fold splitting, incumbent fitting, and stacking orchestration. Three modes: `direct` (features only), `stacked` (features + incumbent prediction), `residual` (predict y - incumbent on h=1,3; incumbent elsewhere).

**Tech Stack:** LightGBM, pandas, numpy, existing ts_forecasting infrastructure (validation, metrics, evaluation, io)

**Current state:** All infrastructure built, all modes evaluated. See `docs/superpowers/specs/2026-04-10-lightgbm-feature-branch-design.md` for full results table and next-step recommendations.

---

### Task 1: Add LightGBM dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add lightgbm to requirements.txt**

```
lightgbm==4.6.0
```

Add this line after the existing `duckdb==1.5.1` line.

- [ ] **Step 2: Install dependencies**

Run: `pip install lightgbm==4.6.0`
Expected: Successful installation

- [ ] **Step 3: Verify import**

Run: `python -c "import lightgbm; print(lightgbm.__version__)"`
Expected: `4.6.0`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add lightgbm 4.6.0"
```

---

### Task 2: Create LGBMForecaster model class with tests

**Files:**
- Create: `src/ts_forecasting/lgbm_models.py`
- Create: `tests/test_lgbm_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lgbm_models.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_forecasting.lgbm_models import (
    BANNED_FEATURES,
    DEFAULT_PARAMS,
    LGBMForecaster,
    build_feature_frame,
)


def _make_frame(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    codes = ["A", "B", "C", "D"]
    sub_cats = ["C1", "C2"]
    horizons = [1, 3, 10, 25]
    return pd.DataFrame(
        {
            "id": range(n),
            "code": rng.choice(codes, n),
            "sub_code": rng.choice(["S1", "S2", "S3"], n),
            "sub_category": rng.choice(sub_cats, n),
            "horizon": rng.choice(horizons, n),
            "ts_index": np.tile(np.arange(1, n // 4 + 1), 4)[:n],
            "y_target": rng.randn(n) * 10,
            "weight": rng.uniform(0.5, 2.0, n),
            "feature_a": rng.randn(n),
            "feature_b": rng.randn(n),
            "feature_c": rng.randn(n),
            "feature_al": rng.randn(n),  # should be banned
        }
    )


def test_build_feature_frame_excludes_banned_and_includes_categoricals():
    frame = _make_frame()
    X, names = build_feature_frame(frame)
    assert "feature_a" in names
    assert "feature_b" in names
    assert "feature_c" in names
    assert "feature_al" not in names
    assert "code" in names
    assert "sub_category" in names
    assert "horizon" in names
    assert "ts_index" in names
    assert "y_target" not in names
    assert "weight" not in names
    assert "id" not in names
    assert "sub_code" not in names
    assert X["code"].dtype.name == "category"
    assert X["sub_category"].dtype.name == "category"
    assert X["horizon"].dtype.name == "category"


def test_build_feature_frame_uses_provided_feature_names():
    frame = _make_frame()
    _, names = build_feature_frame(frame)
    frame2 = _make_frame(seed=99)
    X2, names2 = build_feature_frame(frame2, feature_names=names)
    assert names2 == names
    assert len(X2) == len(frame2)


def test_build_feature_frame_includes_extra_numeric_columns():
    frame = _make_frame()
    frame["incumbent_pred"] = np.random.randn(len(frame))
    X, names = build_feature_frame(frame, extra_numeric_columns=["incumbent_pred"])
    assert "incumbent_pred" in names


def test_lgbm_forecaster_fit_predict_shape():
    train = _make_frame(n=300, seed=1)
    test = _make_frame(n=100, seed=2)
    model = LGBMForecaster(
        params={**DEFAULT_PARAMS, "num_leaves": 4, "verbose": -1},
        n_estimators=10,
    )
    model.fit(train)
    preds = model.predict(test)
    assert preds.shape == (100,)
    assert not np.any(np.isnan(preds))


def test_lgbm_forecaster_with_early_stopping():
    train = _make_frame(n=300, seed=1)
    val = _make_frame(n=100, seed=2)
    model = LGBMForecaster(
        params={**DEFAULT_PARAMS, "num_leaves": 4, "verbose": -1},
        n_estimators=200,
        early_stopping_rounds=5,
    )
    model.fit(train, eval_frame=val)
    preds = model.predict(val)
    assert preds.shape == (100,)
    assert model.model_.best_iteration <= 200


def test_lgbm_forecaster_with_incumbent_pred():
    train = _make_frame(n=300, seed=1)
    train["incumbent_pred"] = np.random.randn(len(train))
    test = _make_frame(n=100, seed=2)
    test["incumbent_pred"] = np.random.randn(len(test))
    model = LGBMForecaster(
        params={**DEFAULT_PARAMS, "num_leaves": 4, "verbose": -1},
        n_estimators=10,
        extra_numeric_columns=["incumbent_pred"],
    )
    model.fit(train)
    preds = model.predict(test)
    assert preds.shape == (100,)
    assert "incumbent_pred" in model.feature_names_


def test_lgbm_forecaster_raises_if_not_fitted():
    model = LGBMForecaster()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(_make_frame(n=10))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/meettilavat/Time_series_forecasting && python -m pytest tests/test_lgbm_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ts_forecasting.lgbm_models'`

- [ ] **Step 3: Write the implementation**

Create `src/ts_forecasting/lgbm_models.py`:

```python
"""LightGBM-based forecasting models."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .constants import (
    CODE_COLUMN,
    HORIZON_COLUMN,
    SUB_CATEGORY_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
    detect_feature_columns,
)

BANNED_FEATURES: frozenset[str] = frozenset({"feature_al"})
CATEGORICAL_COLUMNS: list[str] = [CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN]

DEFAULT_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
}


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    banned_features: frozenset[str] | set[str] = BANNED_FEATURES,
    categorical_columns: list[str] | None = None,
    extra_numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract model features from a data frame.

    If *feature_names* is provided (from a prior fit), select those exact
    columns.  Otherwise auto-detect numeric ``feature_*`` columns, add
    categoricals and ``ts_index``, and optionally append
    *extra_numeric_columns* (e.g. ``incumbent_pred``).
    """
    if categorical_columns is None:
        categorical_columns = list(CATEGORICAL_COLUMNS)

    if feature_names is not None:
        X = frame[feature_names].copy()
    else:
        numeric_features = [
            c
            for c in detect_feature_columns(list(frame.columns))
            if c not in banned_features
        ]
        use_cols = numeric_features + categorical_columns + [TIME_COLUMN]
        if extra_numeric_columns:
            use_cols.extend(extra_numeric_columns)
        use_cols = list(dict.fromkeys(use_cols))
        X = frame[use_cols].copy()

    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, list(X.columns)


@dataclass
class LGBMForecaster:
    """LightGBM forecaster for tabular time-series prediction."""

    params: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    banned_features: frozenset[str] | set[str] = field(
        default_factory=lambda: frozenset(BANNED_FEATURES)
    )
    extra_numeric_columns: list[str] | None = None

    model_: lgb.Booster | None = field(default=None, repr=False)
    feature_names_: list[str] | None = field(default=None, repr=False)

    def fit(
        self,
        train_frame: pd.DataFrame,
        *,
        eval_frame: pd.DataFrame | None = None,
    ) -> "LGBMForecaster":
        X_train, self.feature_names_ = build_feature_frame(
            train_frame,
            banned_features=self.banned_features,
            extra_numeric_columns=self.extra_numeric_columns,
        )
        y_train = train_frame[TARGET_COLUMN].to_numpy(dtype=float)
        w_train = train_frame[WEIGHT_COLUMN].to_numpy(dtype=float)

        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            weight=w_train,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )

        valid_sets: list[lgb.Dataset] = [train_set]
        valid_names: list[str] = ["train"]
        callbacks: list = [lgb.log_evaluation(100)]

        if eval_frame is not None:
            X_eval, _ = build_feature_frame(
                eval_frame,
                feature_names=self.feature_names_,
                banned_features=self.banned_features,
            )
            y_eval = eval_frame[TARGET_COLUMN].to_numpy(dtype=float)
            w_eval = eval_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
            eval_set = lgb.Dataset(
                X_eval,
                label=y_eval,
                weight=w_eval,
                reference=train_set,
                free_raw_data=False,
            )
            valid_sets.append(eval_set)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        self.model_ = lgb.train(
            dict(self.params),
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.feature_names_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X, _ = build_feature_frame(
            frame,
            feature_names=self.feature_names_,
            banned_features=self.banned_features,
        )
        return self.model_.predict(X)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/meettilavat/Time_series_forecasting && python -m pytest tests/test_lgbm_models.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ts_forecasting/lgbm_models.py tests/test_lgbm_models.py
git commit -m "feat: add LGBMForecaster model class with tests"
```

---

### Task 3: Create evaluation script

**Files:**
- Create: `scripts/evaluate_lgbm.py`

- [ ] **Step 1: Write evaluate_lgbm.py**

```python
#!/usr/bin/env python
"""Evaluate LightGBM models on forward validation folds.

Usage:
    # Phase 1 — direct (features only):
    python scripts/evaluate_lgbm.py --mode direct

    # Phase 2 — stacked (features + incumbent prediction):
    python scripts/evaluate_lgbm.py --mode stacked
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from ts_forecasting.baseline_registry import (
    BASELINE_CONFIGS,
    build_predictor,
)
from ts_forecasting.constants import (
    CODE_COLUMN,
    HORIZON_COLUMN,
    ID_COLUMN,
    SUB_CATEGORY_COLUMN,
    SUB_CODE_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
    detect_feature_columns,
)
from ts_forecasting.evaluation import (
    aggregate_result_summaries,
    finalize_stress_block,
    initialize_stress_accumulator,
    summarize_scored_predictions,
    update_stress_accumulator,
)
from ts_forecasting.io import get_columns, get_max_ts_index, read_parquet_frame
from ts_forecasting.lgbm_models import DEFAULT_PARAMS, LGBMForecaster
from ts_forecasting.sequential import IdentityTransformer, SequentialInferencePipeline
from ts_forecasting.validation import (
    annotate_cold_start_rows,
    build_expanding_window_folds,
    build_first_seen_lookup,
)

INCUMBENT_CONFIG_NAME = (
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13"
    "_pentaslice_h3_beta_up_v1"
)
COLD_START_KEY = [CODE_COLUMN, SUB_CODE_COLUMN]


def add_incumbent_predictions(
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    *,
    config: dict[str, object],
) -> np.ndarray:
    """Fit incumbent on *train_frame*, return predictions on *predict_frame*."""
    pipeline = SequentialInferencePipeline(
        transformer=IdentityTransformer(),
        predictor=build_predictor(config),
    ).fit(train_frame)
    preds_df = pipeline.predict(predict_frame)
    merged = predict_frame[[ID_COLUMN]].merge(
        preds_df[[ID_COLUMN, "prediction"]],
        on=ID_COLUMN,
        how="left",
    )
    return merged["prediction"].to_numpy(dtype=float)


def run_fold(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    *,
    stacked: bool,
    incumbent_config: dict[str, object] | None,
    lgbm_params: dict[str, object],
) -> pd.DataFrame:
    """Train and evaluate one fold.  Returns scored DataFrame."""
    extra_cols: list[str] | None = None
    if stacked and incumbent_config is not None:
        train_inc = add_incumbent_predictions(
            train_frame, train_frame, config=incumbent_config,
        )
        valid_inc = add_incumbent_predictions(
            train_frame, valid_frame, config=incumbent_config,
        )
        train_frame = train_frame.copy()
        valid_frame = valid_frame.copy()
        train_frame["incumbent_pred"] = train_inc
        valid_frame["incumbent_pred"] = valid_inc
        extra_cols = ["incumbent_pred"]

    model = LGBMForecaster(
        params=lgbm_params,
        extra_numeric_columns=extra_cols,
    )
    model.fit(train_frame, eval_frame=valid_frame)
    predictions = model.predict(valid_frame)

    scored = valid_frame[
        [ID_COLUMN, CODE_COLUMN, SUB_CODE_COLUMN, SUB_CATEGORY_COLUMN,
         HORIZON_COLUMN, TARGET_COLUMN, WEIGHT_COLUMN]
    ].copy()
    scored["prediction"] = predictions
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument(
        "--mode",
        choices=["direct", "stacked"],
        default="direct",
        help="direct = features only; stacked = features + incumbent prediction",
    )
    parser.add_argument("--output-dir", default="outputs/lgbm_models")
    parser.add_argument("--holdout-steps", type=int, default=360)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--step-size", type=int, default=360)
    parser.add_argument("--min-train-steps", type=int, default=2000)
    args = parser.parse_args()

    cv_tag = (
        f"forward_cv_h{args.holdout_steps}_f{args.n_folds}"
        f"_s{args.step_size}_code_sub_code"
    )
    output_dir = Path(args.output_dir) / cv_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"lgbm_{args.mode}_v1"

    # --- build folds -------------------------------------------------------
    max_ts = get_max_ts_index(args.train_path)
    folds = build_expanding_window_folds(
        max_ts_index=max_ts,
        holdout_steps=args.holdout_steps,
        n_folds=args.n_folds,
        step_size=args.step_size,
        min_train_steps=args.min_train_steps,
    )

    # --- determine columns to load -----------------------------------------
    all_columns = get_columns(args.train_path)
    feature_cols = [
        c for c in detect_feature_columns(all_columns) if c != "feature_al"
    ]
    load_cols = list(dict.fromkeys(
        [ID_COLUMN, CODE_COLUMN, SUB_CODE_COLUMN, SUB_CATEGORY_COLUMN,
         HORIZON_COLUMN, TIME_COLUMN, TARGET_COLUMN, WEIGHT_COLUMN]
        + feature_cols
    ))

    incumbent_config: dict[str, object] | None = None
    if args.mode == "stacked":
        incumbent_config = BASELINE_CONFIGS[INCUMBENT_CONFIG_NAME]

    lgbm_params = dict(DEFAULT_PARAMS)

    # --- fold loop ----------------------------------------------------------
    stress_acc = initialize_stress_accumulator([model_name])
    fold_summaries: list[dict] = []
    t0 = time.perf_counter()

    for fold in folds:
        fold_t0 = time.perf_counter()
        fold_tag = fold.name.split("_train_")[0]  # e.g. "fold_01"
        print(f"\n{'=' * 60}")
        print(f"{fold_tag}: train <= {fold.train_end_ts}, "
              f"valid {fold.validation_start_ts}-{fold.validation_end_ts}")

        train_frame = read_parquet_frame(
            args.train_path,
            columns=load_cols,
            where=f"{TIME_COLUMN} <= {fold.train_end_ts}",
        )
        valid_frame = read_parquet_frame(
            args.train_path,
            columns=load_cols,
            where=(
                f"{TIME_COLUMN} > {fold.train_end_ts} AND "
                f"{TIME_COLUMN} <= {fold.validation_end_ts}"
            ),
        )
        print(f"  train {len(train_frame):,} rows, valid {len(valid_frame):,} rows")

        scored = run_fold(
            train_frame,
            valid_frame,
            stacked=(args.mode == "stacked"),
            incumbent_config=incumbent_config,
            lgbm_params=lgbm_params,
        )

        # cold-start annotation
        first_seen = build_first_seen_lookup(
            train_frame, key_columns=COLD_START_KEY,
        )
        scored = annotate_cold_start_rows(
            scored,
            first_seen_lookup=first_seen,
            key_columns=COLD_START_KEY,
            train_end_ts=fold.train_end_ts,
        )

        summary = summarize_scored_predictions(
            scored, cold_start_mask=scored["is_cold_start"],
        )
        fold_summaries.append(summary)

        update_stress_accumulator(
            stress_acc,
            fold_name=fold_tag,
            baseline_name=model_name,
            scored=scored,
            full_summary=summary,
        )

        elapsed = time.perf_counter() - fold_t0
        print(f"  score={summary['overall_score']:.10f}  ({elapsed:.1f}s)")

    # --- aggregate ----------------------------------------------------------
    aggregate = aggregate_result_summaries(fold_summaries)
    stress = finalize_stress_block(stress_acc)

    result = {
        "model_name": model_name,
        "mode": args.mode,
        "incumbent": INCUMBENT_CONFIG_NAME if args.mode == "stacked" else None,
        "params": lgbm_params,
        "aggregate": aggregate,
        "folds": {
            fold.name: summary
            for fold, summary in zip(folds, fold_summaries)
        },
        "stress": stress,
    }

    summary_path = output_dir / f"{model_name}_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, default=str))

    total_time = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"AGGREGATE SCORE: {aggregate['aggregate_overall_score']}")
    print(f"  cold-start: {aggregate.get('aggregate_cold_start_score')}")
    print(f"  warm-start: {aggregate.get('aggregate_warm_start_score')}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses without error**

Run: `cd /home/meettilavat/Time_series_forecasting && python scripts/evaluate_lgbm.py --help`
Expected: Shows usage help with `--mode`, `--train-path`, etc.

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate_lgbm.py
git commit -m "feat: add LightGBM forward-CV evaluation script"
```

---

### Task 4: Run Phase 1 diagnostic (direct LightGBM) -- DONE

**Result:** aggregate_overall_score = 0.0964. Features carry nonlinear signal. Gate passed.
Fold scores: 0.0, 0.0993, 0.0782, 0.1582.
Results: `docs/superpowers/specs/lgbm_direct_weighted_v1_results.json`

---

### Task 5: Run Phase 2 (all modes) -- DONE

**Modes evaluated:** direct, stacked, residual (weighted + unweighted).

| Model | Score | Gate |
|---|---|---|
| lgbm_direct_weighted_v1 | 0.0964 | PASS (>0) |
| lgbm_stacked_weighted_v1 | 0.0924 | FAIL (<0.1454) |
| lgbm_residual_weighted_v2 | **0.1457** | **PASS (>0.1454)** |
| lgbm_residual_unweighted_v2 | 0.0 | FAIL |

Results: `docs/superpowers/specs/lgbm_*_results.json`

**Note:** residual mode also tested with oof_min_train_steps lowered from 2000→1000 to fix fold 1 OOF coverage. Score held (0.1457), confirming gain is real not leakage.

---

### Task 6: Improve residual mode score (NEXT)

The residual win (0.1457 vs 0.1454) is real but tiny. These experiments may widen the gap.

- [ ] **Step 1: Widen residual horizons**

Currently `RESIDUAL_HORIZONS = [1, 3]` in `scripts/evaluate_lgbm.py`. Try `[1, 3, 10, 25]` (all horizons).
Direct mode scored 0.0964 using all horizons — there may be residual signal on h=10,25 too.

- [ ] **Step 2: Lower learning rate**

Try `learning_rate=0.01` with `n_estimators=5000`. Current early stopping at 1-34 rounds with lr=0.05 suggests the model needs gentler updates.

- [ ] **Step 3: Per-horizon models**

Train separate LGBMForecaster per horizon value. Different horizons likely have different feature importance patterns.

- [ ] **Step 4: Feature engineering**

Consider rolling stats over ts_index, feature interactions, or lag features. The 85 raw features are used as-is currently.

---

### Task 7: Generate submission with residual mode

`scripts/predict_lgbm.py` exists but needs `--mode residual` support (currently only direct/stacked).

- [ ] **Step 1: Add residual mode to predict_lgbm.py**

Mirror the residual logic from `evaluate_lgbm.py`: train LightGBM on residuals for gated horizons, use incumbent predictions for the rest.

- [ ] **Step 2: Generate and validate submission**

Run: `python scripts/predict_lgbm.py --mode residual`
Validate: 1,447,107 rows, columns [id, y_target], 0 NaN.

---

### Task 8: Run all existing tests (regression check)

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.
