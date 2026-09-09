"""LightGBM-based forecasting models."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .constants import (
    CODE_COLUMN,
    HORIZON_COLUMN,
    SUB_CATEGORY_COLUMN,
    SUB_CODE_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
    detect_feature_columns,
)

BANNED_FEATURES: frozenset[str] = frozenset({"feature_al"})
CATEGORICAL_COLUMNS: list[str] = [CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN]
AUTO_NUM_THREADS_CAP = 64
AUTO_COL_WISE_THREAD_THRESHOLD = 32


def infer_auto_num_threads(
    logical_cpu_count: int | None = None,
    *,
    cap: int = AUTO_NUM_THREADS_CAP,
) -> int:
    """Choose a conservative LightGBM thread count from host parallelism."""
    detected = logical_cpu_count if logical_cpu_count is not None else os.cpu_count()
    if detected is None:
        detected = 1
    return max(1, min(int(detected), int(cap)))


def build_runtime_lgbm_params(
    logical_cpu_count: int | None = None,
) -> dict[str, object]:
    """Return hardware-aware LightGBM defaults for the current host."""
    num_threads = infer_auto_num_threads(logical_cpu_count)
    params: dict[str, object] = {"num_threads": num_threads}
    if num_threads >= AUTO_COL_WISE_THREAD_THRESHOLD:
        # LightGBM recommends column-wise histograms on wide many-core hosts.
        params["force_col_wise"] = True
    return params

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
    **build_runtime_lgbm_params(),
}


def build_seen_sub_code_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Return the `(code, sub_code)` pairs present in the provided frame."""
    return set(
        zip(
            frame[CODE_COLUMN],
            frame[SUB_CODE_COLUMN],
            strict=True,
        )
    )


def _select_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    banned_features: frozenset[str] | set[str] = BANNED_FEATURES,
    categorical_columns: list[str] | None = None,
    extra_numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
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

    return X, list(X.columns), list(categorical_columns)


def _infer_category_maps(
    frame: pd.DataFrame,
    *,
    categorical_columns: list[str],
) -> dict[str, pd.CategoricalDtype]:
    out_maps: dict[str, pd.CategoricalDtype] = {}
    for col in categorical_columns:
        if col in frame.columns:
            out_maps[col] = frame[col].astype("category").dtype
    return out_maps


@dataclass(frozen=True)
class PreparedFeatureFrame:
    """Reusable prepared model inputs for a frame or row slice."""

    feature_frame: pd.DataFrame
    feature_names: list[str]
    category_maps: dict[str, pd.CategoricalDtype]
    banned_features: frozenset[str] | set[str] = field(
        default_factory=lambda: frozenset(BANNED_FEATURES)
    )
    categorical_columns: list[str] = field(
        default_factory=lambda: list(CATEGORICAL_COLUMNS)
    )

    def slice_rows(
        self,
        row_index: pd.Index | list[int] | np.ndarray,
    ) -> "PreparedFeatureFrame":
        """Return a row-aligned slice without reselecting the base columns."""
        sliced_frame = self.feature_frame.loc[row_index, self.feature_names].copy()
        return PreparedFeatureFrame(
            feature_frame=sliced_frame,
            feature_names=list(self.feature_names),
            category_maps=_infer_category_maps(
                sliced_frame,
                categorical_columns=self.categorical_columns,
            ),
            banned_features=self.banned_features,
            categorical_columns=list(self.categorical_columns),
        )

    def to_frame(
        self,
        *,
        category_maps: dict[str, pd.CategoricalDtype] | None = None,
    ) -> tuple[pd.DataFrame, list[str], dict[str, pd.CategoricalDtype]]:
        """Materialize a LightGBM-ready frame from the prepared base matrix."""
        X = self.feature_frame.loc[:, self.feature_names].copy()
        maps = self.category_maps if category_maps is None else category_maps
        out_maps: dict[str, pd.CategoricalDtype] = {}
        for col in self.categorical_columns:
            if col not in X.columns:
                continue
            if maps and col in maps:
                X[col] = X[col].astype(maps[col])
            else:
                X[col] = X[col].astype("category")
            out_maps[col] = X[col].dtype
        return X, list(X.columns), out_maps


def prepare_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    banned_features: frozenset[str] | set[str] = BANNED_FEATURES,
    categorical_columns: list[str] | None = None,
    extra_numeric_columns: list[str] | None = None,
    category_maps: dict[str, pd.CategoricalDtype] | None = None,
) -> PreparedFeatureFrame:
    """Prepare a reusable model-input representation from a raw frame."""
    feature_frame, selected_feature_names, categorical_columns = _select_feature_frame(
        frame,
        feature_names=feature_names,
        banned_features=banned_features,
        categorical_columns=categorical_columns,
        extra_numeric_columns=extra_numeric_columns,
    )
    inferred_maps = (
        category_maps
        if category_maps is not None
        else _infer_category_maps(
            feature_frame,
            categorical_columns=categorical_columns,
        )
    )
    return PreparedFeatureFrame(
        feature_frame=feature_frame,
        feature_names=selected_feature_names,
        category_maps=inferred_maps,
        banned_features=banned_features,
        categorical_columns=categorical_columns,
    )


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    banned_features: frozenset[str] | set[str] = BANNED_FEATURES,
    categorical_columns: list[str] | None = None,
    extra_numeric_columns: list[str] | None = None,
    category_maps: dict[str, pd.CategoricalDtype] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, pd.CategoricalDtype]]:
    """Extract model features from a data frame.

    Returns ``(feature_frame, feature_names, category_maps)``.

    *category_maps* from a prior ``fit`` call should be passed back at
    predict time so that train and eval/test share identical category
    codes — a requirement for correct LightGBM categorical splits.
    """
    prepared = prepare_feature_frame(
        frame,
        feature_names=feature_names,
        banned_features=banned_features,
        categorical_columns=categorical_columns,
        extra_numeric_columns=extra_numeric_columns,
        category_maps=category_maps,
    )
    return prepared.to_frame(category_maps=category_maps)


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
    use_sample_weight: bool = True

    model_: lgb.Booster | None = field(default=None, repr=False)
    feature_names_: list[str] | None = field(default=None, repr=False)
    category_maps_: dict[str, pd.CategoricalDtype] | None = field(
        default=None, repr=False
    )

    def fit(
        self,
        train_frame: pd.DataFrame,
        *,
        eval_frame: pd.DataFrame | None = None,
        prepared_train_frame: pd.DataFrame | None = None,
        prepared_eval_frame: pd.DataFrame | None = None,
    ) -> "LGBMForecaster":
        if prepared_train_frame is not None:
            X_train = prepared_train_frame.copy()
            self.feature_names_ = list(X_train.columns)
            self.category_maps_ = _infer_category_maps(
                X_train,
                categorical_columns=list(CATEGORICAL_COLUMNS),
            )
        else:
            X_train, self.feature_names_, self.category_maps_ = build_feature_frame(
                train_frame,
                banned_features=self.banned_features,
                extra_numeric_columns=self.extra_numeric_columns,
            )
        y_train = train_frame[TARGET_COLUMN].to_numpy(dtype=float)
        w_train = (
            train_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
            if self.use_sample_weight
            else None
        )

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
            if prepared_eval_frame is not None:
                X_eval = prepared_eval_frame.copy()
            else:
                X_eval, _, _ = build_feature_frame(
                    eval_frame,
                    feature_names=self.feature_names_,
                    banned_features=self.banned_features,
                    category_maps=self.category_maps_,
                )
            y_eval = eval_frame[TARGET_COLUMN].to_numpy(dtype=float)
            w_eval = (
                eval_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
                if self.use_sample_weight
                else None
            )
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

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        prepared_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if self.model_ is None or self.feature_names_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if prepared_frame is not None:
            X = prepared_frame.copy()
        else:
            X, _, _ = build_feature_frame(
                frame,
                feature_names=self.feature_names_,
                banned_features=self.banned_features,
                category_maps=self.category_maps_,
            )
        return self.model_.predict(X)
