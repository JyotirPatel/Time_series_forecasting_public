"""Residual-model helpers built on top of a base prediction."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    CODE_COLUMN,
    HORIZON_COLUMN,
    ID_COLUMN,
    SUB_CATEGORY_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
    detect_feature_columns,
)
from .sequential import IdentityTransformer, SequentialInferencePipeline
from .validation import build_expanding_window_folds

RESIDUAL_CATEGORICAL_COLUMNS = [CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN]
RESIDUAL_PRIMARY_SCALE_COLUMNS = [CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN]
RESIDUAL_FALLBACK_SCALE_COLUMNS = [SUB_CATEGORY_COLUMN, HORIZON_COLUMN]
RESIDUAL_SCALE_COLUMN = "residual_scale"
NORMALIZED_RESIDUAL_TARGET_COLUMN = "normalized_residual_target"
RAW_RESIDUAL_TARGET_COLUMN = "raw_residual_target"


def build_slice_mask(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] | list[int] | None = None,
    sub_categories: tuple[str, ...] | list[str] | None = None,
    codes: tuple[str, ...] | list[str] | None = None,
) -> pd.Series:
    """Return a boolean mask for the requested residual-training slice."""

    mask = pd.Series(True, index=frame.index)
    if horizons is not None:
        mask &= frame[HORIZON_COLUMN].isin(list(horizons))
    if sub_categories is not None:
        mask &= frame[SUB_CATEGORY_COLUMN].isin(list(sub_categories))
    if codes is not None:
        mask &= frame[CODE_COLUMN].isin(list(codes))
    return mask


def detect_residual_feature_columns(
    columns: list[str] | tuple[str, ...],
    *,
    banned_feature_columns: set[str] | None = None,
    include_time_column: bool = True,
) -> list[str]:
    """Return causal feature columns for residual models."""

    banned = set() if banned_feature_columns is None else set(banned_feature_columns)
    feature_columns = [
        column for column in detect_feature_columns(columns) if column not in banned
    ]
    if include_time_column:
        return [TIME_COLUMN, *feature_columns]
    return feature_columns


def build_residual_scale_lookup(
    frame: pd.DataFrame,
    *,
    level_columns: list[str],
    target_column: str = TARGET_COLUMN,
    weight_column: str = WEIGHT_COLUMN,
    scale_column: str = RESIDUAL_SCALE_COLUMN,
    min_scale: float = 1e-6,
) -> pd.DataFrame:
    """Estimate a weighted target RMS scale for each requested group."""

    if min_scale <= 0.0:
        raise ValueError("min_scale must be positive")

    required_columns = [*level_columns, target_column, weight_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing required columns for residual scale lookup: {missing_columns}")

    scale_frame = frame[required_columns].copy()
    scale_frame["_weighted_target_sq"] = (
        scale_frame[target_column].to_numpy(dtype=float) ** 2
    ) * scale_frame[weight_column].to_numpy(dtype=float)

    aggregated = (
        scale_frame.groupby(level_columns, dropna=False, observed=True)
        .agg(
            weighted_target_sq_sum=("_weighted_target_sq", "sum"),
            weight_sum=(weight_column, "sum"),
        )
        .reset_index()
    )
    aggregated[scale_column] = np.where(
        aggregated["weight_sum"].to_numpy(dtype=float) > 0.0,
        np.sqrt(
            aggregated["weighted_target_sq_sum"].to_numpy(dtype=float)
            / aggregated["weight_sum"].to_numpy(dtype=float)
        ),
        np.nan,
    )
    aggregated[scale_column] = aggregated[scale_column].clip(lower=min_scale)
    return aggregated[level_columns + [scale_column]]


def compute_global_residual_scale(
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    weight_column: str = WEIGHT_COLUMN,
    min_scale: float = 1e-6,
) -> float:
    """Estimate a global weighted target RMS scale."""

    if min_scale <= 0.0:
        raise ValueError("min_scale must be positive")

    required_columns = [target_column, weight_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing required columns for global residual scale: {missing_columns}")

    target = frame[target_column].to_numpy(dtype=float)
    weight = frame[weight_column].to_numpy(dtype=float)
    weight_sum = float(weight.sum())
    if weight_sum <= 0.0:
        return float(
            max(
                np.sqrt(np.mean(np.square(target))) if len(target) else min_scale,
                min_scale,
            )
        )
    weighted_target_sq_sum = float(np.sum(weight * np.square(target)))
    return float(max(np.sqrt(weighted_target_sq_sum / weight_sum), min_scale))


def attach_residual_scales(
    frame: pd.DataFrame,
    *,
    primary_lookup: pd.DataFrame,
    fallback_lookup: pd.DataFrame,
    global_scale: float,
    primary_level_columns: list[str] = RESIDUAL_PRIMARY_SCALE_COLUMNS,
    fallback_level_columns: list[str] = RESIDUAL_FALLBACK_SCALE_COLUMNS,
    scale_column: str = RESIDUAL_SCALE_COLUMN,
    min_scale: float = 1e-6,
) -> pd.DataFrame:
    """Attach prefix-only residual scales with coarse fallback and a global floor."""

    if min_scale <= 0.0:
        raise ValueError("min_scale must be positive")

    def align_lookup(
        lookup: pd.DataFrame,
        level_columns: list[str],
    ) -> pd.Series:
        lookup_indexed = lookup.set_index(level_columns)
        if not lookup_indexed.index.is_unique:
            raise ValueError(
                f"duplicate lookup keys for residual scale columns: {level_columns}"
            )
        lookup_series = lookup_indexed[scale_column]
        frame_keys = pd.MultiIndex.from_frame(frame[level_columns], names=level_columns)
        return pd.Series(
            lookup_series.reindex(frame_keys).to_numpy(dtype=float),
            index=frame.index,
        )

    primary_values = align_lookup(primary_lookup, primary_level_columns)
    fallback_values = align_lookup(fallback_lookup, fallback_level_columns)

    enriched = frame.copy()
    enriched[scale_column] = primary_values
    fill_mask = enriched[scale_column].isna() | (enriched[scale_column] <= min_scale)
    enriched.loc[fill_mask, scale_column] = fallback_values.loc[fill_mask]
    fill_mask = enriched[scale_column].isna() | (enriched[scale_column] <= min_scale)
    enriched.loc[fill_mask, scale_column] = max(float(global_scale), min_scale)
    enriched[scale_column] = enriched[scale_column].clip(lower=min_scale)
    return enriched


def compute_normalized_residual_targets(
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    base_prediction_column: str = "base_prediction",
    scale_column: str = RESIDUAL_SCALE_COLUMN,
    output_column: str = NORMALIZED_RESIDUAL_TARGET_COLUMN,
) -> pd.DataFrame:
    """Attach a residual target normalized by the prefix-only target scale."""

    required_columns = [target_column, base_prediction_column, scale_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"missing required columns for normalized residual targets: {missing_columns}"
        )

    normalized = frame.copy()
    normalized[output_column] = (
        normalized[target_column].to_numpy(dtype=float)
        - normalized[base_prediction_column].to_numpy(dtype=float)
    ) / normalized[scale_column].to_numpy(dtype=float)
    return normalized


def compute_raw_residual_targets(
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    base_prediction_column: str = "base_prediction",
    output_column: str = RAW_RESIDUAL_TARGET_COLUMN,
) -> pd.DataFrame:
    """Attach an unnormalized residual target."""

    required_columns = [target_column, base_prediction_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing required columns for raw residual targets: {missing_columns}")

    residuals = frame.copy()
    residuals[output_column] = (
        residuals[target_column].to_numpy(dtype=float)
        - residuals[base_prediction_column].to_numpy(dtype=float)
    )
    return residuals


def shrink_residual_corrections(
    raw_corrections: np.ndarray | pd.Series,
    *,
    correction_scale: float,
    clip_abs: float | None,
) -> np.ndarray:
    """Clip and shrink residual corrections toward zero."""

    if correction_scale < 0.0:
        raise ValueError("correction_scale must be non-negative")
    if clip_abs is not None and clip_abs < 0.0:
        raise ValueError("clip_abs must be non-negative")

    corrections = np.asarray(raw_corrections, dtype=float)
    if clip_abs is not None:
        corrections = np.clip(corrections, -clip_abs, clip_abs)
    return corrections * correction_scale


def apply_normalized_residual_corrections(
    raw_normalized_corrections: np.ndarray | pd.Series,
    *,
    scales: np.ndarray | pd.Series,
    correction_scale: float,
    clip_abs: float | None,
) -> np.ndarray:
    """Clip and shrink normalized corrections, then restore raw residual scale."""

    normalized_corrections = shrink_residual_corrections(
        raw_normalized_corrections,
        correction_scale=correction_scale,
        clip_abs=clip_abs,
    )
    return normalized_corrections * np.asarray(scales, dtype=float)


def _sample_frame(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    time_column: str = TIME_COLUMN,
    random_state: int = 42,
) -> pd.DataFrame:
    sort_columns = [time_column]
    if ID_COLUMN in frame.columns:
        sort_columns.append(ID_COLUMN)

    ordered = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if len(ordered) <= max_rows:
        return ordered
    return ordered.sample(n=max_rows, random_state=random_state).sort_values(
        sort_columns,
        kind="stable",
    )


def _max_expanding_window_folds(
    *,
    max_ts_index: int,
    holdout_steps: int,
    step_size: int,
    min_train_steps: int,
) -> int:
    latest_train_end = max_ts_index - holdout_steps
    if latest_train_end < min_train_steps:
        return 0
    return int(((latest_train_end - min_train_steps) // step_size) + 1)


def _build_raw_residual_model(
    *,
    model_type: str,
    numeric_columns: list[str],
    alpha: float,
    epsilon: float,
    max_iter: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                RESIDUAL_CATEGORICAL_COLUMNS,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ],
        sparse_threshold=0.0,
    )

    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "huber":
        model = HuberRegressor(
            alpha=alpha,
            epsilon=epsilon,
            max_iter=max_iter,
        )
    elif model_type == "hist_gbm":
        model = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
    else:
        raise ValueError(f"unsupported residual model type: {model_type}")

    return Pipeline([("pre", preprocessor), ("model", model)])


@dataclass
class FixedPrefixResidualCorrectionRegressor:
    """Train a residual model on a recent slice using prefix-only pseudo-history."""

    base_predictor: object
    model_type: str = "ridge"
    horizons: tuple[int, ...] | list[int] | None = None
    sub_categories: tuple[str, ...] | list[str] | None = None
    codes: tuple[str, ...] | list[str] | None = None
    residual_train_steps: int = 720
    correction_scale: float = 0.22
    correction_clip_quantile: float = 0.9
    max_train_rows: int = 180_000
    alpha: float = 10.0
    epsilon: float = 1.35
    max_iter: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 200
    banned_feature_columns: set[str] = field(default_factory=set)
    time_column: str = TIME_COLUMN
    target_column: str = TARGET_COLUMN
    weight_column: str = WEIGHT_COLUMN
    random_state: int = 42
    final_base_predictor_: object | None = None
    residual_model_: Pipeline | None = None
    residual_feature_columns_: list[str] | None = None
    clip_abs_: float | None = None
    training_rows_available_: int = 0
    training_rows_used_: int = 0

    @property
    def supports_full_frame_prediction(self) -> bool:
        return bool(getattr(self.base_predictor, "supports_full_frame_prediction", False))

    def fit(self, train_frame: pd.DataFrame) -> "FixedPrefixResidualCorrectionRegressor":
        if self.residual_train_steps <= 0:
            raise ValueError("residual_train_steps must be positive")
        if self.correction_scale < 0.0:
            raise ValueError("correction_scale must be non-negative")
        if not 0.0 < self.correction_clip_quantile <= 1.0:
            raise ValueError("correction_clip_quantile must be in (0, 1]")
        if self.max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive")

        self.residual_feature_columns_ = detect_residual_feature_columns(
            list(train_frame.columns),
            banned_feature_columns=self.banned_feature_columns,
        )
        self.final_base_predictor_ = copy.deepcopy(self.base_predictor)
        self.final_base_predictor_.fit(train_frame)

        train_end_ts = int(train_frame[self.time_column].max())
        min_residual_train_ts = max(
            int(train_frame[self.time_column].min()),
            train_end_ts - self.residual_train_steps + 1,
        )
        prefix_frame = train_frame.loc[train_frame[self.time_column] < min_residual_train_ts].copy()
        residual_window = train_frame.loc[
            train_frame[self.time_column] >= min_residual_train_ts
        ].copy()
        gated_train = residual_window.loc[
            build_slice_mask(
                residual_window,
                horizons=self.horizons,
                sub_categories=self.sub_categories,
                codes=self.codes,
            )
        ].copy()
        self.training_rows_available_ = int(len(gated_train))
        self.training_rows_used_ = 0
        self.residual_model_ = None
        self.clip_abs_ = None

        if len(prefix_frame) == 0 or len(gated_train) == 0:
            return self

        pseudo_history_predictor = copy.deepcopy(self.base_predictor)
        pseudo_history_predictor.fit(prefix_frame)
        gated_train["base_prediction"] = np.asarray(
            pseudo_history_predictor.predict_batch(gated_train),
            dtype=float,
        )
        gated_train = compute_raw_residual_targets(gated_train)
        gated_train = _sample_frame(
            gated_train,
            max_rows=self.max_train_rows,
            time_column=self.time_column,
            random_state=self.random_state,
        )
        self.training_rows_used_ = int(len(gated_train))
        if self.training_rows_used_ == 0:
            return self

        self.clip_abs_ = float(
            np.quantile(
                np.abs(gated_train[RAW_RESIDUAL_TARGET_COLUMN].to_numpy(dtype=float)),
                self.correction_clip_quantile,
            )
        )
        self.residual_model_ = _build_raw_residual_model(
            model_type=self.model_type,
            numeric_columns=self.residual_feature_columns_,
            alpha=self.alpha,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )
        self.residual_model_.fit(
            gated_train[RESIDUAL_CATEGORICAL_COLUMNS + self.residual_feature_columns_],
            gated_train[RAW_RESIDUAL_TARGET_COLUMN].to_numpy(dtype=float),
            model__sample_weight=gated_train[self.weight_column].to_numpy(dtype=float),
        )
        return self

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        if self.final_base_predictor_ is None:
            raise ValueError("model must be fit before prediction")

        predictions = np.asarray(
            self.final_base_predictor_.predict_batch(batch_frame),
            dtype=float,
        )
        if self.residual_model_ is None or self.residual_feature_columns_ is None:
            return predictions

        gate_mask = build_slice_mask(
            batch_frame,
            horizons=self.horizons,
            sub_categories=self.sub_categories,
            codes=self.codes,
        )
        if not gate_mask.any():
            return predictions

        gated_frame = batch_frame.loc[
            gate_mask,
            RESIDUAL_CATEGORICAL_COLUMNS + self.residual_feature_columns_,
        ]
        raw_corrections = self.residual_model_.predict(gated_frame)
        corrections = shrink_residual_corrections(
            raw_corrections,
            correction_scale=self.correction_scale,
            clip_abs=self.clip_abs_,
        )
        predictions[gate_mask.to_numpy(dtype=bool)] += corrections
        return predictions


@dataclass
class NormalizedPrefixResidualCorrectionRegressor:
    """Train a normalized residual model on a recent slice using prefix-only pseudo-history."""

    base_predictor: object
    model_type: str = "ridge"
    horizons: tuple[int, ...] | list[int] | None = None
    sub_categories: tuple[str, ...] | list[str] | None = None
    codes: tuple[str, ...] | list[str] | None = None
    residual_train_steps: int = 720
    correction_scale: float = 0.08
    correction_clip_quantile: float = 0.85
    max_train_rows: int = 180_000
    alpha: float = 10.0
    epsilon: float = 1.35
    max_iter: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 200
    oof_holdout_steps: int = 360
    oof_step_size: int = 360
    oof_min_train_steps: int = 2000
    banned_feature_columns: set[str] = field(default_factory=set)
    time_column: str = TIME_COLUMN
    target_column: str = TARGET_COLUMN
    weight_column: str = WEIGHT_COLUMN
    random_state: int = 42
    final_base_predictor_: object | None = None
    residual_model_: Pipeline | None = None
    residual_feature_columns_: list[str] | None = None
    primary_scale_lookup_: pd.DataFrame | None = None
    fallback_scale_lookup_: pd.DataFrame | None = None
    global_scale_: float | None = None
    clip_abs_: float | None = None
    training_rows_available_: int = 0
    training_rows_used_: int = 0

    @property
    def supports_full_frame_prediction(self) -> bool:
        return bool(getattr(self.base_predictor, "supports_full_frame_prediction", False))

    def fit(self, train_frame: pd.DataFrame) -> "NormalizedPrefixResidualCorrectionRegressor":
        if self.residual_train_steps <= 0:
            raise ValueError("residual_train_steps must be positive")
        if self.correction_scale < 0.0:
            raise ValueError("correction_scale must be non-negative")
        if not 0.0 < self.correction_clip_quantile <= 1.0:
            raise ValueError("correction_clip_quantile must be in (0, 1]")
        if self.max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive")
        if self.oof_holdout_steps <= 0:
            raise ValueError("oof_holdout_steps must be positive")
        if self.oof_step_size <= 0:
            raise ValueError("oof_step_size must be positive")
        if self.oof_min_train_steps <= 0:
            raise ValueError("oof_min_train_steps must be positive")

        self.residual_feature_columns_ = detect_residual_feature_columns(
            list(train_frame.columns),
            banned_feature_columns=self.banned_feature_columns,
        )
        self.final_base_predictor_ = copy.deepcopy(self.base_predictor)
        self.final_base_predictor_.fit(train_frame)

        self.primary_scale_lookup_ = build_residual_scale_lookup(
            train_frame,
            level_columns=RESIDUAL_PRIMARY_SCALE_COLUMNS,
            target_column=self.target_column,
            weight_column=self.weight_column,
        )
        self.fallback_scale_lookup_ = build_residual_scale_lookup(
            train_frame,
            level_columns=RESIDUAL_FALLBACK_SCALE_COLUMNS,
            target_column=self.target_column,
            weight_column=self.weight_column,
        )
        self.global_scale_ = compute_global_residual_scale(
            train_frame,
            target_column=self.target_column,
            weight_column=self.weight_column,
        )

        train_end_ts = int(train_frame[self.time_column].max())
        min_residual_train_ts = max(int(train_frame[self.time_column].min()), train_end_ts - self.residual_train_steps + 1)
        oof_training_pool = self._build_internal_oof_training_pool(train_frame)
        gated_train = oof_training_pool.loc[
            build_slice_mask(
                oof_training_pool,
                horizons=self.horizons,
                sub_categories=self.sub_categories,
                codes=self.codes,
            )
        ].copy()
        gated_train = gated_train.loc[gated_train[self.time_column] >= min_residual_train_ts].copy()

        self.training_rows_available_ = int(len(gated_train))
        self.training_rows_used_ = 0
        self.residual_model_ = None
        self.clip_abs_ = None

        if len(gated_train) == 0:
            return self

        gated_train = _sample_frame(
            gated_train,
            max_rows=self.max_train_rows,
            time_column=self.time_column,
            random_state=self.random_state,
        )
        self.training_rows_used_ = int(len(gated_train))
        if self.training_rows_used_ == 0:
            return self

        self.clip_abs_ = float(
            np.quantile(
                np.abs(gated_train[NORMALIZED_RESIDUAL_TARGET_COLUMN].to_numpy(dtype=float)),
                self.correction_clip_quantile,
            )
        )
        self.residual_model_ = _build_raw_residual_model(
            model_type=self.model_type,
            numeric_columns=self.residual_feature_columns_,
            alpha=self.alpha,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )
        self.residual_model_.fit(
            gated_train[RESIDUAL_CATEGORICAL_COLUMNS + self.residual_feature_columns_],
            gated_train[NORMALIZED_RESIDUAL_TARGET_COLUMN].to_numpy(dtype=float),
        )
        return self

    def _build_internal_oof_training_pool(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        max_ts_index = int(train_frame[self.time_column].max())
        n_folds = _max_expanding_window_folds(
            max_ts_index=max_ts_index,
            holdout_steps=self.oof_holdout_steps,
            step_size=self.oof_step_size,
            min_train_steps=self.oof_min_train_steps,
        )
        if n_folds == 0:
            return train_frame.head(0).copy()

        ordered_train_frame = train_frame.sort_values(
            [self.time_column, ID_COLUMN],
            kind="stable",
        ).reset_index(drop=True)
        ordered_ts = ordered_train_frame[self.time_column].to_numpy(dtype=int, copy=False)
        oof_parts: list[pd.DataFrame] = []
        folds = build_expanding_window_folds(
            max_ts_index=max_ts_index,
            holdout_steps=self.oof_holdout_steps,
            n_folds=n_folds,
            step_size=self.oof_step_size,
            min_train_steps=self.oof_min_train_steps,
        )
        for fold in folds:
            prefix_end = int(np.searchsorted(ordered_ts, fold.train_end_ts, side="right"))
            validation_start = int(
                np.searchsorted(ordered_ts, fold.validation_start_ts, side="left")
            )
            validation_end = int(
                np.searchsorted(ordered_ts, fold.validation_end_ts, side="right")
            )
            prefix_frame = ordered_train_frame.iloc[:prefix_end]
            validation_frame = ordered_train_frame.iloc[validation_start:validation_end]
            if len(prefix_frame) == 0 or len(validation_frame) == 0:
                continue

            fold_base_predictor = copy.deepcopy(self.base_predictor)
            fold_predictions = SequentialInferencePipeline(
                transformer=IdentityTransformer(),
                predictor=fold_base_predictor,
            ).fit(prefix_frame).predict(
                validation_frame,
                already_sorted=True,
            )
            fold_validation = validation_frame.copy()
            fold_validation["base_prediction"] = fold_predictions[
                "prediction"
            ].to_numpy(dtype=float)
            primary_lookup = build_residual_scale_lookup(
                prefix_frame,
                level_columns=RESIDUAL_PRIMARY_SCALE_COLUMNS,
                target_column=self.target_column,
                weight_column=self.weight_column,
            )
            fallback_lookup = build_residual_scale_lookup(
                prefix_frame,
                level_columns=RESIDUAL_FALLBACK_SCALE_COLUMNS,
                target_column=self.target_column,
                weight_column=self.weight_column,
            )
            global_scale = compute_global_residual_scale(
                prefix_frame,
                target_column=self.target_column,
                weight_column=self.weight_column,
            )
            fold_validation = attach_residual_scales(
                fold_validation,
                primary_lookup=primary_lookup,
                fallback_lookup=fallback_lookup,
                global_scale=global_scale,
            )
            oof_parts.append(
                compute_normalized_residual_targets(
                    fold_validation,
                    target_column=self.target_column,
                    base_prediction_column="base_prediction",
                    scale_column=RESIDUAL_SCALE_COLUMN,
                )
            )

        if not oof_parts:
            return train_frame.head(0).copy()
        return pd.concat(oof_parts, ignore_index=True)

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        if self.final_base_predictor_ is None:
            raise ValueError("model must be fit before prediction")

        predictions = np.asarray(
            self.final_base_predictor_.predict_batch(batch_frame),
            dtype=float,
        )
        if (
            self.residual_model_ is None
            or self.residual_feature_columns_ is None
            or self.primary_scale_lookup_ is None
            or self.fallback_scale_lookup_ is None
            or self.global_scale_ is None
        ):
            return predictions

        gate_mask = build_slice_mask(
            batch_frame,
            horizons=self.horizons,
            sub_categories=self.sub_categories,
            codes=self.codes,
        )
        if not gate_mask.any():
            return predictions

        gated_frame = attach_residual_scales(
            batch_frame.loc[
                gate_mask,
                RESIDUAL_CATEGORICAL_COLUMNS + self.residual_feature_columns_,
            ].copy(),
            primary_lookup=self.primary_scale_lookup_,
            fallback_lookup=self.fallback_scale_lookup_,
            global_scale=float(self.global_scale_),
        )
        raw_normalized_corrections = self.residual_model_.predict(
            gated_frame[RESIDUAL_CATEGORICAL_COLUMNS + self.residual_feature_columns_]
        )
        corrections = apply_normalized_residual_corrections(
            raw_normalized_corrections,
            scales=gated_frame[RESIDUAL_SCALE_COLUMN].to_numpy(dtype=float),
            correction_scale=self.correction_scale,
            clip_abs=self.clip_abs_,
        )
        predictions[gate_mask.to_numpy(dtype=bool)] += corrections
        return predictions
