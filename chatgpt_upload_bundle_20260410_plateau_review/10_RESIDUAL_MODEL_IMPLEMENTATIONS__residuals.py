"""Residual-model helpers built on top of a base prediction."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

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
    SUB_CATEGORY_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
    detect_feature_columns,
)

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

    primary_scale_column = f"{scale_column}_primary"
    fallback_scale_column = f"{scale_column}_fallback"

    merged = frame.merge(
        primary_lookup.rename(columns={scale_column: primary_scale_column}),
        on=primary_level_columns,
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        fallback_lookup.rename(columns={scale_column: fallback_scale_column}),
        on=fallback_level_columns,
        how="left",
        validate="many_to_one",
    )

    merged[scale_column] = merged[primary_scale_column]
    fill_mask = merged[scale_column].isna() | (merged[scale_column] <= min_scale)
    merged.loc[fill_mask, scale_column] = merged.loc[fill_mask, fallback_scale_column]
    fill_mask = merged[scale_column].isna() | (merged[scale_column] <= min_scale)
    merged.loc[fill_mask, scale_column] = max(float(global_scale), min_scale)
    merged[scale_column] = merged[scale_column].clip(lower=min_scale)

    return merged.drop(columns=[primary_scale_column, fallback_scale_column])


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
    if len(frame) <= max_rows:
        return frame.sort_values(time_column, kind="stable")
    return frame.sample(n=max_rows, random_state=random_state).sort_values(
        time_column,
        kind="stable",
    )


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
