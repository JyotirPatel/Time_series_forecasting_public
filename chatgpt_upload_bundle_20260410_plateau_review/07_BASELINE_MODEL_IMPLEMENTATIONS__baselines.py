"""Leakage-safe baseline predictors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .constants import (
    GROUP_COLUMNS,
    HORIZON_COLUMN,
    ID_COLUMN,
    SUB_CATEGORY_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
)


class BatchPredictor(Protocol):
    def fit(self, train_frame: pd.DataFrame) -> "BatchPredictor":
        ...

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        ...


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numerator = float((values * weights).sum())
    denominator = float(weights.sum())
    if denominator <= 0.0:
        return float(values.mean())
    return numerator / denominator


def _aggregate_group_statistics(
    frame: pd.DataFrame,
    *,
    level_columns: list[str],
    target_column: str,
    weight_column: str,
) -> pd.DataFrame:
    level_frame = frame[level_columns + [target_column, weight_column]].copy()
    level_frame["_weighted_target"] = level_frame[target_column] * level_frame[weight_column]
    return (
        level_frame.groupby(level_columns, dropna=False, observed=True)
        .agg(
            weighted_target_sum=("_weighted_target", "sum"),
            weight_sum=(weight_column, "sum"),
            row_count=(target_column, "size"),
        )
        .reset_index()
    )


@dataclass
class WeightedGroupMeanRegressor:
    """Hierarchical weighted mean fallback baseline."""

    fallback_levels: list[list[str]] = field(
        default_factory=lambda: [
            [*GROUP_COLUMNS],
            ["code", "sub_category", HORIZON_COLUMN],
            ["sub_category", HORIZON_COLUMN],
            [HORIZON_COLUMN],
        ]
    )
    target_column: str = TARGET_COLUMN
    weight_column: str = WEIGHT_COLUMN
    prediction_column: str = "prediction"
    global_mean_: float | None = None
    lookup_frames_: list[tuple[list[str], pd.DataFrame]] = field(default_factory=list)

    def fit(self, train_frame: pd.DataFrame) -> "WeightedGroupMeanRegressor":
        self.global_mean_ = _weighted_mean(
            train_frame[self.target_column],
            train_frame[self.weight_column],
        )
        self.lookup_frames_ = []

        for level_columns in self.fallback_levels:
            aggregated = _aggregate_group_statistics(
                train_frame,
                level_columns=level_columns,
                target_column=self.target_column,
                weight_column=self.weight_column,
            )
            aggregated[self.prediction_column] = np.where(
                aggregated["weight_sum"] > 0.0,
                aggregated["weighted_target_sum"] / aggregated["weight_sum"],
                self.global_mean_,
            )
            self.lookup_frames_.append(
                (level_columns, aggregated[level_columns + [self.prediction_column]])
            )

        return self

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        if self.global_mean_ is None:
            raise ValueError("model must be fit before prediction")

        predictions = np.full(len(batch_frame), np.nan, dtype=float)

        for level_columns, lookup_frame in self.lookup_frames_:
            merged = batch_frame[level_columns].merge(
                lookup_frame,
                on=level_columns,
                how="left",
            )
            level_predictions = merged[self.prediction_column].to_numpy(dtype=float)
            fill_mask = np.isnan(predictions) & ~np.isnan(level_predictions)
            predictions[fill_mask] = level_predictions[fill_mask]

        predictions[np.isnan(predictions)] = self.global_mean_
        return predictions


@dataclass
class SmoothedWeightedGroupMeanRegressor:
    """Hierarchical weighted mean baseline with shrinkage toward coarser priors."""

    fallback_levels: list[list[str]] = field(
        default_factory=lambda: [
            ["code", "sub_category", HORIZON_COLUMN],
            ["sub_category", HORIZON_COLUMN],
            [HORIZON_COLUMN],
        ]
    )
    prior_weight: float = 100.0
    target_column: str = TARGET_COLUMN
    weight_column: str = WEIGHT_COLUMN
    prediction_column: str = "prediction"
    global_mean_: float | None = None
    lookup_frames_: list[tuple[list[str], pd.DataFrame]] = field(default_factory=list)

    def fit(self, train_frame: pd.DataFrame) -> "SmoothedWeightedGroupMeanRegressor":
        if self.prior_weight < 0.0:
            raise ValueError("prior_weight must be non-negative")

        self.global_mean_ = _weighted_mean(
            train_frame[self.target_column],
            train_frame[self.weight_column],
        )

        lookup_by_level: dict[tuple[str, ...], pd.DataFrame] = {}
        prior_frame: pd.DataFrame | None = None
        prior_columns: list[str] = []

        for level_columns in reversed(self.fallback_levels):
            aggregated = _aggregate_group_statistics(
                train_frame,
                level_columns=level_columns,
                target_column=self.target_column,
                weight_column=self.weight_column,
            )

            if prior_frame is None:
                prior_predictions = np.full(len(aggregated), self.global_mean_, dtype=float)
            else:
                if not set(prior_columns).issubset(level_columns):
                    raise ValueError("fallback_levels must be nested for smoothing to be valid")
                prior_frame_for_merge = prior_frame.rename(
                    columns={self.prediction_column: f"{self.prediction_column}_prior"}
                )
                merged = aggregated.merge(
                    prior_frame_for_merge,
                    on=prior_columns,
                    how="left",
                    validate="many_to_one",
                )
                prior_predictions = (
                    merged[f"{self.prediction_column}_prior"]
                    .fillna(self.global_mean_)
                    .to_numpy(dtype=float)
                )
                aggregated = merged.drop(columns=[f"{self.prediction_column}_prior"])

            aggregated[self.prediction_column] = np.where(
                aggregated["weight_sum"] > 0.0,
                (
                    aggregated["weighted_target_sum"].to_numpy(dtype=float)
                    + self.prior_weight * prior_predictions
                )
                / (aggregated["weight_sum"].to_numpy(dtype=float) + self.prior_weight),
                prior_predictions,
            )

            current_lookup = aggregated[level_columns + [self.prediction_column]]
            lookup_by_level[tuple(level_columns)] = current_lookup
            prior_frame = current_lookup
            prior_columns = level_columns

        self.lookup_frames_ = [
            (level_columns, lookup_by_level[tuple(level_columns)])
            for level_columns in self.fallback_levels
        ]
        return self

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        if self.global_mean_ is None:
            raise ValueError("model must be fit before prediction")

        predictions = np.full(len(batch_frame), np.nan, dtype=float)

        for level_columns, lookup_frame in self.lookup_frames_:
            merged = batch_frame[level_columns].merge(
                lookup_frame,
                on=level_columns,
                how="left",
            )
            level_predictions = merged[self.prediction_column].to_numpy(dtype=float)
            fill_mask = np.isnan(predictions) & ~np.isnan(level_predictions)
            predictions[fill_mask] = level_predictions[fill_mask]

        predictions[np.isnan(predictions)] = self.global_mean_
        return predictions


@dataclass
class LinearBlendRegressor:
    """Blend several causal batch predictors with fixed normalized weights."""

    predictors: list[BatchPredictor]
    weights: list[float]
    normalize_weights: bool = True
    normalized_weights_: np.ndarray | None = None

    def fit(self, train_frame: pd.DataFrame) -> "LinearBlendRegressor":
        if len(self.predictors) == 0:
            raise ValueError("LinearBlendRegressor requires at least one predictor")
        if len(self.predictors) != len(self.weights):
            raise ValueError("predictors and weights must have the same length")

        weights = np.asarray(self.weights, dtype=float)
        if self.normalize_weights:
            weight_sum = float(weights.sum())
            if not np.isfinite(weight_sum) or weight_sum == 0.0:
                raise ValueError("weights must sum to a finite non-zero value")
            weights = weights / weight_sum

        self.normalized_weights_ = weights
        for predictor in self.predictors:
            predictor.fit(train_frame)
        return self

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        if self.normalized_weights_ is None:
            raise ValueError("model must be fit before prediction")

        predictions = np.zeros(len(batch_frame), dtype=float)
        for weight, predictor in zip(self.normalized_weights_, self.predictors, strict=True):
            predictions += weight * np.asarray(
                predictor.predict_batch(batch_frame),
                dtype=float,
            )
        return predictions


@dataclass
class WarmStartLastTargetBlendRegressor:
    """Blend a base predictor with the last training target for exact warm groups."""

    base_predictor: BatchPredictor
    blend_rules: list[dict[str, float | int | str]]
    group_columns: list[str] = field(default_factory=lambda: [*GROUP_COLUMNS])
    target_column: str = TARGET_COLUMN
    time_column: str = TIME_COLUMN
    id_column: str = ID_COLUMN
    last_target_column: str = "_last_target"
    last_target_lookup_: pd.DataFrame | None = None
    parsed_rules_: list[tuple[dict[str, object], float]] | None = None

    def fit(self, train_frame: pd.DataFrame) -> "WarmStartLastTargetBlendRegressor":
        self.base_predictor.fit(train_frame)

        ordered = train_frame.sort_values(
            [self.time_column, self.id_column],
            kind="stable",
        )
        self.last_target_lookup_ = (
            ordered[self.group_columns + [self.target_column]]
            .drop_duplicates(self.group_columns, keep="last")
            .rename(columns={self.target_column: self.last_target_column})
        )

        parsed_rules: list[tuple[dict[str, object], float]] = []
        for rule in self.blend_rules:
            alpha = float(rule["alpha"])
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError("blend rule alpha must be between 0 and 1")
            if alpha == 0.0:
                continue
            conditions = {
                str(column): value
                for column, value in rule.items()
                if column != "alpha"
            }
            unknown_columns = sorted(set(conditions) - set(self.group_columns))
            if unknown_columns:
                raise ValueError(
                    f"blend rule columns must be a subset of {self.group_columns}: "
                    f"{unknown_columns}"
                )
            parsed_rules.append((conditions, alpha))

        self.parsed_rules_ = parsed_rules
        return self

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        if self.last_target_lookup_ is None or self.parsed_rules_ is None:
            raise ValueError("model must be fit before prediction")

        base_predictions = np.asarray(
            self.base_predictor.predict_batch(batch_frame),
            dtype=float,
        )
        if len(self.parsed_rules_) == 0:
            return base_predictions

        merged = batch_frame[self.group_columns].merge(
            self.last_target_lookup_,
            on=self.group_columns,
            how="left",
            validate="one_to_one",
        )

        last_targets = merged[self.last_target_column].to_numpy(dtype=float)
        seen_mask = ~np.isnan(last_targets)
        if not seen_mask.any():
            return base_predictions

        predictions = base_predictions.copy()
        for conditions, alpha in self.parsed_rules_:
            blend_mask = seen_mask.copy()
            for column, value in conditions.items():
                blend_mask &= batch_frame[column].to_numpy() == value
            if not blend_mask.any():
                continue
            predictions[blend_mask] = (
                (1.0 - alpha) * predictions[blend_mask]
                + alpha * last_targets[blend_mask]
            )
        return predictions


@dataclass
class RecentMeanDeltaRegressor:
    """Add a fixed recent-vs-full-history mean delta on selected stable groups."""

    base_predictor: BatchPredictor
    recent_window: int
    min_recent_rows: int
    beta: float
    level_columns: list[str] = field(
        default_factory=lambda: ["code", SUB_CATEGORY_COLUMN, HORIZON_COLUMN]
    )
    horizons: list[int] | None = None
    beta_by_horizon: dict[int, float] | None = None
    clip_quantile_by_horizon: dict[int, float] | None = None
    apply_rules: list[dict[str, object]] | None = None
    min_recent_weight_sum: float = 0.0
    target_column: str = TARGET_COLUMN
    weight_column: str = WEIGHT_COLUMN
    time_column: str = TIME_COLUMN
    delta_column: str = "_recent_mean_delta"
    clip_threshold_by_horizon_: dict[int, float] = field(default_factory=dict)
    delta_lookup_: pd.DataFrame | None = None

    def fit(self, train_frame: pd.DataFrame) -> "RecentMeanDeltaRegressor":
        if self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.min_recent_rows < 0:
            raise ValueError("min_recent_rows must be non-negative")
        if self.min_recent_weight_sum < 0.0:
            raise ValueError("min_recent_weight_sum must be non-negative")
        if self.beta_by_horizon is not None:
            self.beta_by_horizon = {
                int(horizon): float(beta)
                for horizon, beta in self.beta_by_horizon.items()
            }
        if self.clip_quantile_by_horizon is not None:
            normalized_quantiles: dict[int, float] = {}
            for horizon, quantile in self.clip_quantile_by_horizon.items():
                quantile_value = float(quantile)
                if quantile_value <= 0.0 or quantile_value > 1.0:
                    raise ValueError("clip quantiles must be in (0, 1]")
                normalized_quantiles[int(horizon)] = quantile_value
            self.clip_quantile_by_horizon = normalized_quantiles

        self.base_predictor.fit(train_frame)

        full_stats = _aggregate_group_statistics(
            train_frame,
            level_columns=self.level_columns,
            target_column=self.target_column,
            weight_column=self.weight_column,
        ).rename(
            columns={
                "weighted_target_sum": "full_weighted_target_sum",
                "weight_sum": "full_weight_sum",
                "row_count": "full_row_count",
            }
        )
        recent_cutoff = int(train_frame[self.time_column].max()) - self.recent_window + 1
        recent_frame = train_frame.loc[train_frame[self.time_column] >= recent_cutoff]
        recent_stats = _aggregate_group_statistics(
            recent_frame,
            level_columns=self.level_columns,
            target_column=self.target_column,
            weight_column=self.weight_column,
        ).rename(
            columns={
                "weighted_target_sum": "recent_weighted_target_sum",
                "weight_sum": "recent_weight_sum",
                "row_count": "recent_row_count",
            }
        )

        delta_frame = full_stats.merge(
            recent_stats,
            on=self.level_columns,
            how="inner",
            validate="one_to_one",
        )
        if self.horizons is not None:
            delta_frame = delta_frame.loc[
                delta_frame[HORIZON_COLUMN].isin(self.horizons)
            ].copy()
        delta_frame = delta_frame.loc[
            delta_frame["recent_row_count"] >= self.min_recent_rows
        ].copy()
        if self.min_recent_weight_sum > 0.0:
            delta_frame = delta_frame.loc[
                delta_frame["recent_weight_sum"] >= self.min_recent_weight_sum
            ].copy()

        if delta_frame.empty:
            self.delta_lookup_ = pd.DataFrame(
                columns=[*self.level_columns, self.delta_column]
            )
            self.clip_threshold_by_horizon_ = {}
            return self

        full_mean = np.where(
            delta_frame["full_weight_sum"].to_numpy(dtype=float) > 0.0,
            delta_frame["full_weighted_target_sum"].to_numpy(dtype=float)
            / delta_frame["full_weight_sum"].to_numpy(dtype=float),
            0.0,
        )
        recent_mean = np.where(
            delta_frame["recent_weight_sum"].to_numpy(dtype=float) > 0.0,
            delta_frame["recent_weighted_target_sum"].to_numpy(dtype=float)
            / delta_frame["recent_weight_sum"].to_numpy(dtype=float),
            0.0,
        )
        delta_values = recent_mean - full_mean

        self.clip_threshold_by_horizon_ = {}
        if self.clip_quantile_by_horizon:
            horizon_values = delta_frame[HORIZON_COLUMN].to_numpy()
            for horizon, quantile in self.clip_quantile_by_horizon.items():
                horizon_mask = horizon_values == horizon
                if not horizon_mask.any():
                    continue
                threshold = float(
                    np.quantile(np.abs(delta_values[horizon_mask]), quantile)
                )
                self.clip_threshold_by_horizon_[horizon] = threshold
                delta_values[horizon_mask] = np.clip(
                    delta_values[horizon_mask],
                    -threshold,
                    threshold,
                )

        delta_frame[self.delta_column] = delta_values
        if self.apply_rules is not None:
            allowed_columns = set(self.level_columns)
            apply_mask = np.zeros(len(delta_frame), dtype=bool)
            for rule in self.apply_rules:
                conditions = {
                    str(column): value
                    for column, value in rule.items()
                }
                unknown_columns = sorted(set(conditions) - allowed_columns)
                if unknown_columns:
                    raise ValueError(
                        f"apply_rules columns must be a subset of {self.level_columns}: "
                        f"{unknown_columns}"
                    )
                if len(conditions) == 0:
                    apply_mask |= True
                    continue
                rule_mask = np.ones(len(delta_frame), dtype=bool)
                for column, value in conditions.items():
                    rule_mask &= delta_frame[column].to_numpy() == value
                apply_mask |= rule_mask
            delta_frame = delta_frame.loc[apply_mask].copy()

        self.delta_lookup_ = delta_frame[self.level_columns + [self.delta_column]]
        return self

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        if self.delta_lookup_ is None:
            raise ValueError("model must be fit before prediction")

        base_predictions = np.asarray(
            self.base_predictor.predict_batch(batch_frame),
            dtype=float,
        )
        has_non_zero_beta = self.beta != 0.0 or (
            self.beta_by_horizon is not None
            and any(beta != 0.0 for beta in self.beta_by_horizon.values())
        )
        if not has_non_zero_beta or len(self.delta_lookup_) == 0:
            return base_predictions

        merged = batch_frame[self.level_columns].merge(
            self.delta_lookup_,
            on=self.level_columns,
            how="left",
        )
        delta_values = merged[self.delta_column].to_numpy(dtype=float)
        beta_values = np.full(len(batch_frame), self.beta, dtype=float)
        if self.beta_by_horizon:
            horizon_values = batch_frame[HORIZON_COLUMN].to_numpy()
            for horizon, beta in self.beta_by_horizon.items():
                beta_values[horizon_values == horizon] = beta
        apply_mask = ~np.isnan(delta_values) & (beta_values != 0.0)
        if not apply_mask.any():
            return base_predictions

        predictions = base_predictions.copy()
        predictions[apply_mask] = (
            predictions[apply_mask] + beta_values[apply_mask] * delta_values[apply_mask]
        )
        return predictions
