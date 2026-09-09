"""Competition metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))


@dataclass(frozen=True)
class WeightedRmseBreakdown:
    """Sufficient statistics and derived values for the competition metric."""

    error_sum: float
    denom_sum: float
    ratio: float
    clipped_ratio: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def weighted_rmse_error_sum(
    y_target: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Return the weighted squared error numerator."""

    y_target = np.asarray(y_target, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weight = np.asarray(weight, dtype=float)

    if y_target.shape != y_pred.shape or y_target.shape != weight.shape:
        raise ValueError("y_target, y_pred, and weight must have the same shape")

    return float(np.sum(weight * (y_target - y_pred) ** 2))


def weighted_rmse_denom_sum(
    y_target: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Return the weighted target-energy denominator."""

    y_target = np.asarray(y_target, dtype=float)
    weight = np.asarray(weight, dtype=float)

    if y_target.shape != weight.shape:
        raise ValueError("y_target and weight must have the same shape")

    denom = float(np.sum(weight * y_target**2))
    if denom <= 0.0:
        raise ValueError("metric denominator must be positive")
    return denom


def weighted_rmse_score_from_sums(
    *,
    error_sum: float,
    denom_sum: float,
) -> float:
    """Score the competition metric from aggregate numerator and denominator sums."""

    if denom_sum <= 0.0:
        raise ValueError("metric denominator must be positive")

    ratio = float(error_sum / denom_sum)
    clipped = _clip01(ratio)
    return float(np.sqrt(1.0 - clipped))


def weighted_rmse_breakdown(
    y_target: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
) -> WeightedRmseBreakdown:
    """Return exact metric components plus the final clipped score."""

    error_sum = weighted_rmse_error_sum(y_target, y_pred, weight)
    denom_sum = weighted_rmse_denom_sum(y_target, weight)
    ratio = float(error_sum / denom_sum)
    clipped_ratio = _clip01(ratio)
    return WeightedRmseBreakdown(
        error_sum=error_sum,
        denom_sum=denom_sum,
        ratio=ratio,
        clipped_ratio=clipped_ratio,
        score=float(np.sqrt(1.0 - clipped_ratio)),
    )


def weighted_rmse_score(
    y_target: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Exact competition metric supplied in the Kaggle evaluation page."""

    return weighted_rmse_breakdown(y_target, y_pred, weight).score
