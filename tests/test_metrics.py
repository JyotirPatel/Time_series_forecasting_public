from __future__ import annotations

import numpy as np

from ts_forecasting.metrics import (
    weighted_rmse_breakdown,
    weighted_rmse_score,
    weighted_rmse_score_from_sums,
)


def test_weighted_rmse_score_is_one_for_perfect_predictions() -> None:
    y_target = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    weight = np.array([1.0, 1.0, 1.0])

    assert weighted_rmse_score(y_target, y_pred, weight) == 1.0


def test_weighted_rmse_score_clips_large_errors_to_zero() -> None:
    y_target = np.array([1.0, 2.0])
    y_pred = np.array([100.0, -100.0])
    weight = np.array([1.0, 1.0])

    assert weighted_rmse_score(y_target, y_pred, weight) == 0.0


def test_weighted_rmse_breakdown_matches_score_and_aggregates_exactly() -> None:
    fold_1 = weighted_rmse_breakdown(
        np.array([1.0, 2.0]),
        np.array([1.0, 1.5]),
        np.array([1.0, 1.0]),
    )
    fold_2 = weighted_rmse_breakdown(
        np.array([3.0, 4.0]),
        np.array([2.5, 5.0]),
        np.array([1.0, 2.0]),
    )
    concatenated = weighted_rmse_score(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 1.5, 2.5, 5.0]),
        np.array([1.0, 1.0, 1.0, 2.0]),
    )

    assert fold_1.score == weighted_rmse_score_from_sums(
        error_sum=fold_1.error_sum,
        denom_sum=fold_1.denom_sum,
    )
    assert fold_2.score == weighted_rmse_score_from_sums(
        error_sum=fold_2.error_sum,
        denom_sum=fold_2.denom_sum,
    )
    assert concatenated == weighted_rmse_score_from_sums(
        error_sum=fold_1.error_sum + fold_2.error_sum,
        denom_sum=fold_1.denom_sum + fold_2.denom_sum,
    )
