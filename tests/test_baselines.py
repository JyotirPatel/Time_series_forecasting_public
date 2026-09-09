from __future__ import annotations

import numpy as np
import pandas as pd

from ts_forecasting.baselines import (
    ConditionalLinearBlendRegressor,
    LinearBlendRegressor,
    RecentMeanDeltaRegressor,
    SmoothedWeightedGroupMeanRegressor,
    WarmStartLastTargetBlendRegressor,
    WeightedGroupMeanRegressor,
)


class ConstantPredictor:
    supports_full_frame_prediction = True

    def __init__(self, value: float) -> None:
        self.value = value

    def fit(self, train_frame: pd.DataFrame) -> "ConstantPredictor":
        return self

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(batch_frame), self.value, dtype=float)


def test_weighted_group_mean_regressor_uses_hierarchy_and_fallbacks() -> None:
    train_frame = pd.DataFrame(
        {
            "code": ["A", "A", "B"],
            "sub_code": ["S1", "S1", "S2"],
            "sub_category": ["C1", "C1", "C2"],
            "horizon": [1, 1, 1],
            "y_target": [10.0, 14.0, 30.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "code": ["A", "X"],
            "sub_code": ["S1", "SX"],
            "sub_category": ["C1", "C2"],
            "horizon": [1, 1],
        }
    )

    model = WeightedGroupMeanRegressor(
        fallback_levels=[
            ["code", "sub_code", "sub_category", "horizon"],
            ["sub_category", "horizon"],
            ["horizon"],
        ]
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert np.isclose(predictions[0], 12.0)
    assert np.isclose(predictions[1], 30.0)


def test_smoothed_group_mean_regressor_shrinks_toward_coarser_prior() -> None:
    train_frame = pd.DataFrame(
        {
            "code": ["A", "B"],
            "sub_category": ["C1", "C1"],
            "horizon": [1, 1],
            "y_target": [100.0, 0.0],
            "weight": [1.0, 9.0],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "code": ["A", "B"],
            "sub_category": ["C1", "C1"],
            "horizon": [1, 1],
        }
    )

    model = SmoothedWeightedGroupMeanRegressor(
        fallback_levels=[
            ["code", "sub_category", "horizon"],
            ["sub_category", "horizon"],
            ["horizon"],
        ],
        prior_weight=10.0,
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert np.isclose(predictions[0], 18.18181818)
    assert np.isclose(predictions[1], 5.26315789)


def test_linear_blend_regressor_combines_component_predictions() -> None:
    train_frame = pd.DataFrame(
        {
            "code": ["A", "A", "B", "B"],
            "sub_category": ["C1", "C1", "C2", "C2"],
            "horizon": [1, 1, 1, 1],
            "y_target": [10.0, 14.0, 30.0, 34.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "code": ["A", "B"],
            "sub_category": ["C1", "C2"],
            "horizon": [1, 1],
        }
    )

    coarse = WeightedGroupMeanRegressor(
        fallback_levels=[
            ["code", "sub_category", "horizon"],
            ["horizon"],
        ]
    )
    horizon = WeightedGroupMeanRegressor(fallback_levels=[["horizon"]])
    model = LinearBlendRegressor(
        predictors=[coarse, horizon],
        weights=[0.75, 0.25],
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert np.isclose(predictions[0], 14.5)
    assert np.isclose(predictions[1], 29.5)


def test_conditional_linear_blend_regressor_only_blends_matching_rules() -> None:
    train_frame = pd.DataFrame(
        {
            "code": ["OSJL3A7Y"],
            "sub_category": ["DPPUO5X2"],
            "horizon": [10],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "code": ["OSJL3A7Y", "OTHER"],
            "sub_category": ["DPPUO5X2", "DPPUO5X2"],
            "horizon": [10, 10],
        }
    )

    model = ConditionalLinearBlendRegressor(
        default_predictor=ConstantPredictor(10.0),
        slice_predictor=ConstantPredictor(30.0),
        blend_rules=[
            {"code": "OSJL3A7Y", "horizon": 10, "weight": 0.25},
        ],
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert model.supports_full_frame_prediction is True
    assert np.allclose(predictions, np.array([15.0, 10.0]))


def test_warm_start_last_target_blend_regressor_only_blends_seen_full_groups() -> None:
    train_frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "ts_index": [1, 2, 2],
            "code": ["A", "A", "A"],
            "sub_code": ["S1", "S1", "S2"],
            "sub_category": ["NQ58FVQM", "NQ58FVQM", "NQ58FVQM"],
            "horizon": [25, 25, 25],
            "y_target": [10.0, 20.0, 30.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "code": ["A", "A", "A"],
            "sub_code": ["S1", "S2", "S3"],
            "sub_category": ["NQ58FVQM", "NQ58FVQM", "NQ58FVQM"],
            "horizon": [25, 25, 25],
        }
    )

    base_predictor = WeightedGroupMeanRegressor(
        fallback_levels=[
            ["code", "sub_category", "horizon"],
            ["horizon"],
        ]
    )
    model = WarmStartLastTargetBlendRegressor(
        base_predictor=base_predictor,
        blend_rules=[
            {"horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.1},
        ],
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert np.isclose(predictions[0], 20.0)
    assert np.isclose(predictions[1], 21.0)
    assert np.isclose(predictions[2], 20.0)


def test_warm_start_last_target_blend_regressor_rejects_invalid_alpha() -> None:
    train_frame = pd.DataFrame(
        {
            "id": [1],
            "ts_index": [1],
            "code": ["A"],
            "sub_code": ["S1"],
            "sub_category": ["NQ58FVQM"],
            "horizon": [25],
            "y_target": [10.0],
            "weight": [1.0],
        }
    )

    model = WarmStartLastTargetBlendRegressor(
        base_predictor=WeightedGroupMeanRegressor(fallback_levels=[["horizon"]]),
        blend_rules=[
            {"horizon": 25, "sub_category": "NQ58FVQM", "alpha": 1.1},
        ],
    )

    try:
        model.fit(train_frame)
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("expected invalid alpha to raise ValueError")


def test_warm_start_last_target_blend_regressor_supports_optional_code_filters() -> None:
    train_frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "ts_index": [1, 2, 1, 2],
            "code": ["A", "A", "B", "B"],
            "sub_code": ["S1", "S1", "S1", "S1"],
            "sub_category": ["NQ58FVQM", "NQ58FVQM", "NQ58FVQM", "NQ58FVQM"],
            "horizon": [25, 25, 25, 25],
            "y_target": [10.0, 30.0, 10.0, 30.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "code": ["A", "B"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["NQ58FVQM", "NQ58FVQM"],
            "horizon": [25, 25],
        }
    )

    model = WarmStartLastTargetBlendRegressor(
        base_predictor=WeightedGroupMeanRegressor(
            fallback_levels=[
                ["code", "sub_category", "horizon"],
                ["horizon"],
            ]
        ),
        blend_rules=[
            {"code": "A", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.2},
        ],
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert np.isclose(predictions[0], 22.0)
    assert np.isclose(predictions[1], 20.0)


def test_warm_start_last_target_blend_regressor_allows_repeated_future_groups() -> None:
    train_frame = pd.DataFrame(
        {
            "id": [1, 2],
            "ts_index": [1, 2],
            "code": ["A", "A"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["NQ58FVQM", "NQ58FVQM"],
            "horizon": [25, 25],
            "y_target": [10.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "code": ["A", "A"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["NQ58FVQM", "NQ58FVQM"],
            "horizon": [25, 25],
            "ts_index": [3, 4],
        }
    )

    model = WarmStartLastTargetBlendRegressor(
        base_predictor=WeightedGroupMeanRegressor(
            fallback_levels=[
                ["code", "sub_category", "horizon"],
                ["horizon"],
            ]
        ),
        blend_rules=[
            {"horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.2},
        ],
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert np.allclose(predictions, np.array([22.0, 22.0]))


def test_recent_mean_delta_regressor_adds_recent_offset_on_selected_horizons() -> None:
    train_frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "ts_index": [1, 2, 3, 1, 2, 3],
            "code": ["A", "A", "A", "A", "A", "A"],
            "sub_code": ["S1", "S1", "S1", "S1", "S1", "S1"],
            "sub_category": ["NQ58FVQM", "NQ58FVQM", "NQ58FVQM", "PHHHVYZI", "PHHHVYZI", "PHHHVYZI"],
            "horizon": [25, 25, 25, 1, 1, 1],
            "y_target": [0.0, 10.0, 20.0, 5.0, 5.0, 50.0],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "code": ["A", "A"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["NQ58FVQM", "PHHHVYZI"],
            "horizon": [25, 1],
        }
    )

    model = RecentMeanDeltaRegressor(
        base_predictor=WeightedGroupMeanRegressor(
            fallback_levels=[
                ["code", "sub_category", "horizon"],
                ["horizon"],
            ]
        ),
        recent_window=1,
        min_recent_rows=1,
        beta=0.5,
        horizons=[25],
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert np.isclose(predictions[0], 15.0)
    assert np.isclose(predictions[1], 20.0)


def test_recent_mean_delta_regressor_supports_clipping_and_gating_controls() -> None:
    train_frame = pd.DataFrame(
        {
            "id": list(range(1, 10)),
            "ts_index": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "code": ["A", "A", "A", "B", "B", "B", "A", "A", "A"],
            "sub_code": ["S1"] * 9,
            "sub_category": [
                "NQ58FVQM",
                "NQ58FVQM",
                "NQ58FVQM",
                "NQ58FVQM",
                "NQ58FVQM",
                "NQ58FVQM",
                "PHHHVYZI",
                "PHHHVYZI",
                "PHHHVYZI",
            ],
            "horizon": [10, 10, 10, 10, 10, 10, 25, 25, 25],
            "y_target": [0.0, 0.0, 60.0, 0.0, 0.0, 20.0, 0.0, 0.0, 100.0],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "code": ["A", "B", "A"],
            "sub_code": ["S1", "S1", "S1"],
            "sub_category": ["NQ58FVQM", "NQ58FVQM", "PHHHVYZI"],
            "horizon": [10, 10, 25],
        }
    )

    model = RecentMeanDeltaRegressor(
        base_predictor=WeightedGroupMeanRegressor(
            fallback_levels=[
                ["code", "sub_category", "horizon"],
                ["horizon"],
            ]
        ),
        recent_window=1,
        min_recent_rows=1,
        min_recent_weight_sum=1.0,
        beta=0.15,
        beta_by_horizon={10: 0.5, 25: 0.25},
        clip_quantile_by_horizon={10: 0.5},
        apply_rules=[{"code": "A", "horizon": 10}],
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert np.isclose(model.clip_threshold_by_horizon_[10], 26.666666666666668)
    assert 25 not in model.clip_threshold_by_horizon_
    assert np.isclose(predictions[0], 33.333333333333336)
    assert np.isclose(predictions[1], 6.666666666666667)
    assert np.isclose(predictions[2], 16.666666666666668)
