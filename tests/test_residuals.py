from __future__ import annotations

import numpy as np
import pandas as pd

from ts_forecasting.residuals import (
    RAW_RESIDUAL_TARGET_COLUMN,
    FixedPrefixResidualCorrectionRegressor,
    NORMALIZED_RESIDUAL_TARGET_COLUMN,
    NormalizedPrefixResidualCorrectionRegressor,
    RESIDUAL_SCALE_COLUMN,
    apply_normalized_residual_corrections,
    build_slice_mask,
    build_residual_scale_lookup,
    compute_raw_residual_targets,
    compute_global_residual_scale,
    compute_normalized_residual_targets,
    detect_residual_feature_columns,
    attach_residual_scales,
    shrink_residual_corrections,
)
from ts_forecasting.baselines import WeightedGroupMeanRegressor
from ts_forecasting.constants import ID_COLUMN, TIME_COLUMN


class EchoResidualBasePredictor:
    supports_full_frame_prediction = True

    def fit(self, train_frame: pd.DataFrame) -> "EchoResidualBasePredictor":
        return self

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        return batch_frame["feature_a"].to_numpy(dtype=float)


def test_build_slice_mask_filters_requested_horizons_and_sub_categories() -> None:
    frame = pd.DataFrame(
        {
            "horizon": [1, 3, 10],
            "sub_category": ["A", "B", "A"],
            "code": ["X", "X", "Y"],
        }
    )

    mask = build_slice_mask(
        frame,
        horizons=[1, 3],
        sub_categories=["A"],
    )

    assert mask.tolist() == [True, False, False]


def test_detect_residual_feature_columns_excludes_banned_features() -> None:
    columns = ["ts_index", "feature_a", "feature_al", "feature_b"]

    residual_columns = detect_residual_feature_columns(
        columns,
        banned_feature_columns={"feature_al"},
    )

    assert residual_columns == ["ts_index", "feature_a", "feature_b"]


def test_shrink_residual_corrections_clips_and_scales() -> None:
    corrections = np.array([-10.0, -1.0, 2.0, 10.0])

    shrunk = shrink_residual_corrections(
        corrections,
        correction_scale=0.5,
        clip_abs=3.0,
    )

    assert np.allclose(shrunk, np.array([-1.5, -0.5, 1.0, 1.5]))


def test_attach_residual_scales_uses_primary_then_fallback_then_global() -> None:
    frame = pd.DataFrame(
        {
            "code": ["A", "Z", "Y"],
            "sub_category": ["cat_1", "cat_1", "cat_missing"],
            "horizon": [1, 1, 3],
        }
    )
    primary_lookup = pd.DataFrame(
        {
            "code": ["A"],
            "sub_category": ["cat_1"],
            "horizon": [1],
            RESIDUAL_SCALE_COLUMN: [2.0],
        }
    )
    fallback_lookup = pd.DataFrame(
        {
            "sub_category": ["cat_1"],
            "horizon": [1],
            RESIDUAL_SCALE_COLUMN: [3.0],
        }
    )

    enriched = attach_residual_scales(
        frame,
        primary_lookup=primary_lookup,
        fallback_lookup=fallback_lookup,
        global_scale=5.0,
    )

    assert enriched[RESIDUAL_SCALE_COLUMN].tolist() == [2.0, 3.0, 5.0]


def test_attach_residual_scales_preserves_input_index_order() -> None:
    frame = pd.DataFrame(
        {
            "code": ["B", "A", "Z"],
            "sub_category": ["cat_1", "cat_1", "cat_2"],
            "horizon": [1, 1, 3],
        },
        index=[11, 7, 42],
    )
    primary_lookup = pd.DataFrame(
        {
            "code": ["A", "B"],
            "sub_category": ["cat_1", "cat_1"],
            "horizon": [1, 1],
            RESIDUAL_SCALE_COLUMN: [2.0, 4.0],
        }
    )
    fallback_lookup = pd.DataFrame(
        {
            "sub_category": ["cat_2"],
            "horizon": [3],
            RESIDUAL_SCALE_COLUMN: [6.0],
        }
    )

    enriched = attach_residual_scales(
        frame,
        primary_lookup=primary_lookup,
        fallback_lookup=fallback_lookup,
        global_scale=5.0,
    )

    assert enriched.index.tolist() == [11, 7, 42]
    assert enriched[RESIDUAL_SCALE_COLUMN].tolist() == [4.0, 2.0, 6.0]


def test_build_residual_scale_lookup_and_normalized_targets_use_weighted_rms() -> None:
    frame = pd.DataFrame(
        {
            "code": ["A", "A", "B"],
            "sub_category": ["cat_1", "cat_1", "cat_2"],
            "horizon": [1, 1, 3],
            "y_target": [3.0, 4.0, 12.0],
            "weight": [1.0, 1.0, 4.0],
            "base_prediction": [2.0, 2.0, 6.0],
        }
    )

    lookup = build_residual_scale_lookup(
        frame,
        level_columns=["code", "sub_category", "horizon"],
    )
    scale_a = lookup.loc[lookup["code"] == "A", RESIDUAL_SCALE_COLUMN].item()
    global_scale = compute_global_residual_scale(frame)
    normalized = compute_normalized_residual_targets(
        attach_residual_scales(
            frame.drop(columns=["base_prediction"]),
            primary_lookup=lookup,
            fallback_lookup=build_residual_scale_lookup(
                frame,
                level_columns=["sub_category", "horizon"],
            ),
            global_scale=global_scale,
        ).assign(base_prediction=frame["base_prediction"])
    )

    assert np.isclose(scale_a, 5.0 / np.sqrt(2.0))
    assert np.isclose(global_scale, np.sqrt((9.0 + 16.0 + 4.0 * 144.0) / 6.0))
    assert np.isclose(
        normalized.loc[0, NORMALIZED_RESIDUAL_TARGET_COLUMN],
        (3.0 - 2.0) / scale_a,
    )


def test_compute_raw_residual_targets_subtracts_base_prediction() -> None:
    frame = pd.DataFrame(
        {
            "y_target": [3.0, -1.0],
            "base_prediction": [1.5, -0.5],
        }
    )

    residuals = compute_raw_residual_targets(frame)

    assert residuals[RAW_RESIDUAL_TARGET_COLUMN].tolist() == [1.5, -0.5]


def test_apply_normalized_residual_corrections_restores_raw_scale() -> None:
    corrected = apply_normalized_residual_corrections(
        np.array([-2.0, 0.5, 3.0]),
        scales=np.array([10.0, 4.0, 2.0]),
        correction_scale=0.5,
        clip_abs=1.0,
    )

    assert np.allclose(corrected, np.array([-5.0, 1.0, 1.0]))


def test_fixed_prefix_residual_regressor_only_adjusts_gated_rows() -> None:
    train_frame = pd.DataFrame(
        {
            "id": [f"row_{idx}" for idx in range(1, 9)],
            "code": ["A"] * 8,
            "sub_code": ["S1"] * 8,
            "sub_category": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "horizon": [1] * 8,
            "ts_index": [1, 2, 1, 2, 3, 4, 3, 4],
            "feature_a": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 50.0, 60.0],
            "y_target": [10.0, 10.0, 5.0, 5.0, 14.0, 16.0, 5.0, 5.0],
            "weight": [1.0] * 8,
        }
    )
    batch_frame = pd.DataFrame(
        {
            "id": ["future_gate", "future_ungated"],
            "code": ["A", "A"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["C1", "C2"],
            "horizon": [1, 1],
            "ts_index": [5, 5],
            "feature_a": [3.0, 100.0],
        }
    )

    base_predictor = WeightedGroupMeanRegressor(
        fallback_levels=[
            ["code", "sub_category", "horizon"],
            ["sub_category", "horizon"],
            ["horizon"],
        ]
    )
    base_predictor.fit(train_frame)
    base_predictions = base_predictor.predict_batch(batch_frame)

    model = FixedPrefixResidualCorrectionRegressor(
        base_predictor=WeightedGroupMeanRegressor(
            fallback_levels=[
                ["code", "sub_category", "horizon"],
                ["sub_category", "horizon"],
                ["horizon"],
            ]
        ),
        model_type="ridge",
        horizons=[1],
        sub_categories=["C1"],
        residual_train_steps=2,
        correction_scale=1.0,
        correction_clip_quantile=1.0,
        max_train_rows=10,
        alpha=1e-6,
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert model.training_rows_available_ == 2
    assert model.training_rows_used_ == 2
    assert model.clip_abs_ is not None and model.clip_abs_ > 0.0
    assert predictions[0] > base_predictions[0]
    assert np.isclose(predictions[1], base_predictions[1])


def test_normalized_prefix_residual_regressor_only_adjusts_gated_rows() -> None:
    train_frame = pd.DataFrame(
        {
            "id": [f"row_{idx}" for idx in range(1, 9)],
            "code": ["A"] * 8,
            "sub_code": ["S1"] * 8,
            "sub_category": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "horizon": [1] * 8,
            "ts_index": [1, 2, 1, 2, 3, 4, 3, 4],
            "feature_a": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 50.0, 60.0],
            "y_target": [10.0, 10.0, 5.0, 5.0, 14.0, 16.0, 5.0, 5.0],
            "weight": [1.0] * 8,
        }
    )
    batch_frame = pd.DataFrame(
        {
            "id": ["future_gate", "future_ungated"],
            "code": ["A", "A"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["C1", "C2"],
            "horizon": [1, 1],
            "ts_index": [5, 5],
            "feature_a": [3.0, 100.0],
        }
    )

    base_predictor = WeightedGroupMeanRegressor(
        fallback_levels=[
            ["code", "sub_category", "horizon"],
            ["sub_category", "horizon"],
            ["horizon"],
        ]
    )
    base_predictor.fit(train_frame)
    base_predictions = base_predictor.predict_batch(batch_frame)

    model = NormalizedPrefixResidualCorrectionRegressor(
        base_predictor=WeightedGroupMeanRegressor(
            fallback_levels=[
                ["code", "sub_category", "horizon"],
                ["sub_category", "horizon"],
                ["horizon"],
            ]
        ),
        model_type="ridge",
        horizons=[1],
        sub_categories=["C1"],
        residual_train_steps=2,
        oof_holdout_steps=1,
        oof_step_size=1,
        oof_min_train_steps=2,
        correction_scale=1.0,
        correction_clip_quantile=1.0,
        max_train_rows=10,
        alpha=1e-6,
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert model.supports_full_frame_prediction is True
    assert model.training_rows_available_ == 2
    assert model.training_rows_used_ == 2
    assert model.clip_abs_ is not None and model.clip_abs_ > 0.0
    assert predictions[0] > base_predictions[0]
    assert np.isclose(predictions[1], base_predictions[1])


def test_normalized_prefix_residual_regressor_skips_when_no_internal_oof_history() -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["row_1", "row_2"],
            "code": ["A", "A"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["C1", "C1"],
            "horizon": [1, 1],
            "ts_index": [1, 2],
            "feature_a": [0.0, 1.0],
            "y_target": [10.0, 20.0],
            "weight": [1.0, 1.0],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "id": ["future_gate"],
            "code": ["A"],
            "sub_code": ["S1"],
            "sub_category": ["C1"],
            "horizon": [1],
            "ts_index": [3],
            "feature_a": [2.0],
        }
    )

    base_predictor = WeightedGroupMeanRegressor(
        fallback_levels=[
            ["code", "sub_category", "horizon"],
            ["sub_category", "horizon"],
            ["horizon"],
        ]
    )
    base_predictor.fit(train_frame)
    base_predictions = base_predictor.predict_batch(batch_frame)

    model = NormalizedPrefixResidualCorrectionRegressor(
        base_predictor=WeightedGroupMeanRegressor(
            fallback_levels=[
                ["code", "sub_category", "horizon"],
                ["sub_category", "horizon"],
                ["horizon"],
            ]
        ),
        model_type="ridge",
        horizons=[1],
        sub_categories=["C1"],
        residual_train_steps=2,
        oof_holdout_steps=1,
        oof_step_size=1,
        oof_min_train_steps=2,
        correction_scale=1.0,
        correction_clip_quantile=1.0,
        max_train_rows=10,
        alpha=1e-6,
    ).fit(train_frame)

    predictions = model.predict_batch(batch_frame)

    assert model.training_rows_available_ == 0
    assert model.training_rows_used_ == 0
    assert model.clip_abs_ is None
    assert np.allclose(predictions, base_predictions)


def test_internal_oof_training_pool_preserves_sorted_validation_order() -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["b2", "a1", "b3", "a2", "b4", "a4", "a3", "b1"],
            "code": ["A"] * 8,
            "sub_code": ["S1"] * 8,
            "sub_category": ["C1"] * 8,
            "horizon": [1] * 8,
            "ts_index": [2, 1, 3, 2, 4, 4, 3, 1],
            "feature_a": [20.0, 10.0, 30.0, 21.0, 41.0, 40.0, 31.0, 11.0],
            "y_target": [2.0, 1.0, 3.0, 2.1, 4.1, 4.0, 3.1, 1.1],
            "weight": [1.0] * 8,
        }
    )

    model = NormalizedPrefixResidualCorrectionRegressor(
        base_predictor=EchoResidualBasePredictor(),
        model_type="ridge",
        horizons=[1],
        sub_categories=["C1"],
        residual_train_steps=4,
        oof_holdout_steps=1,
        oof_step_size=1,
        oof_min_train_steps=2,
        correction_scale=1.0,
        correction_clip_quantile=1.0,
        max_train_rows=20,
        alpha=1e-6,
    )

    oof_pool = model._build_internal_oof_training_pool(train_frame)

    assert len(oof_pool) == 4
    assert oof_pool[[TIME_COLUMN, "id"]].values.tolist() == [
        [3, "a3"],
        [3, "b3"],
        [4, "a4"],
        [4, "b4"],
    ]
    assert oof_pool["base_prediction"].tolist() == [31.0, 30.0, 40.0, 41.0]
    assert NORMALIZED_RESIDUAL_TARGET_COLUMN in oof_pool.columns


def test_internal_oof_training_pool_sorts_once_up_front(monkeypatch) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["b2", "a1", "b3", "a2", "b4", "a4", "a3", "b1"],
            "code": ["A"] * 8,
            "sub_code": ["S1"] * 8,
            "sub_category": ["C1"] * 8,
            "horizon": [1] * 8,
            "ts_index": [2, 1, 3, 2, 4, 4, 3, 1],
            "feature_a": [20.0, 10.0, 30.0, 21.0, 41.0, 40.0, 31.0, 11.0],
            "y_target": [2.0, 1.0, 3.0, 2.1, 4.1, 4.0, 3.1, 1.1],
            "weight": [1.0] * 8,
        }
    )
    model = NormalizedPrefixResidualCorrectionRegressor(
        base_predictor=EchoResidualBasePredictor(),
        model_type="ridge",
        horizons=[1],
        sub_categories=["C1"],
        residual_train_steps=4,
        oof_holdout_steps=1,
        oof_step_size=1,
        oof_min_train_steps=2,
        correction_scale=1.0,
        correction_clip_quantile=1.0,
        max_train_rows=20,
        alpha=1e-6,
    )

    original_sort_values = pd.DataFrame.sort_values
    ordered_sort_calls = 0

    def counting_sort_values(self, by=None, *args, **kwargs):
        nonlocal ordered_sort_calls
        if list(by) == [TIME_COLUMN, ID_COLUMN]:
            ordered_sort_calls += 1
        return original_sort_values(self, by=by, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sort_values", counting_sort_values)

    model._build_internal_oof_training_pool(train_frame)

    assert ordered_sort_calls == 1
