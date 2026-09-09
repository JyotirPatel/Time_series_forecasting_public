from __future__ import annotations

import pandas as pd

from scripts import evaluate_residual_models
from ts_forecasting.metrics import weighted_rmse_score_from_sums
from ts_forecasting.residuals import NORMALIZED_RESIDUAL_TARGET_COLUMN


def test_residual_script_defaults_target_current_incumbent_with_one_conservative_config() -> None:
    assert evaluate_residual_models.DEFAULT_BASE_SUMMARY_PATH == (
        "outputs/advanced_models/h13_pentaslice_h3_followup_exact_20260410/"
        "forward_cv_h360_f4_s360_code_sub_code/summary.json"
    )
    assert (
        evaluate_residual_models.DEFAULT_BASELINE_NAME
        == "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_h3_beta_up_v1"
    )
    assert list(evaluate_residual_models.RESIDUAL_CONFIGS) == [
        "ridge_oofnorm_pentaslice_h13_s008_q85_a50_unweighted_v1",
        "ridge_oofnorm_longwatch_h1025_s003_q80_a100_unweighted_v1",
    ]

    config = evaluate_residual_models.RESIDUAL_CONFIGS[
        "ridge_oofnorm_pentaslice_h13_s008_q85_a50_unweighted_v1"
    ]
    assert config["model_type"] == "ridge"
    assert config["horizons"] == [1, 3]
    assert config["sub_categories"] == [
        "DPPUO5X2",
        "PZ9S1Z4V",
        "PHHHVYZI",
        "NQ58FVQM",
        "V8BKY1IV",
    ]
    assert config["correction_scale"] == 0.08
    assert config["correction_clip_quantile"] == 0.85
    assert config["alpha"] == 50.0

    longwatch = evaluate_residual_models.RESIDUAL_CONFIGS[
        "ridge_oofnorm_longwatch_h1025_s003_q80_a100_unweighted_v1"
    ]
    assert longwatch["model_type"] == "ridge"
    assert longwatch["codes"] == ["OSJL3A7Y", "X9BZ68VQ"]
    assert longwatch["horizons"] == [10, 25]
    assert longwatch["sub_categories"] == [
        "DPPUO5X2",
        "PHHHVYZI",
        "PZ9S1Z4V",
        "NQ58FVQM",
    ]
    assert longwatch["correction_scale"] == 0.03
    assert longwatch["correction_clip_quantile"] == 0.8
    assert longwatch["alpha"] == 100.0


def test_fit_residual_model_avoids_fit_time_sample_weight() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.fit_kwargs: dict[str, object] | None = None
            self.fit_columns: list[str] | None = None

        def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> "RecordingModel":
            self.fit_columns = list(X.columns)
            self.fit_kwargs = dict(kwargs)
            return self

    train_frame = pd.DataFrame(
        {
            "code": ["A", "B"],
            "sub_category": ["C1", "C2"],
            "horizon": [1, 3],
            "ts_index": [100, 101],
            "feature_a": [1.0, 2.0],
            NORMALIZED_RESIDUAL_TARGET_COLUMN: [0.1, -0.2],
            "weight": [1.0, 2.0],
        }
    )

    model = RecordingModel()
    evaluate_residual_models.fit_residual_model(
        model,
        train_frame,
        numeric_columns=["ts_index", "feature_a"],
    )

    assert model.fit_columns == [
        "code",
        "sub_category",
        "horizon",
        "ts_index",
        "feature_a",
    ]
    assert model.fit_kwargs == {}


def test_aggregate_config_scores_preserves_cold_and_warm_aggregate_metrics() -> None:
    rows = [
        {
            "overall_score": 0.4,
            "overall_error_sum": 3.0,
            "overall_denom_sum": 5.0,
            "base_overall_score": 0.3,
            "base_overall_error_sum": 4.0,
            "base_overall_denom_sum": 5.0,
            "cold_start_score": 0.5,
            "cold_start_error_sum": 1.0,
            "cold_start_denom_sum": 2.0,
            "warm_start_score": 0.25,
            "warm_start_error_sum": 2.0,
            "warm_start_denom_sum": 3.0,
            "cold_start_rows": 2,
            "warm_start_rows": 3,
            "rows_changed": 10,
            "slice_validation_rows": 5,
            "oof_training_rows_available": 100,
            "training_rows_used": 80,
        },
        {
            "overall_score": 0.2,
            "overall_error_sum": 1.0,
            "overall_denom_sum": 5.0,
            "base_overall_score": 0.1,
            "base_overall_error_sum": 2.0,
            "base_overall_denom_sum": 5.0,
            "cold_start_score": 0.4,
            "cold_start_error_sum": 0.5,
            "cold_start_denom_sum": 2.0,
            "warm_start_score": 0.1,
            "warm_start_error_sum": 0.5,
            "warm_start_denom_sum": 3.0,
            "cold_start_rows": 2,
            "warm_start_rows": 3,
            "rows_changed": 20,
            "slice_validation_rows": 5,
            "oof_training_rows_available": 90,
            "training_rows_used": 70,
        },
    ]

    aggregate = evaluate_residual_models._aggregate_config_scores(rows)

    assert aggregate["aggregate_cold_start_score"] == weighted_rmse_score_from_sums(
        error_sum=1.5,
        denom_sum=4.0,
    )
    assert aggregate["aggregate_warm_start_score"] == weighted_rmse_score_from_sums(
        error_sum=2.5,
        denom_sum=6.0,
    )


def test_sample_frame_is_stable_across_input_row_order() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"row_{idx}" for idx in range(8)],
            "ts_index": [1, 1, 2, 2, 3, 3, 4, 4],
            "feature_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    reordered = frame.iloc[[7, 3, 5, 1, 6, 2, 4, 0]].reset_index(drop=True)

    sampled = evaluate_residual_models._sample_frame(frame, max_rows=4)
    resampled = evaluate_residual_models._sample_frame(reordered, max_rows=4)

    assert sampled["id"].tolist() == resampled["id"].tolist()
