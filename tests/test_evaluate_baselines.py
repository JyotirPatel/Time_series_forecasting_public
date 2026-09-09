from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from ts_forecasting.constants import TARGET_COLUMN, WEIGHT_COLUMN
from ts_forecasting.metrics import weighted_rmse_breakdown, weighted_rmse_score_from_sums

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_baselines, predict_baseline


def _make_scored_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_write_scored_fold_frame_writes_parquet(tmp_path: Path) -> None:
    scored = pd.DataFrame(
        {
            "id": ["a", "b"],
            "ts_index": [1, 2],
            "prediction": [0.1, 0.2],
            "is_cold_start": [True, False],
        }
    )

    output_path = evaluate_baselines.write_scored_fold_frame(
        scored,
        output_dir=tmp_path,
        baseline_name="demo_baseline",
        fold_name="fold_01_train_1_valid_2_3",
    )

    assert output_path == tmp_path / "demo_baseline_fold_01_train_1_valid_2_3_scored.parquet"
    restored = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(restored, scored)


def test_evaluate_baseline_uses_configured_transformer(monkeypatch) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["t1"],
            "code": ["A"],
            "sub_code": ["X"],
            "sub_category": ["S"],
            "horizon": [1],
            "ts_index": [1],
            "feature_a": [2.0],
            TARGET_COLUMN: [0.0],
            WEIGHT_COLUMN: [1.0],
        }
    )
    validation_frame = pd.DataFrame(
        {
            "id": ["v1", "v2"],
            "code": ["A", "A"],
            "sub_code": ["X", "X"],
            "sub_category": ["S", "S"],
            "horizon": [1, 1],
            "ts_index": [2, 3],
            "feature_a": [5.0, 7.0],
            TARGET_COLUMN: [6.0, 8.0],
            WEIGHT_COLUMN: [1.0, 1.0],
        }
    )

    class AddOneTransformer:
        supports_full_frame_prediction = True

        def fit(self, train_frame: pd.DataFrame) -> "AddOneTransformer":
            return self

        def transform_batch(self, batch_frame: pd.DataFrame) -> pd.DataFrame:
            transformed = batch_frame.copy()
            transformed["feature_a"] = transformed["feature_a"] + 1.0
            return transformed

    class FeatureEchoPredictor:
        supports_full_frame_prediction = True

        def fit(self, train_frame: pd.DataFrame) -> "FeatureEchoPredictor":
            return self

        def predict_batch(self, batch_frame: pd.DataFrame):
            return batch_frame["feature_a"].to_numpy()

    monkeypatch.setattr(
        evaluate_baselines,
        "build_transformer",
        lambda config: AddOneTransformer(),
    )
    monkeypatch.setattr(
        evaluate_baselines,
        "build_predictor",
        lambda config: FeatureEchoPredictor(),
    )

    _, predictions = evaluate_baselines.evaluate_baseline(
        train_frame,
        validation_frame,
        name="demo",
        config={"regressor": "group_mean"},
    )

    assert predictions["prediction"].tolist() == [6.0, 8.0]


def test_causal_history_histgbm_variant_is_exposed_and_preserves_transformer_knobs() -> None:
    available_columns = [
        "id",
        "code",
        "sub_code",
        "sub_category",
        "horizon",
        "ts_index",
        "feature_a",
        "feature_al",
        "feature_b",
        "y_target",
        "weight",
    ]
    expected_name = (
        "histgbm_oofnorm_pentaslice_h13_s008_q85_lr005_md6_i200_leaf200_causal_history_v1"
    )

    assert expected_name in evaluate_baselines.BASELINE_CONFIGS
    assert expected_name in predict_baseline.BASELINE_CONFIGS

    for module in (evaluate_baselines, predict_baseline):
        transformer = module.build_transformer(module.BASELINE_CONFIGS[expected_name])
        context_columns = module.resolve_required_columns(
            module.BASELINE_CONFIGS[expected_name],
            available_columns=available_columns,
        )

        assert transformer.level_columns == ["code", "sub_category", "horizon"]
        assert transformer.history_features == ["last", "mean", "delta"]
        assert transformer.banned_feature_columns == {"feature_al"}
        assert context_columns == [
            "id",
            "code",
            "sub_code",
            "sub_category",
            "horizon",
            "ts_index",
            "feature_a",
            "feature_b",
        ]


def test_causal_history_h25_variants_are_exposed_and_preserve_transformer_knobs() -> None:
    available_columns = [
        "id",
        "code",
        "sub_code",
        "sub_category",
        "horizon",
        "ts_index",
        "feature_a",
        "feature_al",
        "feature_b",
        "y_target",
        "weight",
    ]
    expected_names = {
        "histgbm_oofnorm_allsub_h25_s005_q80_lr005_md6_i200_leaf200_causal_last_v1",
        "huber_oofnorm_allsub_h25_s005_q80_a1_eps135_i200_causal_last_v1",
    }

    assert expected_names <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected_names <= predict_baseline.BASELINE_CONFIGS.keys()

    for module in (evaluate_baselines, predict_baseline):
        for name in expected_names:
            transformer = module.build_transformer(module.BASELINE_CONFIGS[name])
            context_columns = module.resolve_required_columns(
                module.BASELINE_CONFIGS[name],
                available_columns=available_columns,
            )

            assert transformer.level_columns == ["code", "sub_category", "horizon"]
            assert transformer.history_features == ["last"]
            assert transformer.banned_feature_columns == {"feature_al"}
            assert context_columns == [
                "id",
                "code",
                "sub_code",
                "sub_category",
                "horizon",
                "ts_index",
                "feature_a",
                "feature_b",
            ]


def test_build_stress_block_reports_fold_subset_per_horizon_and_watchlists() -> None:
    fold_stress_inputs = [
        {
            "name": "fold_01_train_30_valid_31_35",
            "results": {
                "demo": _make_scored_frame(
                    [
                        {
                            "code": "OTHER",
                            "sub_category": "DPPUO5X2",
                            "horizon": 1,
                            TARGET_COLUMN: 10.0,
                            "prediction": 10.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": True,
                        }
                    ]
                )
            },
        },
        {
            "name": "fold_02_train_35_valid_36_40",
            "results": {
                "demo": _make_scored_frame(
                    [
                        {
                            "code": "OSJL3A7Y",
                            "sub_category": "DPPUO5X2",
                            "horizon": 10,
                            TARGET_COLUMN: 10.0,
                            "prediction": 9.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": False,
                        },
                        {
                            "code": "X9BZ68VQ",
                            "sub_category": "PHHHVYZI",
                            "horizon": 25,
                            TARGET_COLUMN: 10.0,
                            "prediction": 8.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": True,
                        },
                    ]
                )
            },
        },
        {
            "name": "fold_03_train_40_valid_41_45",
            "results": {
                "demo": _make_scored_frame(
                    [
                        {
                            "code": "OSJL3A7Y",
                            "sub_category": "PZ9S1Z4V",
                            "horizon": 25,
                            TARGET_COLUMN: 20.0,
                            "prediction": 18.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": False,
                        },
                        {
                            "code": "X9BZ68VQ",
                            "sub_category": "DPPUO5X2",
                            "horizon": 10,
                            TARGET_COLUMN: 20.0,
                            "prediction": 19.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": True,
                        },
                    ]
                )
            },
        },
    ]

    stress = evaluate_baselines.build_stress_block(
        fold_stress_inputs,
        baseline_names=["demo"],
    )

    assert stress["fold_sets"]["fold_02_03"] == ["fold_02", "fold_03"]

    fold_02_03 = stress["baselines"]["demo"]["fold_sets"]["fold_02_03"]
    expected_breakdown = weighted_rmse_breakdown(
        [10.0, 10.0, 20.0, 20.0],
        [9.0, 8.0, 18.0, 19.0],
        [1.0, 1.0, 1.0, 1.0],
    )
    assert fold_02_03["aggregate_overall_score"] == expected_breakdown.score
    assert fold_02_03["aggregate_cold_start_score"] == weighted_rmse_score_from_sums(
        error_sum=5.0,
        denom_sum=500.0,
    )
    assert fold_02_03["aggregate_warm_start_score"] == weighted_rmse_score_from_sums(
        error_sum=5.0,
        denom_sum=500.0,
    )

    horizon_10 = stress["baselines"]["demo"]["per_horizon"]["10"]
    assert horizon_10["total_validation_rows"] == 2
    assert horizon_10["total_cold_start_rows"] == 1
    assert horizon_10["total_warm_start_rows"] == 1

    osjl_watch = stress["baselines"]["demo"]["watchlist_slices"]["osjl3a7y_long_core"]
    assert osjl_watch["total_validation_rows"] == 2
    assert osjl_watch["aggregate_warm_start_score"] == weighted_rmse_score_from_sums(
        error_sum=5.0,
        denom_sum=500.0,
    )

    x9_h10_watch = stress["baselines"]["demo"]["watchlist_slices"]["x9bz68vq_h10_core"]
    assert x9_h10_watch["total_validation_rows"] == 1
    assert x9_h10_watch["aggregate_cold_start_score"] == weighted_rmse_score_from_sums(
        error_sum=1.0,
        denom_sum=400.0,
    )


def test_incremental_stress_accumulator_matches_batch_builder() -> None:
    fold_stress_inputs = [
        {
            "name": "fold_01_train_30_valid_31_35",
            "results": {
                "demo": _make_scored_frame(
                    [
                        {
                            "code": "OTHER",
                            "sub_category": "DPPUO5X2",
                            "horizon": 1,
                            TARGET_COLUMN: 10.0,
                            "prediction": 10.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": True,
                        }
                    ]
                )
            },
        },
        {
            "name": "fold_02_train_35_valid_36_40",
            "results": {
                "demo": _make_scored_frame(
                    [
                        {
                            "code": "OSJL3A7Y",
                            "sub_category": "DPPUO5X2",
                            "horizon": 10,
                            TARGET_COLUMN: 10.0,
                            "prediction": 9.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": False,
                        },
                        {
                            "code": "X9BZ68VQ",
                            "sub_category": "PHHHVYZI",
                            "horizon": 25,
                            TARGET_COLUMN: 10.0,
                            "prediction": 8.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": True,
                        },
                    ]
                )
            },
        },
        {
            "name": "fold_03_train_40_valid_41_45",
            "results": {
                "demo": _make_scored_frame(
                    [
                        {
                            "code": "OSJL3A7Y",
                            "sub_category": "PZ9S1Z4V",
                            "horizon": 25,
                            TARGET_COLUMN: 20.0,
                            "prediction": 18.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": False,
                        },
                        {
                            "code": "X9BZ68VQ",
                            "sub_category": "DPPUO5X2",
                            "horizon": 10,
                            TARGET_COLUMN: 20.0,
                            "prediction": 19.0,
                            WEIGHT_COLUMN: 1.0,
                            "is_cold_start": True,
                        },
                    ]
                )
            },
        },
    ]

    accumulator = evaluate_baselines.initialize_stress_accumulator(["demo"])
    for fold_input in fold_stress_inputs:
        for baseline_name, scored in fold_input["results"].items():
            evaluate_baselines.update_stress_accumulator(
                accumulator,
                fold_name=fold_input["name"],
                baseline_name=baseline_name,
                scored=scored,
            )

    assert evaluate_baselines.finalize_stress_block(accumulator) == evaluate_baselines.build_stress_block(
        fold_stress_inputs,
        baseline_names=["demo"],
    )


def test_current_incumbent_and_residual_predictors_support_full_frame_prediction() -> None:
    for name in (
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_h3_beta_up_v1",
        "ridge_triweak_h13_tw720_s022_q90",
        "ridge_oofnorm_pentaslice_h13_s008_q85_a50_unweighted_v1",
        "winner_plus_longwatch_expert_blend_v1",
    ):
        predictor = evaluate_baselines.build_predictor(evaluate_baselines.BASELINE_CONFIGS[name])
        assert getattr(predictor, "supports_full_frame_prediction", False) is True


def test_recent_delta_clip_variants_are_exposed_in_script_registries() -> None:
    expected_variants = {
        "smoothed_stable_horizon_blend_last_target_recent_delta_clip_v1",
        "smoothed_stable_horizon_blend_last_target_recent_delta_clip_hbeta_v1",
        "smoothed_stable_horizon_blend_last_target_recent_delta_clip_osjlx9_v1",
    }

    assert expected_variants.issubset(evaluate_baselines.BASELINE_CONFIGS)
    assert expected_variants.issubset(predict_baseline.BASELINE_CONFIGS)


def test_recent_delta_builder_preserves_extended_knobs_in_both_scripts() -> None:
    config = {
        "regressor": "recent_mean_delta",
        "base_predictor": {"regressor": "group_mean", "fallback_levels": [["horizon"]]},
        "recent_window": 180,
        "min_recent_rows": 50,
        "beta": 0.15,
        "horizons": [10, 25],
        "beta_by_horizon": {10: 0.08, 25: 0.15},
        "clip_quantile_by_horizon": {10: 0.8, 25: 0.9},
        "apply_rules": [{"code": "OSJL3A7Y", "horizon": 10}],
        "min_recent_weight_sum": 25.0,
    }

    for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
        predictor = builder(config)
        assert predictor.beta_by_horizon == {10: 0.08, 25: 0.15}
        assert predictor.clip_quantile_by_horizon == {10: 0.8, 25: 0.9}
        assert predictor.apply_rules == [{"code": "OSJL3A7Y", "horizon": 10}]
        assert predictor.min_recent_weight_sum == 25.0


def test_narrow_osjl_x9_warm_probe_variants_are_exposed_and_preserve_rules() -> None:
    expected_rules = {
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_osjl_dppu10_a003_v1": [
            {"code": "OSJL3A7Y", "horizon": 10, "sub_category": "DPPUO5X2", "alpha": 0.03}
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_osjl_ph10_a003_v1": [
            {"code": "OSJL3A7Y", "horizon": 10, "sub_category": "PHHHVYZI", "alpha": 0.03}
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_osjl_pz10_a003_v1": [
            {"code": "OSJL3A7Y", "horizon": 10, "sub_category": "PZ9S1Z4V", "alpha": 0.03}
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_x9_ph10_a003_v1": [
            {"code": "X9BZ68VQ", "horizon": 10, "sub_category": "PHHHVYZI", "alpha": 0.03}
        ],
    }

    assert expected_rules.keys() <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected_rules.keys() <= predict_baseline.BASELINE_CONFIGS.keys()

    for name, blend_rules in expected_rules.items():
        for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
            predictor = builder(evaluate_baselines.BASELINE_CONFIGS[name])
            assert predictor.blend_rules == blend_rules
            assert predictor.base_predictor.blend_rules == [
                {"code": "K7Y1TTAH", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.15},
                {"code": "K7Y1TTAH", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.20},
                {"code": "K7Y1TTAH", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.03},
            ]


def test_short_horizon_recent_delta_variants_are_exposed_and_preserve_knobs() -> None:
    expected = {
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_dppu_v1": [
            {"sub_category": "DPPUO5X2", "horizon": 1},
            {"sub_category": "DPPUO5X2", "horizon": 3},
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_v1": [
            {"sub_category": "DPPUO5X2", "horizon": 1},
            {"sub_category": "DPPUO5X2", "horizon": 3},
            {"sub_category": "PZ9S1Z4V", "horizon": 1},
            {"sub_category": "PZ9S1Z4V", "horizon": 3},
            {"sub_category": "PHHHVYZI", "horizon": 1},
            {"sub_category": "PHHHVYZI", "horizon": 3},
        ],
    }

    assert expected.keys() <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected.keys() <= predict_baseline.BASELINE_CONFIGS.keys()

    for name, apply_rules in expected.items():
        for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
            predictor = builder(evaluate_baselines.BASELINE_CONFIGS[name])
            assert predictor.horizons == [1, 3]
            assert predictor.beta_by_horizon == {1: 0.04, 3: 0.06}
            assert predictor.clip_quantile_by_horizon == {1: 0.7, 3: 0.8}
            assert predictor.apply_rules == apply_rules
            assert predictor.min_recent_rows == 100
            assert predictor.min_recent_weight_sum == 50.0


def test_short_horizon_recent_delta_followup_variants_are_exposed_and_preserve_knobs() -> None:
    expected = {
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_lo_v1": {
            "beta_by_horizon": {1: 0.03, 3: 0.05},
            "recent_window": 180,
        },
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_v1": {
            "beta_by_horizon": {1: 0.05, 3: 0.08},
            "recent_window": 180,
        },
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_rw360_v1": {
            "beta_by_horizon": {1: 0.04, 3: 0.06},
            "recent_window": 360,
        },
    }

    assert expected.keys() <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected.keys() <= predict_baseline.BASELINE_CONFIGS.keys()

    for name, knobs in expected.items():
        for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
            predictor = builder(evaluate_baselines.BASELINE_CONFIGS[name])
            assert predictor.horizons == [1, 3]
            assert predictor.apply_rules == [
                {"sub_category": "DPPUO5X2", "horizon": 1},
                {"sub_category": "DPPUO5X2", "horizon": 3},
                {"sub_category": "PZ9S1Z4V", "horizon": 1},
                {"sub_category": "PZ9S1Z4V", "horizon": 3},
                {"sub_category": "PHHHVYZI", "horizon": 1},
                {"sub_category": "PHHHVYZI", "horizon": 3},
            ]
            assert predictor.beta_by_horizon == knobs["beta_by_horizon"]
            assert predictor.recent_window == knobs["recent_window"]
            assert predictor.clip_quantile_by_horizon == {1: 0.7, 3: 0.8}
            assert predictor.min_recent_rows == 100
            assert predictor.min_recent_weight_sum == 50.0


def test_short_horizon_recent_delta_pentaslice_variants_are_exposed_and_preserve_rules() -> None:
    expected = {
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_v1": {
            "beta_by_horizon": {1: 0.04, 3: 0.06},
        },
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_v1": {
            "beta_by_horizon": {1: 0.05, 3: 0.08},
        },
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_h3_beta_up_v1": {
            "beta_by_horizon": {1: 0.05, 3: 0.10},
        },
    }
    expected_rules = [
        {"sub_category": "DPPUO5X2", "horizon": 1},
        {"sub_category": "DPPUO5X2", "horizon": 3},
        {"sub_category": "PZ9S1Z4V", "horizon": 1},
        {"sub_category": "PZ9S1Z4V", "horizon": 3},
        {"sub_category": "PHHHVYZI", "horizon": 1},
        {"sub_category": "PHHHVYZI", "horizon": 3},
        {"sub_category": "NQ58FVQM", "horizon": 1},
        {"sub_category": "NQ58FVQM", "horizon": 3},
        {"sub_category": "V8BKY1IV", "horizon": 1},
        {"sub_category": "V8BKY1IV", "horizon": 3},
    ]

    assert expected.keys() <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected.keys() <= predict_baseline.BASELINE_CONFIGS.keys()

    for name, knobs in expected.items():
        for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
            predictor = builder(evaluate_baselines.BASELINE_CONFIGS[name])
            assert predictor.horizons == [1, 3]
            assert predictor.apply_rules == expected_rules
            assert predictor.beta_by_horizon == knobs["beta_by_horizon"]
            assert predictor.recent_window == 180
            assert predictor.clip_quantile_by_horizon == {1: 0.7, 3: 0.8}
            assert predictor.min_recent_rows == 100
            assert predictor.min_recent_weight_sum == 50.0


def test_short_horizon_non_ridge_residual_variants_are_exposed_and_preserve_knobs() -> None:
    expected = {
        "huber_oofnorm_pentaslice_h13_s008_q85_a1_eps135_i200_v1": {
            "model_type": "huber",
            "alpha": 1.0,
            "epsilon": 1.35,
            "max_iter": 200,
        },
        "histgbm_oofnorm_pentaslice_h13_s008_q85_lr005_md6_i200_leaf200_v1": {
            "model_type": "hist_gbm",
            "learning_rate": 0.05,
            "max_depth": 6,
            "max_iter": 200,
            "min_samples_leaf": 200,
        },
    }

    assert expected.keys() <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected.keys() <= predict_baseline.BASELINE_CONFIGS.keys()

    for name, knobs in expected.items():
        for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
            predictor = builder(evaluate_baselines.BASELINE_CONFIGS[name])
            assert predictor.model_type == knobs["model_type"]
            assert predictor.horizons == [1, 3]
            assert predictor.sub_categories == [
                "DPPUO5X2",
                "PZ9S1Z4V",
                "PHHHVYZI",
                "NQ58FVQM",
                "V8BKY1IV",
            ]
            assert predictor.correction_scale == 0.08
            assert predictor.correction_clip_quantile == 0.85
            assert predictor.max_train_rows == 180000
            if knobs["model_type"] == "huber":
                assert predictor.alpha == 1.0
                assert predictor.epsilon == 1.35
                assert predictor.max_iter == 200
            else:
                assert predictor.learning_rate == 0.05
                assert predictor.max_depth == 6
                assert predictor.max_iter == 200
                assert predictor.min_samples_leaf == 200


def test_short_horizon_osjl_x9_warm_probe_variants_are_exposed_and_preserve_rules() -> None:
    expected_rules = {
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_osjl_h3_core_a003_v1": [
            {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "DPPUO5X2", "alpha": 0.03},
            {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "PHHHVYZI", "alpha": 0.03},
            {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "PZ9S1Z4V", "alpha": 0.03},
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_osjl_h3_core_a005_v1": [
            {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "DPPUO5X2", "alpha": 0.05},
            {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "PHHHVYZI", "alpha": 0.05},
            {"code": "OSJL3A7Y", "horizon": 3, "sub_category": "PZ9S1Z4V", "alpha": 0.05},
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_x9_h3_core_a003_v1": [
            {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "DPPUO5X2", "alpha": 0.03},
            {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "PHHHVYZI", "alpha": 0.03},
            {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "PZ9S1Z4V", "alpha": 0.03},
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_x9_h3_core_a005_v1": [
            {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "DPPUO5X2", "alpha": 0.05},
            {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "PHHHVYZI", "alpha": 0.05},
            {"code": "X9BZ68VQ", "horizon": 3, "sub_category": "PZ9S1Z4V", "alpha": 0.05},
        ],
    }

    assert expected_rules.keys() <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected_rules.keys() <= predict_baseline.BASELINE_CONFIGS.keys()

    for name, blend_rules in expected_rules.items():
        for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
            predictor = builder(evaluate_baselines.BASELINE_CONFIGS[name])
            assert predictor.blend_rules == blend_rules
            assert predictor.base_predictor.beta_by_horizon == {1: 0.05, 3: 0.08}
            assert predictor.base_predictor.apply_rules == [
                {"sub_category": "DPPUO5X2", "horizon": 1},
                {"sub_category": "DPPUO5X2", "horizon": 3},
                {"sub_category": "PZ9S1Z4V", "horizon": 1},
                {"sub_category": "PZ9S1Z4V", "horizon": 3},
                {"sub_category": "PHHHVYZI", "horizon": 1},
                {"sub_category": "PHHHVYZI", "horizon": 3},
            ]


def test_long_horizon_nq58_warm_probe_variants_are_exposed_and_preserve_rules() -> None:
    expected_rules = {
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_osjl_h25_nq_a003_v1": [
            {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.03}
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_osjl_h25_nq_a005_v1": [
            {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.05}
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_x9_h25_nq_a003_v1": [
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.03}
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_trislice_beta_hi_x9_h25_nq_a005_v1": [
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.05}
        ],
    }

    assert expected_rules.keys() <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected_rules.keys() <= predict_baseline.BASELINE_CONFIGS.keys()

    for name, blend_rules in expected_rules.items():
        for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
            predictor = builder(evaluate_baselines.BASELINE_CONFIGS[name])
            assert predictor.blend_rules == blend_rules
            assert predictor.base_predictor.beta_by_horizon == {1: 0.05, 3: 0.08}
            assert predictor.base_predictor.apply_rules == [
                {"sub_category": "DPPUO5X2", "horizon": 1},
                {"sub_category": "DPPUO5X2", "horizon": 3},
                {"sub_category": "PZ9S1Z4V", "horizon": 1},
                {"sub_category": "PZ9S1Z4V", "horizon": 3},
                {"sub_category": "PHHHVYZI", "horizon": 1},
                {"sub_category": "PHHHVYZI", "horizon": 3},
            ]


def test_pentaslice_x9_h25_core_warm_probe_variants_are_exposed_and_preserve_rules() -> None:
    expected_rules = {
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_x9_h25_core_a003_v1": [
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.03},
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.03},
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.03},
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.03},
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_x9_h25_core_a005_v1": [
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.05},
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.05},
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.05},
            {"code": "X9BZ68VQ", "horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.05},
        ],
    }

    assert expected_rules.keys() <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected_rules.keys() <= predict_baseline.BASELINE_CONFIGS.keys()

    for name, blend_rules in expected_rules.items():
        for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
            predictor = builder(evaluate_baselines.BASELINE_CONFIGS[name])
            assert predictor.blend_rules == blend_rules
            assert predictor.base_predictor.beta_by_horizon == {1: 0.05, 3: 0.08}
            assert predictor.base_predictor.apply_rules == [
                {"sub_category": "DPPUO5X2", "horizon": 1},
                {"sub_category": "DPPUO5X2", "horizon": 3},
                {"sub_category": "PZ9S1Z4V", "horizon": 1},
                {"sub_category": "PZ9S1Z4V", "horizon": 3},
                {"sub_category": "PHHHVYZI", "horizon": 1},
                {"sub_category": "PHHHVYZI", "horizon": 3},
                {"sub_category": "NQ58FVQM", "horizon": 1},
                {"sub_category": "NQ58FVQM", "horizon": 3},
                {"sub_category": "V8BKY1IV", "horizon": 1},
                {"sub_category": "V8BKY1IV", "horizon": 3},
            ]


def test_pentaslice_osjl_h25_core_warm_probe_variants_are_exposed_and_preserve_rules() -> None:
    expected_rules = {
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_osjl_h25_core_a003_v1": [
            {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.03},
            {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.03},
            {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.03},
        ],
        "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_beta_hi_osjl_h25_core_a005_v1": [
            {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.05},
            {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.05},
            {"code": "OSJL3A7Y", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.05},
        ],
    }

    assert expected_rules.keys() <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected_rules.keys() <= predict_baseline.BASELINE_CONFIGS.keys()

    for name, blend_rules in expected_rules.items():
        for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
            predictor = builder(evaluate_baselines.BASELINE_CONFIGS[name])
            assert predictor.blend_rules == blend_rules
            assert predictor.base_predictor.beta_by_horizon == {1: 0.05, 3: 0.08}
            assert predictor.base_predictor.apply_rules == [
                {"sub_category": "DPPUO5X2", "horizon": 1},
                {"sub_category": "DPPUO5X2", "horizon": 3},
                {"sub_category": "PZ9S1Z4V", "horizon": 1},
                {"sub_category": "PZ9S1Z4V", "horizon": 3},
                {"sub_category": "PHHHVYZI", "horizon": 1},
                {"sub_category": "PHHHVYZI", "horizon": 3},
                {"sub_category": "NQ58FVQM", "horizon": 1},
                {"sub_category": "NQ58FVQM", "horizon": 3},
                {"sub_category": "V8BKY1IV", "horizon": 1},
                {"sub_category": "V8BKY1IV", "horizon": 3},
            ]


def test_residual_winner_and_conditional_longwatch_blend_are_exposed() -> None:
    expected_rules = [
        {"code": "OSJL3A7Y", "horizon": 10, "weight": 0.15},
        {"code": "OSJL3A7Y", "horizon": 25, "weight": 0.15},
        {"code": "X9BZ68VQ", "horizon": 10, "weight": 0.15},
        {"code": "X9BZ68VQ", "horizon": 25, "weight": 0.15},
    ]

    expected_names = {
        "ridge_oofnorm_pentaslice_h13_s008_q85_a50_unweighted_v1",
        "winner_longwatch_residual_raw_v1",
        "winner_plus_longwatch_expert_blend_v1",
    }

    assert expected_names <= evaluate_baselines.BASELINE_CONFIGS.keys()
    assert expected_names <= predict_baseline.BASELINE_CONFIGS.keys()

    for builder in (evaluate_baselines.build_predictor, predict_baseline.build_predictor):
        winner = builder(
            evaluate_baselines.BASELINE_CONFIGS[
                "ridge_oofnorm_pentaslice_h13_s008_q85_a50_unweighted_v1"
            ]
        )
        assert winner.horizons == [1, 3]
        assert winner.sub_categories == [
            "DPPUO5X2",
            "PZ9S1Z4V",
            "PHHHVYZI",
            "NQ58FVQM",
            "V8BKY1IV",
        ]
        assert winner.correction_scale == 0.08
        assert winner.correction_clip_quantile == 0.85
        assert winner.alpha == 50.0
        assert winner.banned_feature_columns == {"feature_al"}

        specialist = builder(
            evaluate_baselines.BASELINE_CONFIGS["winner_longwatch_residual_raw_v1"]
        )
        assert specialist.codes == ["OSJL3A7Y", "X9BZ68VQ"]
        assert specialist.horizons == [10, 25]
        assert specialist.sub_categories == [
            "DPPUO5X2",
            "PHHHVYZI",
            "PZ9S1Z4V",
            "NQ58FVQM",
        ]

        blend = builder(
            evaluate_baselines.BASELINE_CONFIGS["winner_plus_longwatch_expert_blend_v1"]
        )
        assert blend.blend_rules == expected_rules
        assert getattr(blend.default_predictor, "horizons", None) == [1, 3]
        assert getattr(blend.slice_predictor, "horizons", None) == [10, 25]


def test_residual_registry_required_columns_include_allowed_features() -> None:
    available_columns = [
        "id",
        "code",
        "sub_code",
        "sub_category",
        "horizon",
        "ts_index",
        "feature_a",
        "feature_al",
        "feature_b",
        "y_target",
        "weight",
    ]

    expected_context = [
        "id",
        "code",
        "sub_code",
        "sub_category",
        "horizon",
        "ts_index",
        "feature_a",
        "feature_b",
    ]
    expected_train_columns = [*expected_context, "y_target", "weight"]

    for module in (evaluate_baselines, predict_baseline):
        context_columns = module.resolve_required_columns(
            module.BASELINE_CONFIGS["ridge_oofnorm_pentaslice_h13_s008_q85_a50_unweighted_v1"],
            available_columns=available_columns,
        )
        train_columns = module.resolve_required_columns(
            module.BASELINE_CONFIGS["ridge_oofnorm_pentaslice_h13_s008_q85_a50_unweighted_v1"],
            available_columns=available_columns,
            include_target=True,
            include_weight=True,
        )

        assert context_columns == expected_context
        assert train_columns == expected_train_columns


def test_non_residual_registry_required_columns_remain_context_only() -> None:
    available_columns = [
        "id",
        "code",
        "sub_code",
        "sub_category",
        "horizon",
        "ts_index",
        "feature_a",
        "feature_b",
        "y_target",
        "weight",
    ]
    expected_context = [
        "id",
        "code",
        "sub_code",
        "sub_category",
        "horizon",
        "ts_index",
    ]

    for module in (evaluate_baselines, predict_baseline):
        context_columns = module.resolve_required_columns(
            module.BASELINE_CONFIGS[
                "smoothed_stable_horizon_blend_last_target_recent_delta_k7_h13_pentaslice_h3_beta_up_v1"
            ],
            available_columns=available_columns,
        )

        assert context_columns == expected_context
