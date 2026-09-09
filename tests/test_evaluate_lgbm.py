from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_lgbm import (
    INCUMBENT_CONFIG_NAME,
    build_precomputed_oof_cache_path,
    build_residual_prediction_mask,
    build_lgbm_fit_kwargs,
    build_lgbm_params,
    build_model_name,
    materialize_prepared_fold_features,
    load_fold_frames,
    load_precomputed_oof_cache,
    normalize_residual_horizons,
    resolve_precomputed_train_oof,
    run_fold,
    write_scored_fold_frame,
    write_precomputed_oof_cache,
)
from ts_forecasting.lgbm_models import build_seen_sub_code_pairs, prepare_feature_frame
from ts_forecasting.validation import ExpandingWindowFold


def test_lgbm_residual_path_uses_promoted_ridge_incumbent() -> None:
    assert INCUMBENT_CONFIG_NAME == "ridge_oofnorm_pentaslice_h13_s008_q85_a50_unweighted_v1"


def test_build_seen_sub_code_pairs_only_uses_passed_rows() -> None:
    frame = pd.DataFrame(
        {
            "code": ["A", "A", "B"],
            "sub_code": ["S1", "S2", "S3"],
            "ts_index": [1, 2, 3],
        }
    )

    seen_pairs = build_seen_sub_code_pairs(frame.iloc[:2].copy())

    assert seen_pairs == {("A", "S1"), ("A", "S2")}


def test_build_lgbm_params_applies_cli_overrides() -> None:
    class Args:
        num_leaves = 31
        max_depth = 8
        learning_rate = 0.03
        min_child_samples = 400
        min_sum_hessian_in_leaf = 100.0
        feature_fraction = 0.7
        bagging_fraction = None
        bagging_freq = 2
        min_split_gain = 0.05
        reg_alpha = 0.5
        reg_lambda = 2.0
        num_threads = 48
        device_type = "cpu"
        seed = 123

    params = build_lgbm_params(Args())

    assert params["num_leaves"] == 31
    assert params["max_depth"] == 8
    assert params["learning_rate"] == 0.03
    assert params["min_child_samples"] == 400
    assert params["min_sum_hessian_in_leaf"] == 100.0
    assert params["feature_fraction"] == 0.7
    assert params["bagging_fraction"] == 0.8
    assert params["bagging_freq"] == 2
    assert params["min_split_gain"] == 0.05
    assert params["reg_alpha"] == 0.5
    assert params["reg_lambda"] == 2.0
    assert params["num_threads"] == 48
    assert params["device_type"] == "cpu"
    assert params["seed"] == 123


def test_build_lgbm_fit_kwargs_use_expected_defaults() -> None:
    class Args:
        n_estimators = None
        early_stopping_rounds = None

    fit_kwargs = build_lgbm_fit_kwargs(Args())

    assert fit_kwargs == {
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
    }


def test_build_lgbm_fit_kwargs_apply_cli_overrides() -> None:
    class Args:
        n_estimators = 6000
        early_stopping_rounds = 150

    fit_kwargs = build_lgbm_fit_kwargs(Args())

    assert fit_kwargs == {
        "n_estimators": 6000,
        "early_stopping_rounds": 150,
    }


def test_build_model_name_appends_suffix() -> None:
    assert build_model_name(mode="residual", weighted=True, name_suffix="tuned") == (
        "lgbm_residual_weighted_v2_tuned"
    )


def test_normalize_residual_horizons_uses_defaults_when_unset() -> None:
    assert normalize_residual_horizons(None) == [1, 3]


def test_normalize_residual_horizons_sorts_and_deduplicates() -> None:
    assert normalize_residual_horizons([3, 1, 3]) == [1, 3]


def test_build_residual_prediction_mask_can_skip_cold_rows() -> None:
    frame = pd.DataFrame(
        {
            "code": ["A", "A", "A"],
            "sub_code": ["S1", "S2", "S1"],
            "horizon": [1, 1, 25],
        }
    )

    mask = build_residual_prediction_mask(
        frame,
        residual_horizons=[1, 25],
        apply_residual_only_warm=True,
        seen_pairs={("A", "S1")},
    )

    assert mask.tolist() == [True, False, True]


def test_build_residual_prediction_mask_can_gate_specific_horizons_only() -> None:
    frame = pd.DataFrame(
        {
            "code": ["A", "A", "A", "A"],
            "sub_code": ["S1", "S2", "S2", "S1"],
            "horizon": [1, 1, 25, 25],
        }
    )

    mask = build_residual_prediction_mask(
        frame,
        residual_horizons=[1, 25],
        apply_residual_only_warm=False,
        seen_pairs={("A", "S1")},
        residual_warm_only_horizons=[1],
    )

    assert mask.tolist() == [True, False, True, True]


def test_run_fold_builds_seen_pairs_for_selected_warm_only_horizons(
    monkeypatch,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 25],
            "ts_index": [1, 2],
            "y_target": [1.0, 2.0],
            "weight": [1.0, 1.0],
            "feature_a": [0.1, 0.2],
        }
    )
    valid_frame = pd.DataFrame(
        {
            "id": ["c", "d"],
            "code": ["C1", "C1"],
            "sub_code": ["S2", "S2"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 25],
            "ts_index": [3, 4],
            "y_target": [3.0, 4.0],
            "weight": [1.0, 1.0],
            "feature_a": [0.3, 0.4],
        }
    )

    monkeypatch.setattr(
        "scripts.evaluate_lgbm.build_oof_incumbent_predictions",
        lambda frame, config: pd.Series([0.4, 0.5], index=frame.index, dtype=float),
    )
    monkeypatch.setattr(
        "scripts.evaluate_lgbm._fit_predict_incumbent",
        lambda train, pred, config: np.array([10.0, 20.0], dtype=float),
    )

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(
            self,
            train_frame: pd.DataFrame,
            *,
            eval_frame: pd.DataFrame | None = None,
        ) -> "FakeForecaster":
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.array([1.25] * len(frame), dtype=float)

    monkeypatch.setattr("scripts.evaluate_lgbm.LGBMForecaster", FakeForecaster)

    scored = run_fold(
        train_frame=train_frame,
        valid_frame=valid_frame,
        mode="residual",
        residual_horizons=[1, 25],
        incumbent_config={"name": "dummy"},
        lgbm_params={"seed": 42},
        lgbm_fit_kwargs={"n_estimators": 100, "early_stopping_rounds": 10},
        use_sample_weight=True,
        apply_residual_only_warm=False,
        residual_warm_only_horizons=[1],
    )

    assert scored["prediction"].tolist() == [10.0, 21.25]


def test_run_fold_applies_configured_transformer_before_lgbm_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["SC", "SC"],
            "horizon": [25, 25],
            "ts_index": [1, 2],
            "y_target": [1.0, 2.0],
            "weight": [1.0, 1.0],
            "feature_a": [0.1, 0.2],
        }
    )
    valid_frame = pd.DataFrame(
        {
            "id": ["c", "d"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["SC", "SC"],
            "horizon": [25, 25],
            "ts_index": [3, 4],
            "y_target": [3.0, 4.0],
            "weight": [1.0, 1.0],
            "feature_a": [0.3, 0.4],
        }
    )

    class AddHistoryTransformer:
        supports_full_frame_prediction = False

        def fit(self, train_frame: pd.DataFrame) -> "AddHistoryTransformer":
            return self

        def transform_batch(self, batch_frame: pd.DataFrame) -> pd.DataFrame:
            transformed = batch_frame.copy()
            transformed["feature_a_history_last"] = transformed["feature_a"] - 1.0
            return transformed

    monkeypatch.setattr(
        "scripts.evaluate_lgbm.build_oof_incumbent_predictions",
        lambda frame, config: pd.Series([0.4, 0.5], index=frame.index, dtype=float),
    )
    monkeypatch.setattr(
        "scripts.evaluate_lgbm._fit_predict_incumbent",
        lambda train, pred, config: np.array([10.0, 20.0], dtype=float),
    )
    monkeypatch.setattr(
        "scripts.evaluate_lgbm.build_transformer",
        lambda config: AddHistoryTransformer(),
    )

    captured: dict[str, object] = {}

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(
            self,
            train_frame: pd.DataFrame,
            *,
            eval_frame: pd.DataFrame | None = None,
        ) -> "FakeForecaster":
            captured["fit_columns"] = list(train_frame.columns)
            captured["eval_columns"] = list(eval_frame.columns) if eval_frame is not None else None
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            captured["predict_columns"] = list(frame.columns)
            return np.array([1.25] * len(frame), dtype=float)

    monkeypatch.setattr("scripts.evaluate_lgbm.LGBMForecaster", FakeForecaster)

    scored = run_fold(
        train_frame=train_frame,
        valid_frame=valid_frame,
        mode="residual",
        residual_horizons=[25],
        incumbent_config={"transformer": {"type": "causal_history"}},
        lgbm_params={"seed": 42},
        lgbm_fit_kwargs={"n_estimators": 100, "early_stopping_rounds": 10},
        use_sample_weight=True,
        apply_residual_only_warm=False,
        residual_warm_only_horizons=None,
    )

    assert "feature_a_history_last" in captured["fit_columns"]
    assert "feature_a_history_last" in captured["eval_columns"]
    assert "feature_a_history_last" in captured["predict_columns"]
    assert scored["prediction"].tolist() == [11.25, 21.25]


def test_run_fold_uses_precomputed_train_oof_when_provided(monkeypatch) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 1],
            "ts_index": [1, 2],
            "y_target": [5.0, 7.0],
            "weight": [1.0, 1.0],
            "feature_a": [0.1, 0.2],
        }
    )
    valid_frame = pd.DataFrame(
        {
            "id": ["c", "d"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 1],
            "ts_index": [3, 4],
            "y_target": [11.0, 13.0],
            "weight": [1.0, 1.0],
            "feature_a": [0.3, 0.4],
        },
        index=[10, 11],
    )

    monkeypatch.setattr(
        "scripts.evaluate_lgbm.build_oof_incumbent_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_oof_incumbent_predictions should not run")
        ),
    )
    monkeypatch.setattr(
        "scripts.evaluate_lgbm._fit_predict_incumbent",
        lambda train, pred, config: np.array([10.0, 20.0], dtype=float),
    )

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            self.mean_target = 0.0

        def fit(
            self,
            train_frame: pd.DataFrame,
            *,
            eval_frame: pd.DataFrame | None = None,
        ) -> "FakeForecaster":
            self.mean_target = float(train_frame["y_target"].mean())
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), self.mean_target, dtype=float)

    monkeypatch.setattr("scripts.evaluate_lgbm.LGBMForecaster", FakeForecaster)

    scored = run_fold(
        train_frame=train_frame,
        valid_frame=valid_frame,
        mode="residual",
        residual_horizons=[1],
        incumbent_config={"name": "dummy"},
        lgbm_params={"seed": 42},
        lgbm_fit_kwargs={"n_estimators": 100, "early_stopping_rounds": 10},
        use_sample_weight=True,
        apply_residual_only_warm=False,
        residual_warm_only_horizons=None,
        precomputed_train_oof=pd.Series([1.0, 1.0], index=train_frame.index),
    )

    assert scored["prediction"].tolist() == [15.0, 25.0]


def test_run_fold_uses_precomputed_valid_incumbent_when_available(
    monkeypatch,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 1],
            "ts_index": [1, 2],
            "y_target": [5.0, 7.0],
            "weight": [1.0, 1.0],
            "feature_a": [0.1, 0.2],
        }
    )
    valid_frame = pd.DataFrame(
        {
            "id": ["c", "d"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 1],
            "ts_index": [3, 4],
            "y_target": [11.0, 13.0],
            "weight": [1.0, 1.0],
            "feature_a": [0.3, 0.4],
        },
        index=[2, 3],
    )

    monkeypatch.setattr(
        "scripts.evaluate_lgbm.build_oof_incumbent_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_oof_incumbent_predictions should not run")
        ),
    )
    monkeypatch.setattr(
        "scripts.evaluate_lgbm._fit_predict_incumbent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("_fit_predict_incumbent should not run when validation OOF is available")
        ),
    )

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            self.mean_target = 0.0

        def fit(
            self,
            train_frame: pd.DataFrame,
            *,
            eval_frame: pd.DataFrame | None = None,
        ) -> "FakeForecaster":
            self.mean_target = float(train_frame["y_target"].mean())
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), self.mean_target, dtype=float)

    monkeypatch.setattr("scripts.evaluate_lgbm.LGBMForecaster", FakeForecaster)

    scored = run_fold(
        train_frame=train_frame,
        valid_frame=valid_frame,
        mode="residual",
        residual_horizons=[1],
        incumbent_config={"name": "dummy"},
        lgbm_params={"seed": 42},
        lgbm_fit_kwargs={"n_estimators": 100, "early_stopping_rounds": 10},
        use_sample_weight=True,
        apply_residual_only_warm=False,
        residual_warm_only_horizons=None,
        precomputed_train_oof=pd.Series([1.0, 1.0, 10.0, 20.0], index=[0, 1, 2, 3]),
    )

    assert scored["prediction"].tolist() == [15.0, 25.0]


def test_run_fold_uses_prepared_feature_slices_when_available(monkeypatch) -> None:
    full_train_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "code": ["C1", "C1", "C1", "C1"],
            "sub_code": ["S1", "S1", "S1", "S1"],
            "sub_category": ["SC", "SC", "SC", "SC"],
            "horizon": [1, 1, 1, 1],
            "ts_index": [1, 2, 3, 4],
            "y_target": [5.0, 7.0, 11.0, 13.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
        }
    )
    train_frame = full_train_frame.iloc[:2].copy()
    valid_frame = full_train_frame.iloc[2:].copy()
    prepared_features = prepare_feature_frame(full_train_frame)

    monkeypatch.setattr(
        "ts_forecasting.lgbm_models.build_feature_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_feature_frame should not run for prepared slices")
        ),
    )

    captured: dict[str, pd.DataFrame | None] = {}

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(
            self,
            train_frame: pd.DataFrame,
            *,
            eval_frame: pd.DataFrame | None = None,
            prepared_train_frame: pd.DataFrame | None = None,
            prepared_eval_frame: pd.DataFrame | None = None,
        ) -> "FakeForecaster":
            captured["prepared_train_frame"] = prepared_train_frame
            captured["prepared_eval_frame"] = prepared_eval_frame
            self.mean_target = float(train_frame["y_target"].mean())
            return self

        def predict(
            self,
            frame: pd.DataFrame,
            *,
            prepared_frame: pd.DataFrame | None = None,
        ) -> np.ndarray:
            captured["prepared_frame"] = prepared_frame
            return np.full(len(frame), self.mean_target, dtype=float)

    monkeypatch.setattr("scripts.evaluate_lgbm.LGBMForecaster", FakeForecaster)

    scored = run_fold(
        train_frame=train_frame,
        valid_frame=valid_frame,
        mode="direct",
        residual_horizons=[1],
        incumbent_config=None,
        lgbm_params={"seed": 42},
        lgbm_fit_kwargs={"n_estimators": 100, "early_stopping_rounds": 10},
        use_sample_weight=True,
        apply_residual_only_warm=False,
        residual_warm_only_horizons=None,
        prepared_train_features=prepared_features,
    )

    expected_train_features, expected_valid_features = (
        materialize_prepared_fold_features(
            prepared_features,
            train_index=train_frame.index,
            valid_index=valid_frame.index,
        )
    )

    assert captured["prepared_train_frame"].equals(expected_train_features)
    assert captured["prepared_eval_frame"].equals(expected_valid_features)
    assert captured["prepared_frame"].equals(expected_valid_features)
    assert scored["prediction"].tolist() == [6.0, 6.0]


def test_write_scored_fold_frame_writes_parquet(tmp_path: Path) -> None:
    scored = pd.DataFrame(
        {
            "id": ["a", "b"],
            "ts_index": [1, 2],
            "prediction": [0.1, 0.2],
            "is_cold_start": [True, False],
        }
    )

    output_path = write_scored_fold_frame(
        scored,
        output_dir=tmp_path,
        model_name="demo_model",
        fold_name="fold_01_train_1_valid_2_3",
    )

    assert output_path == tmp_path / "demo_model_fold_01_train_1_valid_2_3_scored.parquet"
    restored = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(restored, scored)


def test_run_fold_layers_incumbent_pred_on_prepared_features(monkeypatch) -> None:
    full_train_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "code": ["C1", "C1", "C1", "C1"],
            "sub_code": ["S1", "S1", "S1", "S1"],
            "sub_category": ["SC", "SC", "SC", "SC"],
            "horizon": [1, 1, 1, 1],
            "ts_index": [1, 2, 3, 4],
            "y_target": [5.0, 7.0, 11.0, 13.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
        }
    )
    train_frame = full_train_frame.iloc[:2].copy()
    valid_frame = full_train_frame.iloc[2:].copy()
    prepared_features = prepare_feature_frame(full_train_frame)

    monkeypatch.setattr(
        "scripts.evaluate_lgbm.build_oof_incumbent_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_oof_incumbent_predictions should not run")
        ),
    )
    monkeypatch.setattr(
        "scripts.evaluate_lgbm._fit_predict_incumbent",
        lambda train, pred, config: np.array([30.0, 40.0], dtype=float),
    )
    monkeypatch.setattr(
        "ts_forecasting.lgbm_models.build_feature_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_feature_frame should not run for prepared slices")
        ),
    )

    captured: dict[str, pd.DataFrame | None] = {}

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(
            self,
            train_frame: pd.DataFrame,
            *,
            eval_frame: pd.DataFrame | None = None,
            prepared_train_frame: pd.DataFrame | None = None,
            prepared_eval_frame: pd.DataFrame | None = None,
        ) -> "FakeForecaster":
            captured["prepared_train_frame"] = prepared_train_frame
            captured["prepared_eval_frame"] = prepared_eval_frame
            return self

        def predict(
            self,
            frame: pd.DataFrame,
            *,
            prepared_frame: pd.DataFrame | None = None,
        ) -> np.ndarray:
            captured["prepared_frame"] = prepared_frame
            return np.full(len(frame), 1.25, dtype=float)

    monkeypatch.setattr("scripts.evaluate_lgbm.LGBMForecaster", FakeForecaster)

    scored = run_fold(
        train_frame=train_frame,
        valid_frame=valid_frame,
        mode="residual",
        residual_horizons=[1],
        incumbent_config={"name": "dummy"},
        lgbm_params={"seed": 42},
        lgbm_fit_kwargs={"n_estimators": 100, "early_stopping_rounds": 10},
        use_sample_weight=True,
        apply_residual_only_warm=False,
        residual_warm_only_horizons=None,
        precomputed_train_oof=pd.Series([10.0, 20.0], index=train_frame.index),
        prepared_train_features=prepared_features,
    )

    expected_train_features, expected_valid_features = (
        materialize_prepared_fold_features(
            prepared_features,
            train_index=train_frame.index,
            valid_index=valid_frame.index,
        )
    )
    expected_train_features = expected_train_features.assign(
        incumbent_pred=[10.0, 20.0],
    )
    expected_valid_features = expected_valid_features.assign(
        incumbent_pred=[30.0, 40.0],
    )

    assert captured["prepared_train_frame"].equals(expected_train_features)
    assert captured["prepared_eval_frame"].equals(expected_valid_features)
    assert captured["prepared_frame"].equals(expected_valid_features)
    assert scored["prediction"].tolist() == [31.25, 41.25]


def test_load_fold_frames_uses_preloaded_full_train_frame(monkeypatch) -> None:
    full_train_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "code": ["C1", "C1", "C1", "C1"],
            "sub_code": ["S1", "S1", "S1", "S1"],
            "sub_category": ["SC", "SC", "SC", "SC"],
            "horizon": [1, 1, 1, 1],
            "ts_index": [1, 2, 3, 4],
            "y_target": [5.0, 6.0, 7.0, 8.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
        }
    )
    fold = ExpandingWindowFold(
        name="fold_01_train_2_valid_3_4",
        train_end_ts=2,
        validation_start_ts=3,
        validation_end_ts=4,
    )

    monkeypatch.setattr(
        "scripts.evaluate_lgbm.read_parquet_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("read_parquet_frame should not run")
        ),
    )

    train_frame, valid_frame = load_fold_frames(
        "unused.parquet",
        fold,
        load_cols=[
            "id",
            "code",
            "sub_code",
            "sub_category",
            "horizon",
            "ts_index",
            "y_target",
            "weight",
            "feature_a",
        ],
        preloaded_train_frame=full_train_frame,
    )

    assert train_frame["id"].tolist() == ["a", "b"]
    assert valid_frame["id"].tolist() == ["c", "d"]


def test_precomputed_oof_cache_round_trip(tmp_path) -> None:
    cache_path = build_precomputed_oof_cache_path(
        tmp_path,
        incumbent_name="ridge_demo",
        train_path="data/train.parquet",
    )
    series = pd.Series([1.5, np.nan, 3.0], index=[10, 11, 12], dtype=float)

    write_precomputed_oof_cache(cache_path, series)
    loaded = load_precomputed_oof_cache(cache_path)

    assert loaded.index.tolist() == [10, 11, 12]
    assert np.isnan(loaded.loc[11])
    assert loaded.loc[10] == 1.5
    assert loaded.loc[12] == 3.0


def test_resolve_precomputed_train_oof_loads_existing_cache(
    tmp_path,
    monkeypatch,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 1],
            "ts_index": [1, 2],
            "y_target": [5.0, 7.0],
            "weight": [1.0, 1.0],
            "feature_a": [0.1, 0.2],
        }
    )
    cached_series = pd.Series([1.0, 2.0], index=train_frame.index, dtype=float)
    cache_path = build_precomputed_oof_cache_path(
        tmp_path,
        incumbent_name="ridge_demo",
        train_path="data/train.parquet",
    )
    write_precomputed_oof_cache(cache_path, cached_series)

    monkeypatch.setattr(
        "scripts.evaluate_lgbm.build_oof_incumbent_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_oof_incumbent_predictions should not run when cache exists")
        ),
    )

    loaded = resolve_precomputed_train_oof(
        cache_dir=tmp_path,
        train_path="data/train.parquet",
        incumbent_name="ridge_demo",
        train_frame=train_frame,
        incumbent_config={"name": "dummy"},
    )

    assert loaded.equals(cached_series)
