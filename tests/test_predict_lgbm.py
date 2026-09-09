from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import predict_lgbm


def test_resolve_full_train_n_estimators_prefers_explicit_value() -> None:
    assert predict_lgbm.resolve_full_train_n_estimators(512, 718) == 512


def test_resolve_full_train_n_estimators_uses_calibrated_value() -> None:
    assert predict_lgbm.resolve_full_train_n_estimators(None, 718) == 718


def test_resolve_full_train_n_estimators_requires_value() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        predict_lgbm.resolve_full_train_n_estimators(None, None)


def test_predict_residual_lgbm_uses_only_incumbent_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "code": ["C1", "C1", "C1"],
            "sub_code": ["S1", "S1", "S1"],
            "sub_category": ["SC", "SC", "SC"],
            "horizon": [1, 3, 25],
            "ts_index": [1, 2, 3],
            "y_target": [1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3],
        }
    )
    predict_frame = pd.DataFrame(
        {
            "id": ["d", "e"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S2"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 25],
            "ts_index": [4, 5],
            "feature_a": [0.4, 0.5],
        }
    )

    monkeypatch.setattr(
        predict_lgbm,
        "build_oof_incumbent_predictions",
        lambda frame, config: pd.Series([0.5, 0.6, 0.7], index=frame.index, dtype=float),
    )
    monkeypatch.setattr(
        predict_lgbm,
        "_fit_predict_incumbent",
        lambda train, pred, config: np.array([10.0, 20.0], dtype=float),
    )

    captured: dict[str, object] = {}

    class FakeForecaster:
        def __init__(
            self,
            *,
            params: dict[str, object],
            extra_numeric_columns: list[str] | None,
            use_sample_weight: bool,
            n_estimators: int,
            early_stopping_rounds: int,
        ) -> None:
            captured["extra_numeric_columns"] = extra_numeric_columns
            captured["n_estimators"] = n_estimators

        def fit(self, train_frame: pd.DataFrame, *, eval_frame: pd.DataFrame | None = None) -> "FakeForecaster":
            captured["fit_eval_frame"] = eval_frame
            captured["fit_columns"] = list(train_frame.columns)
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            captured["predict_columns"] = list(frame.columns)
            return np.array([1.25] * len(frame), dtype=float)

    monkeypatch.setattr(predict_lgbm, "LGBMForecaster", FakeForecaster)

    predictions = predict_lgbm.predict_residual_lgbm(
        train_frame=train_frame,
        predict_frame=predict_frame,
        incumbent_config={"name": "dummy"},
        residual_horizons=[1, 25],
        lgbm_params={"seed": 42},
        n_estimators=718,
        early_stopping_rounds=50,
        use_sample_weight=True,
    )

    assert captured["extra_numeric_columns"] == ["incumbent_pred"]
    assert captured["n_estimators"] == 718
    assert captured["fit_eval_frame"] is None
    assert "sub_code_is_seen" not in captured["fit_columns"]
    assert "sub_code_is_seen" not in captured["predict_columns"]
    assert predictions["prediction"].tolist() == [11.25, 21.25]


def test_predict_residual_lgbm_applies_configured_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "code": ["C1", "C1", "C1"],
            "sub_code": ["S1", "S1", "S1"],
            "sub_category": ["SC", "SC", "SC"],
            "horizon": [25, 25, 25],
            "ts_index": [1, 2, 3],
            "y_target": [1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3],
        }
    )
    predict_frame = pd.DataFrame(
        {
            "id": ["d", "e"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S1"],
            "sub_category": ["SC", "SC"],
            "horizon": [25, 25],
            "ts_index": [4, 5],
            "feature_a": [0.4, 0.5],
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
        predict_lgbm,
        "build_oof_incumbent_predictions",
        lambda frame, config: pd.Series([0.5, 0.6, 0.7], index=frame.index, dtype=float),
    )
    monkeypatch.setattr(
        predict_lgbm,
        "_fit_predict_incumbent",
        lambda train, pred, config: np.array([10.0, 20.0], dtype=float),
    )
    monkeypatch.setattr(
        predict_lgbm,
        "build_transformer",
        lambda config: AddHistoryTransformer(),
    )

    captured: dict[str, object] = {}

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, train_frame: pd.DataFrame, *, eval_frame: pd.DataFrame | None = None) -> "FakeForecaster":
            captured["fit_columns"] = list(train_frame.columns)
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            captured["predict_columns"] = list(frame.columns)
            return np.array([1.25] * len(frame), dtype=float)

    monkeypatch.setattr(predict_lgbm, "LGBMForecaster", FakeForecaster)

    predictions = predict_lgbm.predict_residual_lgbm(
        train_frame=train_frame,
        predict_frame=predict_frame,
        incumbent_config={"transformer": {"type": "causal_history"}},
        residual_horizons=[25],
        lgbm_params={"seed": 42},
        n_estimators=718,
        early_stopping_rounds=50,
        use_sample_weight=True,
    )

    assert "feature_a_history_last" in captured["fit_columns"]
    assert "feature_a_history_last" in captured["predict_columns"]
    assert predictions["prediction"].tolist() == [11.25, 21.25]


def test_predict_residual_lgbm_can_skip_cold_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "code": ["C1", "C1", "C1"],
            "sub_code": ["S1", "S1", "S1"],
            "sub_category": ["SC", "SC", "SC"],
            "horizon": [1, 3, 25],
            "ts_index": [1, 2, 3],
            "y_target": [1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3],
        }
    )
    predict_frame = pd.DataFrame(
        {
            "id": ["d", "e"],
            "code": ["C1", "C1"],
            "sub_code": ["S1", "S2"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 25],
            "ts_index": [4, 5],
            "feature_a": [0.4, 0.5],
        }
    )

    monkeypatch.setattr(
        predict_lgbm,
        "build_oof_incumbent_predictions",
        lambda frame, config: pd.Series([0.5, 0.6, 0.7], index=frame.index, dtype=float),
    )
    monkeypatch.setattr(
        predict_lgbm,
        "_fit_predict_incumbent",
        lambda train, pred, config: np.array([10.0, 20.0], dtype=float),
    )

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, train_frame: pd.DataFrame, *, eval_frame: pd.DataFrame | None = None) -> "FakeForecaster":
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.array([1.25] * len(frame), dtype=float)

    monkeypatch.setattr(predict_lgbm, "LGBMForecaster", FakeForecaster)

    predictions = predict_lgbm.predict_residual_lgbm(
        train_frame=train_frame,
        predict_frame=predict_frame,
        incumbent_config={"name": "dummy"},
        residual_horizons=[1, 25],
        lgbm_params={"seed": 42},
        n_estimators=718,
        early_stopping_rounds=50,
        use_sample_weight=True,
        apply_residual_only_warm=True,
    )

    assert predictions["prediction"].tolist() == [11.25, 20.0]


def test_predict_residual_lgbm_can_skip_cold_rows_for_selected_horizons_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "code": ["C1", "C1", "C1"],
            "sub_code": ["S1", "S1", "S1"],
            "sub_category": ["SC", "SC", "SC"],
            "horizon": [1, 3, 25],
            "ts_index": [1, 2, 3],
            "y_target": [1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3],
        }
    )
    predict_frame = pd.DataFrame(
        {
            "id": ["d", "e"],
            "code": ["C1", "C1"],
            "sub_code": ["S2", "S2"],
            "sub_category": ["SC", "SC"],
            "horizon": [1, 25],
            "ts_index": [4, 5],
            "feature_a": [0.4, 0.5],
        }
    )

    monkeypatch.setattr(
        predict_lgbm,
        "build_oof_incumbent_predictions",
        lambda frame, config: pd.Series([0.5, 0.6, 0.7], index=frame.index, dtype=float),
    )
    monkeypatch.setattr(
        predict_lgbm,
        "_fit_predict_incumbent",
        lambda train, pred, config: np.array([10.0, 20.0], dtype=float),
    )

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, train_frame: pd.DataFrame, *, eval_frame: pd.DataFrame | None = None) -> "FakeForecaster":
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.array([1.25] * len(frame), dtype=float)

    monkeypatch.setattr(predict_lgbm, "LGBMForecaster", FakeForecaster)

    predictions = predict_lgbm.predict_residual_lgbm(
        train_frame=train_frame,
        predict_frame=predict_frame,
        incumbent_config={"name": "dummy"},
        residual_horizons=[1, 25],
        lgbm_params={"seed": 42},
        n_estimators=718,
        early_stopping_rounds=50,
        use_sample_weight=True,
        apply_residual_only_warm=False,
        residual_warm_only_horizons=[1],
    )

    assert predictions["prediction"].tolist() == [10.0, 21.25]


def test_predict_lgbm_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(Path("scripts/predict_lgbm.py")), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate residual LightGBM predictions" in result.stdout


def test_calibration_uses_evaluator_oof_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "code": ["C1", "C1", "C1"],
            "sub_code": ["S1", "S1", "S1"],
            "sub_category": ["SC", "SC", "SC"],
            "horizon": [1, 1, 1],
            "ts_index": [1, 2, 3],
            "y_target": [1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3],
        }
    )

    captured: dict[str, object] = {}

    def fake_build_oof(frame: pd.DataFrame, config: dict[str, object], **kwargs: object) -> pd.Series:
        captured["oof_kwargs"] = kwargs
        return pd.Series([0.1, 0.2], index=frame.index[:2], dtype=float)

    monkeypatch.setattr(predict_lgbm, "build_oof_incumbent_predictions", fake_build_oof)
    monkeypatch.setattr(
        predict_lgbm,
        "_fit_predict_incumbent",
        lambda train, pred, config: np.array([0.3], dtype=float),
    )

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            self.model_ = type("FakeModel", (), {"best_iteration": 123})()

        def fit(self, train_frame: pd.DataFrame, *, eval_frame: pd.DataFrame | None = None) -> "FakeForecaster":
            return self

    monkeypatch.setattr(predict_lgbm, "LGBMForecaster", FakeForecaster)

    best_iteration, metadata = predict_lgbm.calibrate_residual_lgbm_n_estimators(
        train_frame=train_frame,
        incumbent_config={"name": "dummy"},
        residual_horizons=[1],
        lgbm_params={"seed": 42},
        max_n_estimators=2000,
        early_stopping_rounds=50,
        calibration_holdout_steps=1,
        calibration_min_train_steps=1,
        use_sample_weight=True,
    )

    assert captured["oof_kwargs"] == {
        "oof_holdout_steps": 360,
        "oof_step_size": 360,
        "oof_min_train_steps": 1000,
    }
    assert best_iteration == 123
    assert metadata["best_iteration"] == 123.0
