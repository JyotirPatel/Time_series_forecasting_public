from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_lgbm_submission_proxy as proxy
from ts_forecasting.constants import (
    CODE_COLUMN,
    HORIZON_COLUMN,
    ID_COLUMN,
    SUB_CATEGORY_COLUMN,
    SUB_CODE_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
)


def _make_train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COLUMN: [f"row_{idx}" for idx in range(1, 7)],
            CODE_COLUMN: ["C1", "C1", "C2", "C2", "C3", "C3"],
            SUB_CODE_COLUMN: ["S1", "S1", "S2", "S2", "S3", "S4"],
            SUB_CATEGORY_COLUMN: ["SC"] * 6,
            HORIZON_COLUMN: [1, 3, 1, 3, 1, 3],
            TIME_COLUMN: [1, 2, 3, 4, 5, 6],
            TARGET_COLUMN: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            WEIGHT_COLUMN: [1.0] * 6,
            "feature_a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )


def test_split_tail_holdout_uses_last_steps() -> None:
    train_frame = _make_train_frame()

    prefix_frame, holdout_frame = proxy.split_tail_holdout(
        train_frame,
        pseudo_test_steps=2,
        min_train_steps=2,
    )

    assert prefix_frame[TIME_COLUMN].tolist() == [1, 2, 3, 4]
    assert holdout_frame[TIME_COLUMN].tolist() == [5, 6]


def test_split_tail_holdout_rejects_too_large_holdout() -> None:
    train_frame = _make_train_frame()

    with pytest.raises(ValueError, match="pseudo_test_steps"):
        proxy.split_tail_holdout(
            train_frame,
            pseudo_test_steps=5,
            min_train_steps=2,
        )


def test_bootstrap_public_scores_are_deterministic() -> None:
    scored_frame = pd.DataFrame(
        {
            HORIZON_COLUMN: [1, 1, 3, 3],
            TARGET_COLUMN: [10.0, 10.0, 10.0, 10.0],
            WEIGHT_COLUMN: [1.0, 1.0, 1.0, 1.0],
            "prediction": [10.0, 8.0, 7.0, 10.0],
        }
    )

    summary_a = proxy.bootstrap_public_scores(
        scored_frame,
        public_fraction=0.5,
        n_bootstraps=5,
        stratify_by_horizon=True,
        random_state=7,
    )
    summary_b = proxy.bootstrap_public_scores(
        scored_frame,
        public_fraction=0.5,
        n_bootstraps=5,
        stratify_by_horizon=True,
        random_state=7,
    )

    assert summary_a == summary_b
    assert summary_a["n_bootstraps"] == 5
    assert summary_a["sample_rows_min"] == 2
    assert summary_a["sample_rows_max"] == 2


def test_evaluate_tail_submission_proxy_uses_prefix_only_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = _make_train_frame()
    captured: dict[str, object] = {}

    def fake_calibrate(**kwargs: object) -> tuple[int, dict[str, float]]:
        calibration_train = kwargs["train_frame"]
        captured["calibration_train_max_ts"] = int(
            calibration_train[TIME_COLUMN].max()
        )
        captured["calibration_apply_only_warm"] = kwargs["apply_residual_only_warm"]
        captured["calibration_warm_only_horizons"] = kwargs["residual_warm_only_horizons"]
        return 123, {"best_iteration": 123.0}

    def fake_predict(**kwargs: object) -> pd.DataFrame:
        prefix_frame = kwargs["train_frame"]
        predict_frame = kwargs["predict_frame"]
        captured["predict_train_max_ts"] = int(prefix_frame[TIME_COLUMN].max())
        captured["predict_holdout_min_ts"] = int(predict_frame[TIME_COLUMN].min())
        captured["predict_apply_only_warm"] = kwargs["apply_residual_only_warm"]
        captured["predict_warm_only_horizons"] = kwargs["residual_warm_only_horizons"]
        return pd.DataFrame(
            {
                ID_COLUMN: predict_frame[ID_COLUMN].to_numpy(copy=False),
                "prediction": train_frame.loc[
                    train_frame[ID_COLUMN].isin(predict_frame[ID_COLUMN]),
                    TARGET_COLUMN,
                ].to_numpy(copy=False),
            }
        )

    monkeypatch.setattr(proxy.predict_lgbm, "calibrate_residual_lgbm_n_estimators", fake_calibrate)
    monkeypatch.setattr(proxy.predict_lgbm, "predict_residual_lgbm", fake_predict)

    summary = proxy.evaluate_tail_submission_proxy(
        train_frame=train_frame,
        incumbent_config={"name": "dummy"},
        residual_horizons=[1, 3],
        lgbm_params={"seed": 42},
        pseudo_test_steps=2,
        min_train_steps=2,
        early_stopping_rounds=50,
        calibration_holdout_steps=1,
        calibration_min_train_steps=2,
        calibration_max_n_estimators=2000,
        n_estimators=None,
        use_sample_weight=True,
        apply_residual_only_warm=False,
        residual_warm_only_horizons=[1, 3],
        public_fraction=0.5,
        n_bootstraps=3,
        random_state=11,
    )

    assert captured["calibration_train_max_ts"] == 4
    assert captured["calibration_apply_only_warm"] is False
    assert captured["calibration_warm_only_horizons"] == [1, 3]
    assert captured["predict_train_max_ts"] == 4
    assert captured["predict_holdout_min_ts"] == 5
    assert captured["predict_apply_only_warm"] is False
    assert captured["predict_warm_only_horizons"] == [1, 3]
    assert summary["holdout"]["overall_score"] == 1.0
    assert summary["coverage"]["code_sub_code"]["unseen_validation_rows"] == 2
    assert summary["random_public_bootstrap"]["mean_score"] == 1.0
    assert summary["stratified_public_bootstrap"]["mean_score"] == 1.0
    assert summary["full_train_n_estimators"] == 123


def test_evaluate_tail_submission_proxy_writes_holdout_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_frame = _make_train_frame()
    components_output = tmp_path / "holdout_components.parquet"

    def fake_calibrate(**kwargs: object) -> tuple[int, dict[str, float]]:
        return 123, {"best_iteration": 123.0}

    def fake_predict(**kwargs: object) -> pd.DataFrame:
        predict_frame = kwargs["predict_frame"]
        return pd.DataFrame(
            {
                ID_COLUMN: predict_frame[ID_COLUMN].to_numpy(copy=False),
                "prediction": train_frame.loc[
                    train_frame[ID_COLUMN].isin(predict_frame[ID_COLUMN]),
                    TARGET_COLUMN,
                ].to_numpy(copy=False),
            }
        )

    monkeypatch.setattr(proxy.predict_lgbm, "calibrate_residual_lgbm_n_estimators", fake_calibrate)
    monkeypatch.setattr(proxy.predict_lgbm, "predict_residual_lgbm", fake_predict)

    summary = proxy.evaluate_tail_submission_proxy(
        train_frame=train_frame,
        incumbent_config={"name": "dummy"},
        residual_horizons=[1, 3],
        lgbm_params={"seed": 42},
        pseudo_test_steps=2,
        min_train_steps=2,
        early_stopping_rounds=50,
        calibration_holdout_steps=1,
        calibration_min_train_steps=2,
        calibration_max_n_estimators=2000,
        n_estimators=None,
        use_sample_weight=True,
        apply_residual_only_warm=False,
        residual_warm_only_horizons=[1, 3],
        public_fraction=0.5,
        n_bootstraps=3,
        random_state=11,
        components_output=components_output,
    )

    assert summary["components_output"] == str(components_output)
    exported = pd.read_parquet(components_output)
    assert exported.columns.tolist() == [
        ID_COLUMN,
        CODE_COLUMN,
        SUB_CODE_COLUMN,
        SUB_CATEGORY_COLUMN,
        HORIZON_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
        WEIGHT_COLUMN,
        "prediction",
        "is_cold_start",
    ]
    assert exported[ID_COLUMN].tolist() == ["row_5", "row_6"]
    assert exported["is_cold_start"].tolist() == [True, True]
    assert exported["prediction"].tolist() == [5.0, 6.0]


def test_submission_proxy_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(Path("scripts/evaluate_lgbm_submission_proxy.py")), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Evaluate the checked-in LGBM submission path" in result.stdout


def test_submission_proxy_script_accepts_runtime_override_flags() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/evaluate_lgbm_submission_proxy.py")),
            "--num-threads",
            "8",
            "--device-type",
            "gpu",
            "--seed",
            "7",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--num-threads" in result.stdout
    assert "--device-type" in result.stdout
    assert "--seed" in result.stdout
    assert "--components-output" in result.stdout
