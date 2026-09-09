#!/usr/bin/env python
"""Generate residual LightGBM predictions with a calibrated full-train refit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_lgbm import (
    INCUMBENT_CONFIG_NAME,
    _fit_predict_incumbent,
    build_residual_prediction_mask,
    build_lgbm_params,
    build_oof_incumbent_predictions,
    normalize_residual_horizons,
    normalize_residual_warm_only_horizons,
)
from ts_forecasting.baseline_registry import BASELINE_CONFIGS, build_transformer
from ts_forecasting.constants import (
    HORIZON_COLUMN,
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
    detect_feature_columns,
)
from ts_forecasting.io import get_columns, read_parquet_frame
from ts_forecasting.lgbm_models import LGBMForecaster
from ts_forecasting.lgbm_models import build_seen_sub_code_pairs

OOF_HOLDOUT_STEPS = 360
OOF_STEP_SIZE = 360
OOF_MIN_TRAIN_STEPS = 1000


def resolve_full_train_n_estimators(
    explicit_n_estimators: int | None,
    calibrated_best_iteration: int | None,
) -> int:
    if explicit_n_estimators is not None:
        return int(explicit_n_estimators)
    if calibrated_best_iteration is not None:
        return int(calibrated_best_iteration)
    raise ValueError(
        "Full-train LGBM prediction requires n_estimators or a calibrated best iteration."
    )


def predict_residual_lgbm(
    *,
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    incumbent_config: dict[str, object],
    residual_horizons: list[int],
    lgbm_params: dict[str, object],
    n_estimators: int,
    early_stopping_rounds: int,
    use_sample_weight: bool,
    apply_residual_only_warm: bool = False,
    residual_warm_only_horizons: list[int] | None = None,
) -> pd.DataFrame:
    effective_warm_only_horizons = normalize_residual_warm_only_horizons(
        residual_warm_only_horizons,
        residual_horizons=residual_horizons,
        apply_residual_only_warm=apply_residual_only_warm,
    )
    train_seen_pairs = (
        build_seen_sub_code_pairs(train_frame) if effective_warm_only_horizons else None
    )
    train_oof = build_oof_incumbent_predictions(train_frame, incumbent_config)
    has_oof = train_oof.notna()

    predict_inc = _fit_predict_incumbent(train_frame, predict_frame, incumbent_config)
    train_frame = train_frame.copy()
    train_frame["incumbent_pred"] = train_oof.to_numpy(dtype=float)
    predict_frame = predict_frame.copy()
    predict_frame["incumbent_pred"] = predict_inc

    transformer = build_transformer(incumbent_config or {})
    if bool((incumbent_config or {}).get("transformer") is not None):
        train_frame = transformer.fit(train_frame).transform_batch(train_frame)
        predict_frame = transformer.transform_batch(predict_frame)

    lgbm_train = train_frame.loc[has_oof].copy()

    residual_mask_train = lgbm_train[HORIZON_COLUMN].isin(residual_horizons)
    residual_mask_predict = build_residual_prediction_mask(
        predict_frame,
        residual_horizons=residual_horizons,
        apply_residual_only_warm=apply_residual_only_warm,
        seen_pairs=train_seen_pairs,
        residual_warm_only_horizons=effective_warm_only_horizons,
    )

    lgbm_train = lgbm_train.loc[residual_mask_train].copy()
    lgbm_train[TARGET_COLUMN] = (
        lgbm_train[TARGET_COLUMN] - lgbm_train["incumbent_pred"]
    )

    model = LGBMForecaster(
        params=lgbm_params,
        extra_numeric_columns=["incumbent_pred"],
        use_sample_weight=use_sample_weight,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(lgbm_train, eval_frame=None)

    predictions = predict_frame["incumbent_pred"].copy()
    if residual_mask_predict.any():
        residual_preds = model.predict(predict_frame.loc[residual_mask_predict])
        predictions.loc[residual_mask_predict] = (
            predict_frame.loc[residual_mask_predict, "incumbent_pred"] + residual_preds
        )
    return pd.DataFrame(
        {
            ID_COLUMN: predict_frame[ID_COLUMN].to_numpy(copy=False),
            "prediction": predictions.to_numpy(copy=False),
        }
    )


def calibrate_residual_lgbm_n_estimators(
    *,
    train_frame: pd.DataFrame,
    incumbent_config: dict[str, object],
    residual_horizons: list[int],
    lgbm_params: dict[str, object],
    max_n_estimators: int,
    early_stopping_rounds: int,
    calibration_holdout_steps: int,
    calibration_min_train_steps: int,
    use_sample_weight: bool,
    apply_residual_only_warm: bool = False,
    residual_warm_only_horizons: list[int] | None = None,
) -> tuple[int, dict[str, float]]:
    max_ts = int(train_frame[TIME_COLUMN].max())
    calibration_train_end = max_ts - calibration_holdout_steps
    if calibration_train_end < calibration_min_train_steps:
        raise ValueError(
            "Calibration split leaves too little prefix history for a causal holdout."
        )

    calibration_train = train_frame.loc[
        train_frame[TIME_COLUMN] <= calibration_train_end
    ].copy()
    calibration_valid = train_frame.loc[
        train_frame[TIME_COLUMN] > calibration_train_end
    ].copy()

    train_oof = build_oof_incumbent_predictions(
        calibration_train,
        incumbent_config,
        oof_holdout_steps=OOF_HOLDOUT_STEPS,
        oof_step_size=OOF_STEP_SIZE,
        oof_min_train_steps=OOF_MIN_TRAIN_STEPS,
    )
    calibration_inc = _fit_predict_incumbent(
        calibration_train,
        calibration_valid,
        incumbent_config,
    )
    calibration_train["incumbent_pred"] = train_oof.to_numpy(dtype=float)
    calibration_valid = calibration_valid.copy()
    calibration_valid["incumbent_pred"] = calibration_inc

    transformer = build_transformer(incumbent_config or {})
    if bool((incumbent_config or {}).get("transformer") is not None):
        calibration_train = transformer.fit(calibration_train).transform_batch(calibration_train)
        calibration_valid = transformer.transform_batch(calibration_valid)

    has_oof = train_oof.notna()
    lgbm_train = calibration_train.loc[has_oof].copy()

    residual_mask_train = lgbm_train[HORIZON_COLUMN].isin(residual_horizons)
    effective_warm_only_horizons = normalize_residual_warm_only_horizons(
        residual_warm_only_horizons,
        residual_horizons=residual_horizons,
        apply_residual_only_warm=apply_residual_only_warm,
    )
    train_seen_pairs = (
        build_seen_sub_code_pairs(calibration_train)
        if effective_warm_only_horizons
        else None
    )
    residual_mask_valid = build_residual_prediction_mask(
        calibration_valid,
        residual_horizons=residual_horizons,
        apply_residual_only_warm=apply_residual_only_warm,
        seen_pairs=train_seen_pairs,
        residual_warm_only_horizons=effective_warm_only_horizons,
    )

    lgbm_train = lgbm_train.loc[residual_mask_train].copy()
    lgbm_train[TARGET_COLUMN] = (
        lgbm_train[TARGET_COLUMN] - lgbm_train["incumbent_pred"]
    )
    lgbm_valid = calibration_valid.loc[residual_mask_valid].copy()
    lgbm_valid[TARGET_COLUMN] = (
        lgbm_valid[TARGET_COLUMN] - lgbm_valid["incumbent_pred"]
    )

    model = LGBMForecaster(
        params=lgbm_params,
        extra_numeric_columns=["incumbent_pred"],
        use_sample_weight=use_sample_weight,
        n_estimators=max_n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(lgbm_train, eval_frame=lgbm_valid)
    best_iteration = int(model.model_.best_iteration)
    if best_iteration <= 0:
        best_iteration = int(max_n_estimators)

    metadata = {
        "calibration_train_end_ts": float(calibration_train_end),
        "calibration_valid_start_ts": float(calibration_train_end + 1),
        "calibration_valid_end_ts": float(max_ts),
        "best_iteration": float(best_iteration),
    }
    return best_iteration, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--test-path", default="data/test.parquet")
    parser.add_argument("--output", default="outputs/submissions/lgbm_submission.csv")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--residual-horizons", type=int, nargs="+", default=None)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-child-samples", type=int, default=None)
    parser.add_argument("--min-sum-hessian-in-leaf", type=float, default=None)
    parser.add_argument("--feature-fraction", type=float, default=None)
    parser.add_argument("--bagging-fraction", type=float, default=None)
    parser.add_argument("--bagging-freq", type=int, default=None)
    parser.add_argument("--min-split-gain", type=float, default=None)
    parser.add_argument("--reg-alpha", type=float, default=None)
    parser.add_argument("--reg-lambda", type=float, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--device-type", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--calibration-holdout-steps", type=int, default=360)
    parser.add_argument("--calibration-min-train-steps", type=int, default=2000)
    parser.add_argument("--calibration-max-n-estimators", type=int, default=2000)
    parser.add_argument(
        "--apply-residual-only-warm",
        action="store_true",
        help="Apply residual corrections only to test rows whose (code, sub_code) key is already seen in the training prefix.",
    )
    parser.add_argument(
        "--residual-warm-only-horizons",
        type=int,
        nargs="+",
        default=None,
        help="If set, apply the warm-only gate only on these residual horizons.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else output_path.with_name(f"{output_path.stem}_metadata.json")
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    residual_horizons = normalize_residual_horizons(args.residual_horizons)
    residual_warm_only_horizons = normalize_residual_warm_only_horizons(
        args.residual_warm_only_horizons,
        residual_horizons=residual_horizons,
        apply_residual_only_warm=args.apply_residual_only_warm,
    )
    incumbent_config = BASELINE_CONFIGS[INCUMBENT_CONFIG_NAME]
    lgbm_params = build_lgbm_params(args)

    all_columns = get_columns(args.train_path)
    feature_cols = [
        column for column in detect_feature_columns(all_columns) if column != "feature_al"
    ]
    train_cols = [
        ID_COLUMN,
        "code",
        "sub_code",
        "sub_category",
        HORIZON_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
        WEIGHT_COLUMN,
        *feature_cols,
    ]
    test_cols = [
        ID_COLUMN,
        "code",
        "sub_code",
        "sub_category",
        HORIZON_COLUMN,
        TIME_COLUMN,
        *feature_cols,
    ]

    train_frame = read_parquet_frame(args.train_path, columns=train_cols)
    test_frame = read_parquet_frame(args.test_path, columns=test_cols)

    calibrated_best_iteration = None
    calibration_metadata: dict[str, float] | None = None
    if args.n_estimators is None:
        calibrated_best_iteration, calibration_metadata = (
            calibrate_residual_lgbm_n_estimators(
                train_frame=train_frame,
                incumbent_config=incumbent_config,
                residual_horizons=residual_horizons,
                lgbm_params=lgbm_params,
                max_n_estimators=int(args.calibration_max_n_estimators),
                early_stopping_rounds=int(args.early_stopping_rounds),
                calibration_holdout_steps=int(args.calibration_holdout_steps),
                calibration_min_train_steps=int(args.calibration_min_train_steps),
                use_sample_weight=not args.unweighted,
                apply_residual_only_warm=args.apply_residual_only_warm,
                residual_warm_only_horizons=residual_warm_only_horizons,
            )
        )

    full_train_n_estimators = resolve_full_train_n_estimators(
        args.n_estimators,
        calibrated_best_iteration,
    )
    predictions = predict_residual_lgbm(
        train_frame=train_frame,
        predict_frame=test_frame,
        incumbent_config=incumbent_config,
        residual_horizons=residual_horizons,
        lgbm_params=lgbm_params,
        n_estimators=full_train_n_estimators,
        early_stopping_rounds=int(args.early_stopping_rounds),
        use_sample_weight=not args.unweighted,
        apply_residual_only_warm=args.apply_residual_only_warm,
        residual_warm_only_horizons=residual_warm_only_horizons,
    )
    predictions.to_csv(output_path, index=False)

    metadata = {
        "incumbent": INCUMBENT_CONFIG_NAME,
        "weighted": not args.unweighted,
        "apply_residual_only_warm": args.apply_residual_only_warm,
        "residual_warm_only_horizons": residual_warm_only_horizons,
        "residual_horizons": residual_horizons,
        "params": lgbm_params,
        "full_train_n_estimators": full_train_n_estimators,
        "calibration": calibration_metadata,
        "output": str(output_path),
    }
    summary_output.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote predictions to {output_path}")
    print(f"Wrote metadata to {summary_output}")


if __name__ == "__main__":
    main()
