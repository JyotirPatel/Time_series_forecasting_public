#!/usr/bin/env python
"""Evaluate strict-OOF normalized residual models on top of a saved base predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
from ts_forecasting.io import get_columns, read_parquet_frame
from ts_forecasting.metrics import weighted_rmse_breakdown, weighted_rmse_score_from_sums
from ts_forecasting.residuals import (
    NORMALIZED_RESIDUAL_TARGET_COLUMN,
    RESIDUAL_CATEGORICAL_COLUMNS,
    RESIDUAL_FALLBACK_SCALE_COLUMNS,
    RESIDUAL_PRIMARY_SCALE_COLUMNS,
    RESIDUAL_SCALE_COLUMN,
    apply_normalized_residual_corrections,
    attach_residual_scales,
    build_residual_scale_lookup,
    build_slice_mask,
    compute_global_residual_scale,
    compute_normalized_residual_targets,
    detect_residual_feature_columns,
)


RESIDUAL_CONFIGS: dict[str, dict[str, object]] = {
    "ridge_oofnorm_triweak_h1_tw720_s015_q85": {
        "model_type": "ridge",
        "horizons": [1],
        "sub_categories": ["DPPUO5X2", "PZ9S1Z4V", "PHHHVYZI"],
        "residual_train_steps": 720,
        "correction_scale": 0.15,
        "correction_clip_quantile": 0.85,
        "max_train_rows": 120_000,
        "alpha": 10.0,
    },
    "ridge_oofnorm_triweak_h1_tw720_s022_q90": {
        "model_type": "ridge",
        "horizons": [1],
        "sub_categories": ["DPPUO5X2", "PZ9S1Z4V", "PHHHVYZI"],
        "residual_train_steps": 720,
        "correction_scale": 0.22,
        "correction_clip_quantile": 0.90,
        "max_train_rows": 120_000,
        "alpha": 10.0,
    },
    "ridge_oofnorm_triweak_h3_tw720_s015_q85": {
        "model_type": "ridge",
        "horizons": [3],
        "sub_categories": ["DPPUO5X2", "PZ9S1Z4V", "PHHHVYZI"],
        "residual_train_steps": 720,
        "correction_scale": 0.15,
        "correction_clip_quantile": 0.85,
        "max_train_rows": 120_000,
        "alpha": 10.0,
    },
    "ridge_oofnorm_triweak_h3_tw720_s022_q90": {
        "model_type": "ridge",
        "horizons": [3],
        "sub_categories": ["DPPUO5X2", "PZ9S1Z4V", "PHHHVYZI"],
        "residual_train_steps": 720,
        "correction_scale": 0.22,
        "correction_clip_quantile": 0.90,
        "max_train_rows": 120_000,
        "alpha": 10.0,
    },
    "ridge_oofnorm_triweak_h13_tw720_s015_q85": {
        "model_type": "ridge",
        "horizons": [1, 3],
        "sub_categories": ["DPPUO5X2", "PZ9S1Z4V", "PHHHVYZI"],
        "residual_train_steps": 720,
        "correction_scale": 0.15,
        "correction_clip_quantile": 0.85,
        "max_train_rows": 180_000,
        "alpha": 10.0,
    },
    "ridge_oofnorm_triweak_h13_tw720_s022_q90": {
        "model_type": "ridge",
        "horizons": [1, 3],
        "sub_categories": ["DPPUO5X2", "PZ9S1Z4V", "PHHHVYZI"],
        "residual_train_steps": 720,
        "correction_scale": 0.22,
        "correction_clip_quantile": 0.90,
        "max_train_rows": 180_000,
        "alpha": 10.0,
    },
}


def _sql_list(values: list[object]) -> str:
    rendered: list[str] = []
    for value in values:
        if isinstance(value, str):
            rendered.append("'" + value.replace("'", "''") + "'")
        else:
            rendered.append(str(value))
    return ", ".join(rendered)


def _build_union_where(configs: dict[str, dict[str, object]]) -> str:
    all_horizons = sorted(
        {
            int(horizon)
            for config in configs.values()
            for horizon in config.get("horizons", [])
        }
    )
    all_sub_categories = sorted(
        {
            str(sub_category)
            for config in configs.values()
            for sub_category in config.get("sub_categories", [])
        }
    )
    clauses: list[str] = []
    if all_horizons:
        clauses.append(f"{HORIZON_COLUMN} IN ({_sql_list(all_horizons)})")
    if all_sub_categories:
        clauses.append(f"{SUB_CATEGORY_COLUMN} IN ({_sql_list(all_sub_categories)})")
    return " AND ".join(clauses) if clauses else "TRUE"


def _build_residual_model(
    config: dict[str, object],
    *,
    numeric_columns: list[str],
) -> Pipeline:
    model_type = str(config["model_type"])
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                RESIDUAL_CATEGORICAL_COLUMNS,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ],
        sparse_threshold=0.0,
    )

    if model_type == "ridge":
        model = Ridge(alpha=float(config["alpha"]))
    elif model_type == "huber":
        model = HuberRegressor(
            alpha=float(config["alpha"]),
            epsilon=float(config["epsilon"]),
            max_iter=int(config["max_iter"]),
        )
    elif model_type == "hist_gbm":
        model = HistGradientBoostingRegressor(
            learning_rate=float(config["learning_rate"]),
            max_depth=int(config["max_depth"]),
            max_iter=int(config["max_iter"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            random_state=42,
        )
    else:
        raise ValueError(f"unsupported residual model type: {model_type}")

    return Pipeline([("pre", preprocessor), ("model", model)])


def _sample_frame(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.sort_values(TIME_COLUMN)
    return frame.sample(n=max_rows, random_state=random_state).sort_values(TIME_COLUMN)


def _score_predictions(scored_frame: pd.DataFrame) -> dict[str, float]:
    breakdown = weighted_rmse_breakdown(
        scored_frame[TARGET_COLUMN].to_numpy(),
        scored_frame["prediction"].to_numpy(),
        scored_frame[WEIGHT_COLUMN].to_numpy(),
    )
    return breakdown.to_dict()


def _aggregate_config_scores(
    config_fold_rows: list[dict[str, object]],
) -> dict[str, float | int | None]:
    overall_scores = [float(row["overall_score"]) for row in config_fold_rows]
    base_scores = [float(row["base_overall_score"]) for row in config_fold_rows]
    total_error_sum = float(sum(float(row["overall_error_sum"]) for row in config_fold_rows))
    total_denom_sum = float(sum(float(row["overall_denom_sum"]) for row in config_fold_rows))
    total_base_error_sum = float(
        sum(float(row["base_overall_error_sum"]) for row in config_fold_rows)
    )
    total_base_denom_sum = float(
        sum(float(row["base_overall_denom_sum"]) for row in config_fold_rows)
    )

    aggregate_score = weighted_rmse_score_from_sums(
        error_sum=total_error_sum,
        denom_sum=total_denom_sum,
    )
    base_aggregate_score = weighted_rmse_score_from_sums(
        error_sum=total_base_error_sum,
        denom_sum=total_base_denom_sum,
    )
    return {
        "aggregate_overall_score": aggregate_score,
        "aggregate_overall_error_sum": total_error_sum,
        "aggregate_overall_denom_sum": total_denom_sum,
        "aggregate_base_overall_score": base_aggregate_score,
        "aggregate_score_gain_vs_base": aggregate_score - base_aggregate_score,
        "mean_fold_overall_score": mean(overall_scores),
        "std_fold_overall_score": pstdev(overall_scores) if len(overall_scores) > 1 else 0.0,
        "mean_fold_base_overall_score": mean(base_scores),
        "total_rows_changed": int(sum(int(row["rows_changed"]) for row in config_fold_rows)),
        "total_slice_validation_rows": int(
            sum(int(row["slice_validation_rows"]) for row in config_fold_rows)
        ),
        "total_oof_training_rows_available": int(
            sum(int(row["oof_training_rows_available"]) for row in config_fold_rows)
        ),
        "total_oof_training_rows_used": int(
            sum(int(row["training_rows_used"]) for row in config_fold_rows)
        ),
    }


def _build_prefix_scale_artifacts(
    train_path: str,
    *,
    train_end_ts: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    scale_train_frame = read_parquet_frame(
        train_path,
        columns=[
            CODE_COLUMN,
            SUB_CATEGORY_COLUMN,
            HORIZON_COLUMN,
            TARGET_COLUMN,
            WEIGHT_COLUMN,
        ],
        where=f"{TIME_COLUMN} <= {train_end_ts}",
    )
    primary_lookup = build_residual_scale_lookup(
        scale_train_frame,
        level_columns=RESIDUAL_PRIMARY_SCALE_COLUMNS,
    )
    fallback_lookup = build_residual_scale_lookup(
        scale_train_frame,
        level_columns=RESIDUAL_FALLBACK_SCALE_COLUMNS,
    )
    global_scale = compute_global_residual_scale(scale_train_frame)
    return primary_lookup, fallback_lookup, global_scale


def _prepare_oof_fold_frames(
    train_path: str,
    *,
    evaluation_dir: Path,
    fold: dict[str, object],
    base_baseline: str,
    residual_numeric_columns: list[str],
    union_where: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_name = str(fold["name"])
    train_end_ts = int(fold["train_end_ts"])
    validation_start_ts = int(fold["validation_start_ts"])
    validation_end_ts = int(fold["validation_end_ts"])

    base_validation_predictions = pd.read_csv(
        evaluation_dir / f"{fold_name}_{base_baseline}_predictions.csv"
    ).rename(columns={"prediction": "base_prediction"})

    full_validation_frame = read_parquet_frame(
        train_path,
        columns=[
            ID_COLUMN,
            CODE_COLUMN,
            SUB_CODE_COLUMN,
            SUB_CATEGORY_COLUMN,
            HORIZON_COLUMN,
            TIME_COLUMN,
            TARGET_COLUMN,
            WEIGHT_COLUMN,
        ],
        where=(
            f"{TIME_COLUMN} >= {validation_start_ts} "
            f"AND {TIME_COLUMN} <= {validation_end_ts}"
        ),
        order_by=TIME_COLUMN,
    )
    full_validation_frame = full_validation_frame.merge(
        base_validation_predictions[[ID_COLUMN, "base_prediction"]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if full_validation_frame["base_prediction"].isna().any():
        raise ValueError(f"missing base predictions for fold {fold_name}")

    primary_lookup, fallback_lookup, global_scale = _build_prefix_scale_artifacts(
        train_path,
        train_end_ts=train_end_ts,
    )
    full_validation_frame = attach_residual_scales(
        full_validation_frame,
        primary_lookup=primary_lookup,
        fallback_lookup=fallback_lookup,
        global_scale=global_scale,
    )
    full_validation_frame = compute_normalized_residual_targets(full_validation_frame)
    full_validation_frame["source_fold_name"] = fold_name
    full_validation_frame["source_train_end_ts"] = train_end_ts

    validation_union = read_parquet_frame(
        train_path,
        columns=[
            ID_COLUMN,
            CODE_COLUMN,
            SUB_CATEGORY_COLUMN,
            HORIZON_COLUMN,
            TIME_COLUMN,
            TARGET_COLUMN,
            WEIGHT_COLUMN,
            *[column for column in residual_numeric_columns if column != TIME_COLUMN],
        ],
        where=(
            f"{TIME_COLUMN} >= {validation_start_ts} "
            f"AND {TIME_COLUMN} <= {validation_end_ts} "
            f"AND {union_where}"
        ),
        order_by=TIME_COLUMN,
    )
    validation_union = validation_union.merge(
        full_validation_frame[
            [
                ID_COLUMN,
                "base_prediction",
                RESIDUAL_SCALE_COLUMN,
                NORMALIZED_RESIDUAL_TARGET_COLUMN,
                "source_fold_name",
                "source_train_end_ts",
            ]
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    return full_validation_frame, validation_union


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-summary-path",
        default="outputs/baselines/forward_cv_h360_f4_s360_code_sub_code/summary.json",
    )
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--base-baseline", default="smoothed_stable_group_mean_pw100")
    parser.add_argument("--output-dir", default="outputs/residual_models")
    parser.add_argument("--output-suffix", default="oof_norm")
    args = parser.parse_args()

    base_summary_path = Path(args.base_summary_path)
    base_summary = json.loads(base_summary_path.read_text())
    if args.base_baseline not in base_summary["aggregate"]:
        raise ValueError(f"base baseline {args.base_baseline} is not present in {base_summary_path}")

    evaluation_dir = base_summary_path.parent
    output_dir = Path(args.output_dir) / f"{evaluation_dir.name}_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    residual_numeric_columns = detect_residual_feature_columns(
        get_columns(args.train_path),
        banned_feature_columns={"feature_al"},
    )
    union_where = _build_union_where(RESIDUAL_CONFIGS)

    summary: dict[str, object] = {
        "base_summary_path": str(base_summary_path),
        "base_baseline": args.base_baseline,
        "canonical_score_name": "aggregate_overall_score",
        "residual_target_type": "strict_oof_normalized",
        "residual_scale_levels": {
            "primary": RESIDUAL_PRIMARY_SCALE_COLUMNS,
            "fallback": RESIDUAL_FALLBACK_SCALE_COLUMNS,
            "global": "weighted_target_rms",
        },
        "configs": {},
    }

    per_config_fold_rows: dict[str, list[dict[str, object]]] = {
        config_name: [] for config_name in RESIDUAL_CONFIGS
    }
    fold_full_frames: dict[str, pd.DataFrame] = {}
    fold_union_frames: dict[str, pd.DataFrame] = {}
    base_breakdowns_by_fold: dict[str, dict[str, float]] = {}
    oof_full_frames: list[pd.DataFrame] = []
    oof_union_frames: list[pd.DataFrame] = []

    for fold in base_summary["folds"]:
        full_validation_frame, validation_union = _prepare_oof_fold_frames(
            args.train_path,
            evaluation_dir=evaluation_dir,
            fold=fold,
            base_baseline=args.base_baseline,
            residual_numeric_columns=residual_numeric_columns,
            union_where=union_where,
        )
        fold_name = str(fold["name"])
        fold_full_frames[fold_name] = full_validation_frame
        fold_union_frames[fold_name] = validation_union
        base_breakdowns_by_fold[fold_name] = _score_predictions(
            full_validation_frame.rename(columns={"base_prediction": "prediction"})[
                [TARGET_COLUMN, WEIGHT_COLUMN, "prediction"]
            ]
        )
        oof_full_frames.append(full_validation_frame)
        oof_union_frames.append(validation_union)

    oof_base_predictions_path = output_dir / "oof_base_predictions.parquet"
    pd.concat(oof_full_frames, ignore_index=True).to_parquet(oof_base_predictions_path, index=False)
    oof_residual_pool_path = output_dir / "oof_residual_pool.parquet"
    pd.concat(oof_union_frames, ignore_index=True).to_parquet(oof_residual_pool_path, index=False)
    summary["oof_base_predictions_path"] = str(oof_base_predictions_path)
    summary["oof_residual_pool_path"] = str(oof_residual_pool_path)

    fold_names_in_order = [str(fold["name"]) for fold in base_summary["folds"]]

    for fold in base_summary["folds"]:
        fold_name = str(fold["name"])
        train_end_ts = int(fold["train_end_ts"])
        full_validation_frame = fold_full_frames[fold_name]
        validation_union = fold_union_frames[fold_name]
        base_breakdown = base_breakdowns_by_fold[fold_name]

        previous_training_frames = [
            fold_union_frames[previous_fold_name]
            for previous_fold_name in fold_names_in_order
            if previous_fold_name != fold_name
            and int(fold_union_frames[previous_fold_name][TIME_COLUMN].max())
            <= train_end_ts
        ]
        if previous_training_frames:
            oof_training_pool = pd.concat(previous_training_frames, ignore_index=True)
        else:
            oof_training_pool = validation_union.head(0).copy()

        for config_name, config in RESIDUAL_CONFIGS.items():
            config_output_dir = output_dir / config_name
            config_output_dir.mkdir(parents=True, exist_ok=True)

            min_residual_train_ts = max(
                1,
                train_end_ts - int(config["residual_train_steps"]) + 1,
            )
            gate_mask_train = build_slice_mask(
                oof_training_pool,
                horizons=config.get("horizons"),
                sub_categories=config.get("sub_categories"),
                codes=config.get("codes"),
            )
            gate_mask_validation = build_slice_mask(
                validation_union,
                horizons=config.get("horizons"),
                sub_categories=config.get("sub_categories"),
                codes=config.get("codes"),
            )
            gated_train = oof_training_pool.loc[gate_mask_train].copy()
            gated_train = gated_train.loc[gated_train[TIME_COLUMN] >= min_residual_train_ts].copy()
            gated_validation = validation_union.loc[gate_mask_validation].copy()

            final_predictions = full_validation_frame[[ID_COLUMN, "base_prediction"]].rename(
                columns={"base_prediction": "prediction"}
            )
            rows_changed = 0
            clip_abs: float | None = None
            training_rows_used = 0
            oof_training_rows_available = int(len(gated_train))
            if oof_training_rows_available >= 1_000 and len(gated_validation) > 0:
                gated_train = _sample_frame(
                    gated_train,
                    max_rows=int(config["max_train_rows"]),
                )
                training_rows_used = int(len(gated_train))
                clip_abs = float(
                    np.quantile(
                        np.abs(gated_train[NORMALIZED_RESIDUAL_TARGET_COLUMN].to_numpy()),
                        float(config["correction_clip_quantile"]),
                    )
                )

                residual_model = _build_residual_model(
                    config,
                    numeric_columns=residual_numeric_columns,
                )
                residual_model.fit(
                    gated_train[RESIDUAL_CATEGORICAL_COLUMNS + residual_numeric_columns],
                    gated_train[NORMALIZED_RESIDUAL_TARGET_COLUMN].to_numpy(),
                    model__sample_weight=gated_train[WEIGHT_COLUMN].to_numpy(),
                )
                raw_normalized_correction = residual_model.predict(
                    gated_validation[RESIDUAL_CATEGORICAL_COLUMNS + residual_numeric_columns]
                )
                shrunk_correction = apply_normalized_residual_corrections(
                    raw_normalized_correction,
                    scales=gated_validation[RESIDUAL_SCALE_COLUMN].to_numpy(),
                    correction_scale=float(config["correction_scale"]),
                    clip_abs=clip_abs,
                )
                adjusted_predictions = pd.DataFrame(
                    {
                        ID_COLUMN: gated_validation[ID_COLUMN].to_numpy(),
                        "prediction": gated_validation["base_prediction"].to_numpy()
                        + shrunk_correction,
                    }
                )
                rows_changed = int(len(adjusted_predictions))
                final_predictions = final_predictions.merge(
                    adjusted_predictions,
                    on=ID_COLUMN,
                    how="left",
                    validate="one_to_one",
                    suffixes=("_base", ""),
                )
                final_predictions["prediction"] = final_predictions["prediction"].fillna(
                    final_predictions["prediction_base"]
                )
                final_predictions = final_predictions[[ID_COLUMN, "prediction"]]

            final_predictions.to_csv(
                config_output_dir / f"{fold_name}_predictions.csv",
                index=False,
            )
            scored_frame = full_validation_frame[[ID_COLUMN, TARGET_COLUMN, WEIGHT_COLUMN]].merge(
                final_predictions,
                on=ID_COLUMN,
                how="left",
                validate="one_to_one",
            )
            score_breakdown = _score_predictions(scored_frame)

            slice_validation_breakdown: dict[str, float | None]
            if len(gated_validation) > 0:
                slice_scores = gated_validation[[ID_COLUMN, TARGET_COLUMN, WEIGHT_COLUMN]].merge(
                    final_predictions,
                    on=ID_COLUMN,
                    how="left",
                    validate="one_to_one",
                )
                slice_validation_breakdown = _score_predictions(slice_scores)
            else:
                slice_validation_breakdown = {
                    "score": None,
                    "error_sum": None,
                    "denom_sum": None,
                    "ratio": None,
                    "clipped_ratio": None,
                }

            per_config_fold_rows[config_name].append(
                {
                    "fold_name": fold_name,
                    "overall_score": score_breakdown["score"],
                    "overall_error_sum": score_breakdown["error_sum"],
                    "overall_denom_sum": score_breakdown["denom_sum"],
                    "base_overall_score": base_breakdown["score"],
                    "base_overall_error_sum": base_breakdown["error_sum"],
                    "base_overall_denom_sum": base_breakdown["denom_sum"],
                    "score_gain_vs_base": score_breakdown["score"] - base_breakdown["score"],
                    "rows_changed": rows_changed,
                    "oof_training_rows_available": oof_training_rows_available,
                    "training_rows_used": training_rows_used,
                    "slice_validation_rows": int(len(gated_validation)),
                    "slice_validation_score": slice_validation_breakdown["score"],
                    "clip_abs_normalized": clip_abs,
                    "mean_train_scale": (
                        float(gated_train[RESIDUAL_SCALE_COLUMN].mean())
                        if training_rows_used > 0
                        else None
                    ),
                    "mean_validation_scale": (
                        float(gated_validation[RESIDUAL_SCALE_COLUMN].mean())
                        if len(gated_validation) > 0
                        else None
                    ),
                }
            )

    for config_name, config_fold_rows in per_config_fold_rows.items():
        summary["configs"][config_name] = {
            "config": RESIDUAL_CONFIGS[config_name],
            "aggregate": _aggregate_config_scores(config_fold_rows),
            "folds": config_fold_rows,
        }

    best_config = max(
        summary["configs"],
        key=lambda name: float(summary["configs"][name]["aggregate"]["aggregate_overall_score"]),
    )
    summary["best_config"] = best_config
    summary["best_score"] = summary["configs"][best_config]["aggregate"]["aggregate_overall_score"]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
