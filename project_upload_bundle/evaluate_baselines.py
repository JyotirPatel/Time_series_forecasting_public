#!/usr/bin/env python
"""Evaluate simple leakage-safe baselines on forward validation folds."""

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

import pandas as pd

from ts_forecasting.baselines import (
    LinearBlendRegressor,
    RecentMeanDeltaRegressor,
    SmoothedWeightedGroupMeanRegressor,
    WarmStartLastTargetBlendRegressor,
    WeightedGroupMeanRegressor,
)
from ts_forecasting.constants import GROUP_COLUMNS, ID_COLUMN, TARGET_COLUMN, TIME_COLUMN, WEIGHT_COLUMN
from ts_forecasting.io import (
    get_max_ts_index,
    load_train_validation_split,
    load_train_validation_window,
    read_parquet_frame,
)
from ts_forecasting.metrics import weighted_rmse_breakdown, weighted_rmse_score_from_sums
from ts_forecasting.residuals import FixedPrefixResidualCorrectionRegressor
from ts_forecasting.sequential import IdentityTransformer, SequentialInferencePipeline
from ts_forecasting.validation import (
    annotate_cold_start_rows,
    build_expanding_window_folds,
    build_first_seen_lookup,
    summarize_key_coverage,
)


SMOOTHED_STABLE_GROUP_CONFIG: dict[str, object] = {
    "regressor": "smoothed_group_mean",
    "fallback_levels": [
        ["code", "sub_category", "horizon"],
        ["sub_category", "horizon"],
        ["horizon"],
    ],
    "prior_weight": 100.0,
}
HORIZON_MEAN_CONFIG: dict[str, object] = {
    "regressor": "group_mean",
    "fallback_levels": [["horizon"]],
}
SMOOTHED_STABLE_HORIZON_BLEND_CONFIG: dict[str, object] = {
    "regressor": "linear_blend",
    "weights": [0.6355710510414874, 0.3644289489585126],
    "components": [
        SMOOTHED_STABLE_GROUP_CONFIG,
        HORIZON_MEAN_CONFIG,
    ],
}
WARM_LAST_TARGET_BLEND_RULES: list[dict[str, object]] = [
    {"horizon": 3, "sub_category": "NQ58FVQM", "alpha": 0.05},
    {"horizon": 10, "sub_category": "NQ58FVQM", "alpha": 0.05},
    {"horizon": 10, "sub_category": "V8BKY1IV", "alpha": 0.04},
    {"horizon": 25, "sub_category": "NQ58FVQM", "alpha": 0.12},
    {"horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.01},
    {"horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.04},
    {"horizon": 25, "sub_category": "V8BKY1IV", "alpha": 0.075},
]
K7_H25_WARM_RULES: list[dict[str, object]] = [
    {"code": "K7Y1TTAH", "horizon": 25, "sub_category": "PHHHVYZI", "alpha": 0.15},
    {"code": "K7Y1TTAH", "horizon": 25, "sub_category": "PZ9S1Z4V", "alpha": 0.20},
    {"code": "K7Y1TTAH", "horizon": 25, "sub_category": "DPPUO5X2", "alpha": 0.03},
]
RECENT_DELTA_V1_CONFIG: dict[str, object] = {
    "regressor": "recent_mean_delta",
    "base_predictor": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": SMOOTHED_STABLE_HORIZON_BLEND_CONFIG,
        "blend_rules": WARM_LAST_TARGET_BLEND_RULES,
    },
    "recent_window": 180,
    "min_recent_rows": 50,
    "beta": 0.15,
    "horizons": [10, 25],
}
RIDGE_TRIWEAK_RAW_CONFIG: dict[str, object] = {
    "regressor": "fixed_prefix_residual",
    "base_predictor": SMOOTHED_STABLE_GROUP_CONFIG,
    "model_type": "ridge",
    "horizons": [1, 3],
    "sub_categories": ["DPPUO5X2", "PZ9S1Z4V", "PHHHVYZI"],
    "residual_train_steps": 720,
    "correction_scale": 0.22,
    "correction_clip_quantile": 0.9,
    "max_train_rows": 180_000,
    "alpha": 10.0,
    "banned_feature_columns": ["feature_al"],
}


BASELINE_CONFIGS: dict[str, dict[str, object]] = {
    "horizon_mean": HORIZON_MEAN_CONFIG,
    "stable_group_mean": {
        "regressor": "group_mean",
        "fallback_levels": [
            ["code", "sub_category", "horizon"],
            ["sub_category", "horizon"],
            ["horizon"],
        ],
    },
    "smoothed_stable_group_mean_pw100": SMOOTHED_STABLE_GROUP_CONFIG,
    "smoothed_stable_horizon_blend": SMOOTHED_STABLE_HORIZON_BLEND_CONFIG,
    "smoothed_stable_horizon_blend_last_target_rules": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": SMOOTHED_STABLE_HORIZON_BLEND_CONFIG,
        "blend_rules": WARM_LAST_TARGET_BLEND_RULES,
    },
    "smoothed_stable_horizon_blend_last_target_recent_delta_v1": RECENT_DELTA_V1_CONFIG,
    "smoothed_stable_horizon_blend_last_target_recent_delta_k7_v1": {
        "regressor": "warm_start_last_target_blend",
        "base_predictor": RECENT_DELTA_V1_CONFIG,
        "blend_rules": K7_H25_WARM_RULES,
    },
    "ridge_triweak_h13_tw720_s022_q90": {
        **RIDGE_TRIWEAK_RAW_CONFIG,
        "run_by_default": False,
    },
    "ridge_triweak_h13_tw720_s022_q90_horizon_blend": {
        "regressor": "linear_blend",
        "weights": [0.63786, 0.36214],
        "components": [
            RIDGE_TRIWEAK_RAW_CONFIG,
            HORIZON_MEAN_CONFIG,
        ],
        "run_by_default": False,
    },
    "hierarchical_group_mean": {
        "regressor": "group_mean",
        "fallback_levels": [
            [*GROUP_COLUMNS],
            ["code", "sub_category", "horizon"],
            ["sub_category", "horizon"],
            ["horizon"],
        ],
    },
}

COLD_START_LEVELS: dict[str, list[str]] = {
    "code_sub_code": ["code", "sub_code"],
    "triplet": ["code", "sub_code", "sub_category"],
    "full_group": [*GROUP_COLUMNS],
    "code_sub_category_horizon": ["code", "sub_category", "horizon"],
}

COVERAGE_LEVELS: dict[str, list[str]] = {
    "code_sub_code": ["code", "sub_code"],
    "code_sub_category_horizon": ["code", "sub_category", "horizon"],
}


def build_predictor(config: dict[str, object]):
    regressor_type = str(config["regressor"])
    if regressor_type == "group_mean":
        return WeightedGroupMeanRegressor(
            fallback_levels=[list(level) for level in config["fallback_levels"]],
        )
    if regressor_type == "smoothed_group_mean":
        return SmoothedWeightedGroupMeanRegressor(
            fallback_levels=[list(level) for level in config["fallback_levels"]],
            prior_weight=float(config["prior_weight"]),
        )
    if regressor_type == "linear_blend":
        return LinearBlendRegressor(
            predictors=[
                build_predictor(component_config)
                for component_config in config["components"]
            ],
            weights=[float(weight) for weight in config["weights"]],
        )
    if regressor_type == "warm_start_last_target_blend":
        return WarmStartLastTargetBlendRegressor(
            base_predictor=build_predictor(dict(config["base_predictor"])),
            blend_rules=[dict(rule) for rule in config["blend_rules"]],
        )
    if regressor_type == "recent_mean_delta":
        return RecentMeanDeltaRegressor(
            base_predictor=build_predictor(dict(config["base_predictor"])),
            recent_window=int(config["recent_window"]),
            min_recent_rows=int(config["min_recent_rows"]),
            beta=float(config["beta"]),
            level_columns=list(config.get("level_columns", ["code", "sub_category", "horizon"])),
            horizons=(
                [int(horizon) for horizon in config["horizons"]]
                if config.get("horizons") is not None
                else None
            ),
        )
    if regressor_type == "fixed_prefix_residual":
        return FixedPrefixResidualCorrectionRegressor(
            base_predictor=build_predictor(dict(config["base_predictor"])),
            model_type=str(config["model_type"]),
            horizons=config.get("horizons"),
            sub_categories=config.get("sub_categories"),
            codes=config.get("codes"),
            residual_train_steps=int(config["residual_train_steps"]),
            correction_scale=float(config["correction_scale"]),
            correction_clip_quantile=float(config["correction_clip_quantile"]),
            max_train_rows=int(config["max_train_rows"]),
            alpha=float(config.get("alpha", 10.0)),
            epsilon=float(config.get("epsilon", 1.35)),
            max_iter=int(config.get("max_iter", 200)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            max_depth=int(config.get("max_depth", 3)),
            min_samples_leaf=int(config.get("min_samples_leaf", 200)),
            banned_feature_columns=set(config.get("banned_feature_columns", [])),
        )
    raise ValueError(f"unknown regressor type: {regressor_type}")


def evaluate_baseline(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    name: str,
    config: dict[str, object],
) -> tuple[dict[str, float], pd.DataFrame]:
    pipeline = SequentialInferencePipeline(
        transformer=IdentityTransformer(),
        predictor=build_predictor(config),
    ).fit(train_frame)
    predictions = pipeline.predict(validation_frame)
    scored = validation_frame[[ID_COLUMN, TARGET_COLUMN, WEIGHT_COLUMN]].merge(
        predictions[[ID_COLUMN, "prediction"]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if scored["prediction"].isna().any():
        raise ValueError(f"baseline {name} produced missing predictions")

    breakdown = weighted_rmse_breakdown(
        scored[TARGET_COLUMN].to_numpy(),
        scored["prediction"].to_numpy(),
        scored[WEIGHT_COLUMN].to_numpy(),
    )
    return breakdown.to_dict(), predictions


def summarize_scored_predictions(
    scored: pd.DataFrame,
    *,
    cold_start_mask: pd.Series,
) -> dict[str, float | int | None]:
    cold_mask = cold_start_mask.to_numpy(dtype=bool)
    warm_mask = ~cold_mask

    overall = weighted_rmse_breakdown(
        scored[TARGET_COLUMN].to_numpy(),
        scored["prediction"].to_numpy(),
        scored[WEIGHT_COLUMN].to_numpy(),
    )
    summary: dict[str, float | int | None] = {
        "overall_score": overall.score,
        "overall_error_sum": overall.error_sum,
        "overall_denom_sum": overall.denom_sum,
        "overall_ratio": overall.ratio,
        "overall_clipped_ratio": overall.clipped_ratio,
        "cold_start_rows": int(cold_mask.sum()),
        "warm_start_rows": int(warm_mask.sum()),
    }

    if cold_mask.any():
        cold = weighted_rmse_breakdown(
            scored.loc[cold_mask, TARGET_COLUMN].to_numpy(),
            scored.loc[cold_mask, "prediction"].to_numpy(),
            scored.loc[cold_mask, WEIGHT_COLUMN].to_numpy(),
        )
        summary["cold_start_score"] = cold.score
        summary["cold_start_error_sum"] = cold.error_sum
        summary["cold_start_denom_sum"] = cold.denom_sum
        summary["cold_start_ratio"] = cold.ratio
    else:
        summary["cold_start_score"] = None
        summary["cold_start_error_sum"] = None
        summary["cold_start_denom_sum"] = None
        summary["cold_start_ratio"] = None

    if warm_mask.any():
        warm = weighted_rmse_breakdown(
            scored.loc[warm_mask, TARGET_COLUMN].to_numpy(),
            scored.loc[warm_mask, "prediction"].to_numpy(),
            scored.loc[warm_mask, WEIGHT_COLUMN].to_numpy(),
        )
        summary["warm_start_score"] = warm.score
        summary["warm_start_error_sum"] = warm.error_sum
        summary["warm_start_denom_sum"] = warm.denom_sum
        summary["warm_start_ratio"] = warm.ratio
    else:
        summary["warm_start_score"] = None
        summary["warm_start_error_sum"] = None
        summary["warm_start_denom_sum"] = None
        summary["warm_start_ratio"] = None

    return summary


def aggregate_fold_scores(
    fold_summaries: list[dict[str, object]],
) -> tuple[dict[str, dict[str, float | int | None]], str]:
    return aggregate_fold_scores_for_names(
        fold_summaries,
        list(BASELINE_CONFIGS),
    )


def aggregate_fold_scores_for_names(
    fold_summaries: list[dict[str, object]],
    baseline_names: list[str],
) -> tuple[dict[str, dict[str, float | int | None]], str]:
    aggregate: dict[str, dict[str, float | int | None]] = {}
    for baseline_name in baseline_names:
        overall_scores = [
            float(fold_summary["results"][baseline_name]["overall_score"])
            for fold_summary in fold_summaries
        ]
        cold_scores = [
            float(result["cold_start_score"])
            for fold_summary in fold_summaries
            if (result := fold_summary["results"][baseline_name])["cold_start_score"] is not None
        ]
        warm_scores = [
            float(result["warm_start_score"])
            for fold_summary in fold_summaries
            if (result := fold_summary["results"][baseline_name])["warm_start_score"] is not None
        ]

        overall_error_sum = float(
            sum(
                float(fold_summary["results"][baseline_name]["overall_error_sum"])
                for fold_summary in fold_summaries
            )
        )
        overall_denom_sum = float(
            sum(
                float(fold_summary["results"][baseline_name]["overall_denom_sum"])
                for fold_summary in fold_summaries
            )
        )
        cold_error_sum = float(
            sum(
                float(result["cold_start_error_sum"])
                for fold_summary in fold_summaries
                if (result := fold_summary["results"][baseline_name])["cold_start_error_sum"]
                is not None
            )
        )
        cold_denom_sum = float(
            sum(
                float(result["cold_start_denom_sum"])
                for fold_summary in fold_summaries
                if (result := fold_summary["results"][baseline_name])["cold_start_denom_sum"]
                is not None
            )
        )
        warm_error_sum = float(
            sum(
                float(result["warm_start_error_sum"])
                for fold_summary in fold_summaries
                if (result := fold_summary["results"][baseline_name])["warm_start_error_sum"]
                is not None
            )
        )
        warm_denom_sum = float(
            sum(
                float(result["warm_start_denom_sum"])
                for fold_summary in fold_summaries
                if (result := fold_summary["results"][baseline_name])["warm_start_denom_sum"]
                is not None
            )
        )

        validation_rows = [int(fold_summary["validation_rows"]) for fold_summary in fold_summaries]
        cold_rows = [
            int(fold_summary["results"][baseline_name]["cold_start_rows"])
            for fold_summary in fold_summaries
        ]
        warm_rows = [
            int(fold_summary["results"][baseline_name]["warm_start_rows"])
            for fold_summary in fold_summaries
        ]

        aggregate[baseline_name] = {
            "aggregate_overall_score": weighted_rmse_score_from_sums(
                error_sum=overall_error_sum,
                denom_sum=overall_denom_sum,
            ),
            "aggregate_overall_error_sum": overall_error_sum,
            "aggregate_overall_denom_sum": overall_denom_sum,
            "aggregate_overall_ratio": overall_error_sum / overall_denom_sum,
            "mean_fold_overall_score": mean(overall_scores),
            "std_fold_overall_score": pstdev(overall_scores) if len(overall_scores) > 1 else 0.0,
            "aggregate_cold_start_score": (
                weighted_rmse_score_from_sums(
                    error_sum=cold_error_sum,
                    denom_sum=cold_denom_sum,
                )
                if cold_denom_sum > 0.0
                else None
            ),
            "aggregate_cold_start_error_sum": cold_error_sum if cold_denom_sum > 0.0 else None,
            "aggregate_cold_start_denom_sum": cold_denom_sum if cold_denom_sum > 0.0 else None,
            "aggregate_warm_start_score": (
                weighted_rmse_score_from_sums(
                    error_sum=warm_error_sum,
                    denom_sum=warm_denom_sum,
                )
                if warm_denom_sum > 0.0
                else None
            ),
            "aggregate_warm_start_error_sum": warm_error_sum if warm_denom_sum > 0.0 else None,
            "aggregate_warm_start_denom_sum": warm_denom_sum if warm_denom_sum > 0.0 else None,
            "mean_fold_cold_start_score": mean(cold_scores) if cold_scores else None,
            "mean_fold_warm_start_score": mean(warm_scores) if warm_scores else None,
            "total_validation_rows": int(sum(validation_rows)),
            "total_cold_start_rows": int(sum(cold_rows)),
            "total_warm_start_rows": int(sum(warm_rows)),
        }

    best_name = max(
        aggregate,
        key=lambda name: float(aggregate[name]["aggregate_overall_score"]),
    )
    return aggregate, best_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--test-path", default="data/test.parquet")
    parser.add_argument("--holdout-steps", type=int, default=180)
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--min-train-steps", type=int, default=1)
    parser.add_argument(
        "--cold-start-level",
        choices=sorted(COLD_START_LEVELS),
        default="code_sub_code",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=sorted(BASELINE_CONFIGS),
        default=None,
    )
    parser.add_argument("--output-dir", default="outputs/baselines")
    args = parser.parse_args()
    selected_baselines = (
        args.baselines
        if args.baselines is not None
        else [
            name
            for name, config in BASELINE_CONFIGS.items()
            if bool(config.get("run_by_default", True))
        ]
    )

    if args.n_folds == 1:
        output_dir = Path(args.output_dir) / f"holdout_{args.holdout_steps}"
    else:
        step_size = args.step_size or args.holdout_steps
        output_dir = (
            Path(args.output_dir)
            / (
                f"forward_cv_h{args.holdout_steps}_f{args.n_folds}"
                f"_s{step_size}_{args.cold_start_level}"
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    cold_start_columns = COLD_START_LEVELS[args.cold_start_level]
    metadata_columns = sorted(
        {
            TIME_COLUMN,
            *cold_start_columns,
            *COVERAGE_LEVELS["code_sub_code"],
            *COVERAGE_LEVELS["code_sub_category_horizon"],
        }
    )
    all_train_metadata = read_parquet_frame(
        args.train_path,
        columns=metadata_columns,
    )
    first_seen_lookup = build_first_seen_lookup(
        all_train_metadata,
        key_columns=cold_start_columns,
    )

    summary: dict[str, object] = {
        "holdout_steps": args.holdout_steps,
        "n_folds": args.n_folds,
        "step_size": args.step_size or args.holdout_steps,
        "cold_start_level": args.cold_start_level,
        "selected_baselines": selected_baselines,
        "canonical_score_name": "aggregate_overall_score",
        "folds": [],
    }

    test_path = Path(args.test_path)
    if test_path.exists():
        test_metadata = read_parquet_frame(test_path, columns=metadata_columns)
        summary["test_reference"] = {
            name: summarize_key_coverage(
                all_train_metadata,
                test_metadata,
                key_columns=columns,
            )
            for name, columns in COVERAGE_LEVELS.items()
        }

    if args.n_folds == 1:
        train_frame, validation_frame, split_ts_index = load_train_validation_split(
            args.train_path,
            holdout_steps=args.holdout_steps,
        )
        folds = [
            {
                "name": f"holdout_{args.holdout_steps}",
                "train_end_ts": split_ts_index,
                "validation_start_ts": split_ts_index + 1,
                "validation_end_ts": int(validation_frame[TIME_COLUMN].max()),
                "train_frame": train_frame,
                "validation_frame": validation_frame,
            }
        ]
    else:
        folds = []
        for fold in build_expanding_window_folds(
            max_ts_index=get_max_ts_index(args.train_path),
            holdout_steps=args.holdout_steps,
            n_folds=args.n_folds,
            step_size=args.step_size,
            min_train_steps=args.min_train_steps,
        ):
            train_frame, validation_frame = load_train_validation_window(
                args.train_path,
                train_end_ts=fold.train_end_ts,
                validation_end_ts=fold.validation_end_ts,
            )
            folds.append(
                {
                    "name": fold.name,
                    "train_end_ts": fold.train_end_ts,
                    "validation_start_ts": fold.validation_start_ts,
                    "validation_end_ts": fold.validation_end_ts,
                    "train_frame": train_frame,
                    "validation_frame": validation_frame,
                }
            )

    fold_summaries: list[dict[str, object]] = []
    for fold in folds:
        train_frame = fold.pop("train_frame")
        validation_frame = fold.pop("validation_frame")

        validation_with_cold_mask = annotate_cold_start_rows(
            validation_frame[[ID_COLUMN, TARGET_COLUMN, WEIGHT_COLUMN, *cold_start_columns]].copy(),
            first_seen_lookup=first_seen_lookup,
            key_columns=cold_start_columns,
            train_end_ts=int(fold["train_end_ts"]),
        )

        fold_summary: dict[str, object] = {
            **fold,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "cold_start_rows": int(validation_with_cold_mask["is_cold_start"].sum()),
            "cold_start_rate": float(validation_with_cold_mask["is_cold_start"].mean()),
            "coverage": {
                name: summarize_key_coverage(
                    train_frame,
                    validation_frame,
                    key_columns=columns,
                )
                for name, columns in COVERAGE_LEVELS.items()
            },
            "results": {},
        }

        for name in selected_baselines:
            config = BASELINE_CONFIGS[name]
            overall_breakdown, predictions = evaluate_baseline(
                train_frame,
                validation_frame,
                name=name,
                config=config,
            )
            prediction_path = output_dir / f"{fold['name']}_{name}_predictions.csv"
            predictions.to_csv(prediction_path, index=False)

            scored = validation_with_cold_mask.merge(
                predictions[[ID_COLUMN, "prediction"]],
                on=ID_COLUMN,
                how="left",
                validate="one_to_one",
            )
            if scored["prediction"].isna().any():
                raise ValueError(f"baseline {name} produced missing predictions on {fold['name']}")

            result_summary = summarize_scored_predictions(
                scored,
                cold_start_mask=scored["is_cold_start"],
            )
            if abs(float(result_summary["overall_score"]) - float(overall_breakdown["score"])) > 1e-12:
                raise ValueError(f"inconsistent overall score for baseline {name} on {fold['name']}")
            if abs(
                float(result_summary["overall_error_sum"]) - float(overall_breakdown["error_sum"])
            ) > 1e-9:
                raise ValueError(f"inconsistent overall numerator for baseline {name} on {fold['name']}")
            if abs(
                float(result_summary["overall_denom_sum"]) - float(overall_breakdown["denom_sum"])
            ) > 1e-9:
                raise ValueError(f"inconsistent overall denominator for baseline {name} on {fold['name']}")
            fold_summary["results"][name] = result_summary

        fold_summaries.append(fold_summary)

    summary["folds"] = fold_summaries
    aggregate, best_name = aggregate_fold_scores_for_names(fold_summaries, selected_baselines)
    summary["aggregate"] = aggregate
    summary["best_baseline"] = best_name
    summary["best_score"] = aggregate[best_name]["aggregate_overall_score"]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
