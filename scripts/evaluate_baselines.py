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

from ts_forecasting.baseline_registry import (
    BASELINE_CONFIGS,
    build_predictor,
    build_transformer,
    resolve_required_columns,
)
from ts_forecasting.constants import (
    CONTEXT_COLUMNS,
    GROUP_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
)
from ts_forecasting.io import (
    get_columns,
    get_max_ts_index,
    load_train_validation_split,
    load_train_validation_window,
    read_parquet_frame,
)
from ts_forecasting.metrics import weighted_rmse_breakdown, weighted_rmse_score_from_sums
from ts_forecasting.sequential import SequentialInferencePipeline
from ts_forecasting.validation import (
    annotate_cold_start_rows,
    build_expanding_window_folds,
    build_first_seen_lookup,
    summarize_key_coverage,
)

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

STRESS_FOLD_SETS: dict[str, list[str]] = {
    "fold_02_03": ["fold_02", "fold_03"],
}
STRESS_WATCHLIST_SLICES: dict[str, dict[str, list[object]]] = {
    "osjl3a7y_long_core": {
        "codes": ["OSJL3A7Y"],
        "sub_categories": ["DPPUO5X2", "PZ9S1Z4V", "PHHHVYZI"],
        "horizons": [10, 25],
    },
    "x9bz68vq_h10_core": {
        "codes": ["X9BZ68VQ"],
        "sub_categories": ["DPPUO5X2", "PHHHVYZI"],
        "horizons": [10],
    },
    "x9bz68vq_h25_core": {
        "codes": ["X9BZ68VQ"],
        "sub_categories": ["DPPUO5X2", "PHHHVYZI", "PZ9S1Z4V", "NQ58FVQM"],
        "horizons": [25],
    },
}


def evaluate_baseline(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    name: str,
    config: dict[str, object],
) -> tuple[dict[str, float], pd.DataFrame]:
    pipeline = SequentialInferencePipeline(
        transformer=build_transformer(config),
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


def aggregate_result_summaries(
    result_summaries: list[dict[str, float | int | None]],
) -> dict[str, float | int | None]:
    if len(result_summaries) == 0:
        return {
            "aggregate_overall_score": None,
            "aggregate_overall_error_sum": None,
            "aggregate_overall_denom_sum": None,
            "aggregate_overall_ratio": None,
            "mean_fold_overall_score": None,
            "std_fold_overall_score": None,
            "aggregate_cold_start_score": None,
            "aggregate_cold_start_error_sum": None,
            "aggregate_cold_start_denom_sum": None,
            "aggregate_warm_start_score": None,
            "aggregate_warm_start_error_sum": None,
            "aggregate_warm_start_denom_sum": None,
            "mean_fold_cold_start_score": None,
            "mean_fold_warm_start_score": None,
            "total_validation_rows": 0,
            "total_cold_start_rows": 0,
            "total_warm_start_rows": 0,
        }

    overall_scores = [float(summary["overall_score"]) for summary in result_summaries]
    cold_scores = [
        float(summary["cold_start_score"])
        for summary in result_summaries
        if summary["cold_start_score"] is not None
    ]
    warm_scores = [
        float(summary["warm_start_score"])
        for summary in result_summaries
        if summary["warm_start_score"] is not None
    ]

    overall_error_sum = float(
        sum(float(summary["overall_error_sum"]) for summary in result_summaries)
    )
    overall_denom_sum = float(
        sum(float(summary["overall_denom_sum"]) for summary in result_summaries)
    )
    cold_error_sum = float(
        sum(
            float(summary["cold_start_error_sum"])
            for summary in result_summaries
            if summary["cold_start_error_sum"] is not None
        )
    )
    cold_denom_sum = float(
        sum(
            float(summary["cold_start_denom_sum"])
            for summary in result_summaries
            if summary["cold_start_denom_sum"] is not None
        )
    )
    warm_error_sum = float(
        sum(
            float(summary["warm_start_error_sum"])
            for summary in result_summaries
            if summary["warm_start_error_sum"] is not None
        )
    )
    warm_denom_sum = float(
        sum(
            float(summary["warm_start_denom_sum"])
            for summary in result_summaries
            if summary["warm_start_denom_sum"] is not None
        )
    )

    validation_rows = [int(summary["cold_start_rows"]) + int(summary["warm_start_rows"]) for summary in result_summaries]
    cold_rows = [int(summary["cold_start_rows"]) for summary in result_summaries]
    warm_rows = [int(summary["warm_start_rows"]) for summary in result_summaries]

    return {
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


def _build_watchlist_mask(
    scored: pd.DataFrame,
    definition: dict[str, list[object]],
) -> pd.Series:
    return (
        scored["code"].isin(definition["codes"])
        & scored["sub_category"].isin(definition["sub_categories"])
        & scored["horizon"].isin(definition["horizons"])
    )


def _matches_fold_name(fold_name: str, fold_spec: str) -> bool:
    return fold_name == fold_spec or fold_name.startswith(f"{fold_spec}_")


def initialize_stress_accumulator(
    baseline_names: list[str],
) -> dict[str, dict[str, object]]:
    return {
        baseline_name: {
            "fold_sets": {name: [] for name in STRESS_FOLD_SETS},
            "per_horizon": {},
            "watchlist_slices": {name: [] for name in STRESS_WATCHLIST_SLICES},
        }
        for baseline_name in baseline_names
    }


def update_stress_accumulator(
    stress_accumulator: dict[str, dict[str, object]],
    *,
    fold_name: str,
    baseline_name: str,
    scored: pd.DataFrame,
    full_summary: dict[str, float | int | None] | None = None,
) -> None:
    if len(scored) == 0:
        return

    baseline_stress = stress_accumulator[baseline_name]
    baseline_full_summary = (
        summarize_scored_predictions(
            scored,
            cold_start_mask=scored["is_cold_start"],
        )
        if full_summary is None
        else full_summary
    )
    for stress_name, fold_names in STRESS_FOLD_SETS.items():
        if any(_matches_fold_name(fold_name, fold_spec) for fold_spec in fold_names):
            baseline_stress["fold_sets"][stress_name].append(baseline_full_summary)

    for horizon in sorted(scored["horizon"].dropna().unique().tolist()):
        horizon_key = str(int(horizon))
        horizon_scored = scored.loc[scored["horizon"] == horizon]
        if len(horizon_scored) == 0:
            continue
        baseline_stress["per_horizon"].setdefault(horizon_key, []).append(
            summarize_scored_predictions(
                horizon_scored,
                cold_start_mask=horizon_scored["is_cold_start"],
            )
        )

    for slice_name, definition in STRESS_WATCHLIST_SLICES.items():
        slice_mask = _build_watchlist_mask(scored, definition)
        if not slice_mask.any():
            continue
        watchlist_scored = scored.loc[slice_mask]
        baseline_stress["watchlist_slices"][slice_name].append(
            summarize_scored_predictions(
                watchlist_scored,
                cold_start_mask=watchlist_scored["is_cold_start"],
            )
        )


def finalize_stress_block(
    stress_accumulator: dict[str, dict[str, object]],
) -> dict[str, object]:
    stress: dict[str, object] = {
        "fold_sets": STRESS_FOLD_SETS,
        "watchlist_definitions": STRESS_WATCHLIST_SLICES,
        "baselines": {},
    }
    for baseline_name, baseline_accumulator in stress_accumulator.items():
        baseline_accumulator = stress_accumulator[baseline_name]
        stress["baselines"][baseline_name] = {
            "fold_sets": {
                name: aggregate_result_summaries(summaries)
                for name, summaries in baseline_accumulator["fold_sets"].items()
            },
            "per_horizon": {
                horizon: aggregate_result_summaries(summaries)
                for horizon, summaries in sorted(baseline_accumulator["per_horizon"].items())
            },
            "watchlist_slices": {
                name: aggregate_result_summaries(summaries)
                for name, summaries in baseline_accumulator["watchlist_slices"].items()
            },
        }
    return stress


def build_stress_block(
    fold_stress_inputs: list[dict[str, object]],
    *,
    baseline_names: list[str],
) -> dict[str, object]:
    stress_accumulator = initialize_stress_accumulator(baseline_names)
    for fold_input in fold_stress_inputs:
        fold_name = str(fold_input["name"])
        results = dict(fold_input["results"])
        for baseline_name in baseline_names:
            update_stress_accumulator(
                stress_accumulator,
                fold_name=fold_name,
                baseline_name=baseline_name,
                scored=results[baseline_name],
            )
    return finalize_stress_block(stress_accumulator)


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
        aggregate[baseline_name] = aggregate_result_summaries(
            [
                fold_summary["results"][baseline_name]
                for fold_summary in fold_summaries
            ]
        )

    best_name = max(
        aggregate,
        key=lambda name: float(aggregate[name]["aggregate_overall_score"]),
    )
    return aggregate, best_name


def write_scored_fold_frame(
    scored: pd.DataFrame,
    *,
    output_dir: Path,
    baseline_name: str,
    fold_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{baseline_name}_{fold_name}_scored.parquet"
    scored.to_parquet(output_path, index=False)
    return output_path


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
    parser.add_argument("--scored-output-dir", default=None)
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
    scored_output_dir = (
        Path(args.scored_output_dir)
        if args.scored_output_dir is not None
        else None
    )
    if scored_output_dir is not None:
        summary["scored_output_dir"] = str(scored_output_dir)

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

    max_train_ts_index = get_max_ts_index(args.train_path)
    if args.n_folds == 1:
        split_ts_index = max_train_ts_index - args.holdout_steps
        if split_ts_index <= 0:
            raise ValueError("holdout_steps is larger than the available train range")
        fold_specs = [
            {
                "name": f"holdout_{args.holdout_steps}",
                "train_end_ts": split_ts_index,
                "validation_start_ts": split_ts_index + 1,
                "validation_end_ts": max_train_ts_index,
            }
        ]
    else:
        fold_specs = []
        for fold in build_expanding_window_folds(
            max_ts_index=max_train_ts_index,
            holdout_steps=args.holdout_steps,
            n_folds=args.n_folds,
            step_size=args.step_size,
            min_train_steps=args.min_train_steps,
        ):
            fold_specs.append(
                {
                    "name": fold.name,
                    "train_end_ts": fold.train_end_ts,
                    "validation_start_ts": fold.validation_start_ts,
                    "validation_end_ts": fold.validation_end_ts,
                }
            )

    train_schema_columns = get_columns(args.train_path)
    required_train_columns: list[str] = []
    for name in selected_baselines:
        required_train_columns.extend(
            resolve_required_columns(
                BASELINE_CONFIGS[name],
                available_columns=train_schema_columns,
                include_target=True,
                include_weight=True,
            )
        )
    requested_extra_columns = [
        column
        for column in dict.fromkeys(required_train_columns)
        if column not in {*CONTEXT_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN}
    ]

    fold_summaries: list[dict[str, object]] = []
    stress_accumulator = initialize_stress_accumulator(selected_baselines)
    for fold in fold_specs:
        train_frame, validation_frame = load_train_validation_window(
            args.train_path,
            train_end_ts=int(fold["train_end_ts"]),
            validation_end_ts=int(fold["validation_end_ts"]),
            extra_columns=requested_extra_columns,
        )

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
        scored_template = validation_frame[
            [ID_COLUMN, TARGET_COLUMN, WEIGHT_COLUMN, "code", "sub_category", "horizon"]
        ].merge(
            validation_with_cold_mask[[ID_COLUMN, "is_cold_start"]],
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

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

            scored = scored_template.merge(
                predictions[[ID_COLUMN, "prediction"]],
                on=ID_COLUMN,
                how="left",
                validate="one_to_one",
            )
            if scored["prediction"].isna().any():
                raise ValueError(f"baseline {name} produced missing predictions on {fold['name']}")
            if scored_output_dir is not None:
                write_scored_fold_frame(
                    scored,
                    output_dir=scored_output_dir,
                    baseline_name=name,
                    fold_name=str(fold["name"]),
                )

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
            update_stress_accumulator(
                stress_accumulator,
                fold_name=str(fold["name"]),
                baseline_name=name,
                scored=scored[
                    [
                        "code",
                        "sub_category",
                        "horizon",
                        TARGET_COLUMN,
                        WEIGHT_COLUMN,
                        "prediction",
                        "is_cold_start",
                    ]
                ].copy(),
                full_summary=result_summary,
            )

        fold_summaries.append(fold_summary)

    summary["folds"] = fold_summaries
    aggregate, best_name = aggregate_fold_scores_for_names(fold_summaries, selected_baselines)
    summary["aggregate"] = aggregate
    summary["stress"] = finalize_stress_block(stress_accumulator)
    summary["best_baseline"] = best_name
    summary["best_score"] = aggregate[best_name]["aggregate_overall_score"]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
