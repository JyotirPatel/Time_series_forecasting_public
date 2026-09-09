"""Shared evaluation helpers for offline scoring and stress diagnostics."""

from __future__ import annotations

from statistics import mean, pstdev

import pandas as pd

from .constants import TARGET_COLUMN, WEIGHT_COLUMN
from .metrics import weighted_rmse_breakdown, weighted_rmse_score_from_sums

COLD_START_LEVELS: dict[str, list[str]] = {
    "code_sub_code": ["code", "sub_code"],
    "triplet": ["code", "sub_code", "sub_category"],
    "full_group": ["code", "sub_code", "sub_category", "horizon"],
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

    validation_rows = [
        int(summary["cold_start_rows"]) + int(summary["warm_start_rows"])
        for summary in result_summaries
    ]
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
