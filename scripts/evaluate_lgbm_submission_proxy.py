#!/usr/bin/env python
"""Evaluate the checked-in LGBM submission path on a pseudo-public tail holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from scripts import predict_lgbm
from ts_forecasting.baseline_registry import BASELINE_CONFIGS
from ts_forecasting.constants import (
    CODE_COLUMN,
    HORIZON_COLUMN,
    ID_COLUMN,
    SUB_CATEGORY_COLUMN,
    SUB_CODE_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    WEIGHT_COLUMN,
    detect_feature_columns,
)
from ts_forecasting.evaluation import summarize_scored_predictions
from ts_forecasting.io import get_columns, read_parquet_frame
from ts_forecasting.metrics import weighted_rmse_breakdown
from ts_forecasting.validation import (
    annotate_cold_start_rows,
    build_first_seen_lookup,
    summarize_key_coverage,
)

COLD_START_KEY = [CODE_COLUMN, SUB_CODE_COLUMN]
COMPONENT_EXPORT_COLUMNS = [
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


def split_tail_holdout(
    train_frame: pd.DataFrame,
    *,
    pseudo_test_steps: int,
    min_train_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pseudo_test_steps <= 0:
        raise ValueError("pseudo_test_steps must be positive")
    if min_train_steps <= 0:
        raise ValueError("min_train_steps must be positive")

    ordered = train_frame.sort_values([TIME_COLUMN, ID_COLUMN], kind="stable").reset_index(
        drop=True
    )
    max_ts = int(ordered[TIME_COLUMN].max())
    holdout_start_ts = max_ts - int(pseudo_test_steps) + 1
    prefix_end_ts = holdout_start_ts - 1
    if prefix_end_ts < min_train_steps:
        raise ValueError("pseudo_test_steps leaves too little prefix history")

    prefix_frame = ordered.loc[ordered[TIME_COLUMN] <= prefix_end_ts].copy()
    holdout_frame = ordered.loc[ordered[TIME_COLUMN] >= holdout_start_ts].copy()
    if len(prefix_frame) == 0 or len(holdout_frame) == 0:
        raise ValueError("tail split produced an empty prefix or holdout frame")
    return prefix_frame, holdout_frame


def _allocate_stratified_sample_sizes(
    counts: pd.Series,
    *,
    sample_size: int,
    fraction: float,
) -> dict[object, int]:
    allocations = {
        horizon: min(int(np.floor(count * fraction)), int(count))
        for horizon, count in counts.items()
    }
    total = sum(allocations.values())

    if total == 0 and sample_size > 0:
        largest_horizon = counts.sort_values(ascending=False).index[0]
        allocations[largest_horizon] = 1
        total = 1

    remainders = {
        horizon: float(count * fraction - np.floor(count * fraction))
        for horizon, count in counts.items()
    }

    while total < sample_size:
        candidates = [
            horizon
            for horizon, count in counts.items()
            if allocations[horizon] < int(count)
        ]
        if not candidates:
            break
        best_horizon = max(
            candidates,
            key=lambda horizon: (remainders[horizon], counts[horizon], str(horizon)),
        )
        allocations[best_horizon] += 1
        total += 1

    while total > sample_size:
        candidates = [horizon for horizon, size in allocations.items() if size > 0]
        if not candidates:
            break
        worst_horizon = min(
            candidates,
            key=lambda horizon: (remainders[horizon], counts[horizon], str(horizon)),
        )
        allocations[worst_horizon] -= 1
        total -= 1

    return allocations


def _sample_public_subset(
    scored_frame: pd.DataFrame,
    *,
    public_fraction: float,
    stratify_by_horizon: bool,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    if not 0.0 < public_fraction <= 1.0:
        raise ValueError("public_fraction must be in (0, 1]")

    total_rows = int(len(scored_frame))
    sample_size = max(1, min(total_rows, int(round(total_rows * public_fraction))))
    if sample_size == total_rows:
        return scored_frame.copy()

    if not stratify_by_horizon:
        chosen = rng.choice(total_rows, size=sample_size, replace=False)
        return scored_frame.iloc[np.sort(chosen)].reset_index(drop=True)

    counts = (
        scored_frame.groupby(HORIZON_COLUMN, dropna=False, observed=True)
        .size()
        .sort_index()
    )
    allocations = _allocate_stratified_sample_sizes(
        counts,
        sample_size=sample_size,
        fraction=public_fraction,
    )
    parts: list[pd.DataFrame] = []
    for horizon, size in allocations.items():
        if size <= 0:
            continue
        group = scored_frame.loc[scored_frame[HORIZON_COLUMN] == horizon]
        chosen = rng.choice(len(group), size=size, replace=False)
        parts.append(group.iloc[np.sort(chosen)])
    sort_columns = [
        column for column in (TIME_COLUMN, ID_COLUMN) if column in scored_frame.columns
    ]
    sampled = pd.concat(parts, ignore_index=False)
    if sort_columns:
        sampled = sampled.sort_values(sort_columns, kind="stable")
    return (
        sampled
        .reset_index(drop=True)
    )


def bootstrap_public_scores(
    scored_frame: pd.DataFrame,
    *,
    public_fraction: float,
    n_bootstraps: int,
    stratify_by_horizon: bool,
    random_state: int,
) -> dict[str, float | int]:
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive")

    rng = np.random.RandomState(random_state)
    scores: list[float] = []
    sample_rows: list[int] = []
    for _ in range(n_bootstraps):
        sample = _sample_public_subset(
            scored_frame,
            public_fraction=public_fraction,
            stratify_by_horizon=stratify_by_horizon,
            rng=rng,
        )
        breakdown = weighted_rmse_breakdown(
            sample[TARGET_COLUMN].to_numpy(dtype=float),
            sample["prediction"].to_numpy(dtype=float),
            sample[WEIGHT_COLUMN].to_numpy(dtype=float),
        )
        scores.append(float(breakdown.score))
        sample_rows.append(int(len(sample)))

    score_array = np.asarray(scores, dtype=float)
    return {
        "public_fraction": float(public_fraction),
        "n_bootstraps": int(n_bootstraps),
        "stratify_by_horizon": bool(stratify_by_horizon),
        "mean_score": float(score_array.mean()),
        "std_score": float(pstdev(scores)) if len(scores) > 1 else 0.0,
        "min_score": float(score_array.min()),
        "p05_score": float(np.percentile(score_array, 5.0)),
        "p50_score": float(np.percentile(score_array, 50.0)),
        "p95_score": float(np.percentile(score_array, 95.0)),
        "max_score": float(score_array.max()),
        "sample_rows_min": int(min(sample_rows)),
        "sample_rows_max": int(max(sample_rows)),
        "sample_rows_mean": float(mean(sample_rows)),
    }


def export_holdout_components(
    scored_frame: pd.DataFrame,
    *,
    output_path: Path,
) -> Path:
    export_frame = scored_frame.loc[:, COMPONENT_EXPORT_COLUMNS].copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_frame.to_parquet(output_path, index=False)
    return output_path


def evaluate_tail_submission_proxy(
    *,
    train_frame: pd.DataFrame,
    incumbent_config: dict[str, object],
    residual_horizons: list[int],
    lgbm_params: dict[str, object],
    pseudo_test_steps: int,
    min_train_steps: int,
    early_stopping_rounds: int,
    calibration_holdout_steps: int,
    calibration_min_train_steps: int,
    calibration_max_n_estimators: int,
    n_estimators: int | None,
    use_sample_weight: bool,
    apply_residual_only_warm: bool,
    residual_warm_only_horizons: list[int] | None,
    public_fraction: float,
    n_bootstraps: int,
    random_state: int,
    components_output: Path | None = None,
) -> dict[str, object]:
    prefix_frame, holdout_frame = split_tail_holdout(
        train_frame,
        pseudo_test_steps=pseudo_test_steps,
        min_train_steps=min_train_steps,
    )

    calibrated_best_iteration: int | None = None
    calibration_metadata: dict[str, float] | None = None
    if n_estimators is None:
        calibrated_best_iteration, calibration_metadata = (
            predict_lgbm.calibrate_residual_lgbm_n_estimators(
                train_frame=prefix_frame,
                incumbent_config=incumbent_config,
                residual_horizons=residual_horizons,
                lgbm_params=lgbm_params,
                max_n_estimators=calibration_max_n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                calibration_holdout_steps=calibration_holdout_steps,
                calibration_min_train_steps=calibration_min_train_steps,
                use_sample_weight=use_sample_weight,
                apply_residual_only_warm=apply_residual_only_warm,
                residual_warm_only_horizons=residual_warm_only_horizons,
            )
        )

    full_train_n_estimators = predict_lgbm.resolve_full_train_n_estimators(
        n_estimators,
        calibrated_best_iteration,
    )
    predictions = predict_lgbm.predict_residual_lgbm(
        train_frame=prefix_frame,
        predict_frame=holdout_frame,
        incumbent_config=incumbent_config,
        residual_horizons=residual_horizons,
        lgbm_params=lgbm_params,
        n_estimators=full_train_n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        use_sample_weight=use_sample_weight,
        apply_residual_only_warm=apply_residual_only_warm,
        residual_warm_only_horizons=residual_warm_only_horizons,
    )

    scored = holdout_frame[
        [
            ID_COLUMN,
            CODE_COLUMN,
            SUB_CODE_COLUMN,
            SUB_CATEGORY_COLUMN,
            HORIZON_COLUMN,
            TIME_COLUMN,
            TARGET_COLUMN,
            WEIGHT_COLUMN,
        ]
    ].merge(
        predictions,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if scored["prediction"].isna().any():
        raise ValueError("submission proxy predictions were missing holdout rows")

    first_seen_lookup = build_first_seen_lookup(
        train_frame[[TIME_COLUMN, CODE_COLUMN, SUB_CODE_COLUMN]],
        key_columns=COLD_START_KEY,
    )
    scored = annotate_cold_start_rows(
        scored,
        first_seen_lookup=first_seen_lookup,
        key_columns=COLD_START_KEY,
        train_end_ts=int(prefix_frame[TIME_COLUMN].max()),
    )
    components_path: Path | None = None
    if components_output is not None:
        components_path = export_holdout_components(
            scored,
            output_path=Path(components_output),
        )
    holdout_summary = summarize_scored_predictions(
        scored,
        cold_start_mask=scored["is_cold_start"],
    )

    coverage = {
        "code_sub_code": summarize_key_coverage(
            prefix_frame,
            holdout_frame,
            key_columns=COLD_START_KEY,
        ),
        "code_sub_category_horizon": summarize_key_coverage(
            prefix_frame,
            holdout_frame,
            key_columns=[CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN],
        ),
    }
    random_public_bootstrap = bootstrap_public_scores(
        scored,
        public_fraction=public_fraction,
        n_bootstraps=n_bootstraps,
        stratify_by_horizon=False,
        random_state=random_state,
    )
    stratified_public_bootstrap = bootstrap_public_scores(
        scored,
        public_fraction=public_fraction,
        n_bootstraps=n_bootstraps,
        stratify_by_horizon=True,
        random_state=random_state,
    )

    summary = {
        "split": {
            "prefix_end_ts": int(prefix_frame[TIME_COLUMN].max()),
            "holdout_start_ts": int(holdout_frame[TIME_COLUMN].min()),
            "holdout_end_ts": int(holdout_frame[TIME_COLUMN].max()),
            "prefix_rows": int(len(prefix_frame)),
            "holdout_rows": int(len(holdout_frame)),
        },
        "coverage": coverage,
        "holdout": holdout_summary,
        "random_public_bootstrap": random_public_bootstrap,
        "stratified_public_bootstrap": stratified_public_bootstrap,
        "residual_horizons": list(residual_horizons),
        "params": dict(lgbm_params),
        "weighted": bool(use_sample_weight),
        "apply_residual_only_warm": bool(apply_residual_only_warm),
        "residual_warm_only_horizons": residual_warm_only_horizons,
        "full_train_n_estimators": int(full_train_n_estimators),
        "calibration": calibration_metadata,
    }
    if components_path is not None:
        summary["components_output"] = str(components_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--components-output", default=None)
    parser.add_argument("--pseudo-test-steps", type=int, default=775)
    parser.add_argument("--min-train-steps", type=int, default=2000)
    parser.add_argument("--public-fraction", type=float, default=0.24)
    parser.add_argument("--n-bootstraps", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--apply-residual-only-warm", action="store_true")
    parser.add_argument("--residual-warm-only-horizons", type=int, nargs="+", default=None)
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
    args = parser.parse_args()

    all_columns = get_columns(args.train_path)
    feature_cols = [
        column for column in detect_feature_columns(all_columns) if column != "feature_al"
    ]
    train_cols = [
        ID_COLUMN,
        CODE_COLUMN,
        SUB_CODE_COLUMN,
        SUB_CATEGORY_COLUMN,
        HORIZON_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
        WEIGHT_COLUMN,
        *feature_cols,
    ]
    train_frame = read_parquet_frame(args.train_path, columns=train_cols)
    incumbent_config = BASELINE_CONFIGS[predict_lgbm.INCUMBENT_CONFIG_NAME]
    residual_horizons = predict_lgbm.normalize_residual_horizons(args.residual_horizons)
    residual_warm_only_horizons = predict_lgbm.normalize_residual_warm_only_horizons(
        args.residual_warm_only_horizons,
        residual_horizons=residual_horizons,
        apply_residual_only_warm=args.apply_residual_only_warm,
    )
    lgbm_params = predict_lgbm.build_lgbm_params(args)

    summary = evaluate_tail_submission_proxy(
        train_frame=train_frame,
        incumbent_config=incumbent_config,
        residual_horizons=residual_horizons,
        lgbm_params=lgbm_params,
        pseudo_test_steps=int(args.pseudo_test_steps),
        min_train_steps=int(args.min_train_steps),
        early_stopping_rounds=int(args.early_stopping_rounds),
        calibration_holdout_steps=int(args.calibration_holdout_steps),
        calibration_min_train_steps=int(args.calibration_min_train_steps),
        calibration_max_n_estimators=int(args.calibration_max_n_estimators),
        n_estimators=args.n_estimators,
        use_sample_weight=not args.unweighted,
        apply_residual_only_warm=args.apply_residual_only_warm,
        residual_warm_only_horizons=residual_warm_only_horizons,
        public_fraction=float(args.public_fraction),
        n_bootstraps=int(args.n_bootstraps),
        random_state=int(args.random_state),
        components_output=Path(args.components_output) if args.components_output else None,
    )
    summary["incumbent"] = predict_lgbm.INCUMBENT_CONFIG_NAME

    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else (
            ROOT
            / "outputs"
            / "lgbm_submission_proxy"
            / "summary.json"
        )
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Summary: {summary_output}")


if __name__ == "__main__":
    main()
