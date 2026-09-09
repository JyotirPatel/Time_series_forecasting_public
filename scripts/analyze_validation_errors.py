#!/usr/bin/env python
"""Analyze validation predictions from the saved multi-fold baseline runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

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
from ts_forecasting.io import read_parquet_frame
from ts_forecasting.metrics import weighted_rmse_breakdown
from ts_forecasting.validation import annotate_cold_start_rows, build_first_seen_lookup


COLD_START_LEVELS: dict[str, list[str]] = {
    "code_sub_code": [CODE_COLUMN, SUB_CODE_COLUMN],
    "triplet": [CODE_COLUMN, SUB_CODE_COLUMN, SUB_CATEGORY_COLUMN],
    "full_group": [CODE_COLUMN, SUB_CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN],
    "code_sub_category_horizon": [CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN],
}


def _safe_breakdown(frame: pd.DataFrame) -> dict[str, float | None]:
    if len(frame) == 0:
        return {
            "score": None,
            "error_sum": None,
            "energy_denominator": None,
            "ratio": None,
            "clipped_ratio": None,
        }

    try:
        breakdown = weighted_rmse_breakdown(
            frame[TARGET_COLUMN].to_numpy(),
            frame["prediction"].to_numpy(),
            frame[WEIGHT_COLUMN].to_numpy(),
        )
    except ValueError:
        return {
            "score": None,
            "error_sum": None,
            "energy_denominator": None,
            "ratio": None,
            "clipped_ratio": None,
        }
    return {
        "score": breakdown.score,
        "error_sum": breakdown.error_sum,
        "energy_denominator": breakdown.denom_sum,
        "ratio": breakdown.ratio,
        "clipped_ratio": breakdown.clipped_ratio,
    }


def _fmt(value: object, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _slice_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    cold_frame = frame.loc[frame["is_cold_start"]]
    warm_frame = frame.loc[~frame["is_cold_start"]]

    summary: dict[str, float | int | None] = {
        "rows": int(len(frame)),
        "cold_start_rows": int(len(cold_frame)),
        "warm_start_rows": int(len(warm_frame)),
    }
    summary.update(_safe_breakdown(frame))
    for prefix, sub_frame in [
        ("cold_start", cold_frame),
        ("warm_start", warm_frame),
    ]:
        breakdown = _safe_breakdown(sub_frame)
        for key, value in breakdown.items():
            summary[f"{prefix}_{key}"] = value
    return summary


def _summarize_groups(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_key, group_frame in frame.groupby(group_columns, dropna=False, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_columns, group_key)}
        row.update(_slice_summary(group_frame))
        rows.append(row)

    rows.sort(
        key=lambda row: (
            float("inf") if row["score"] is None else float(row["score"]),
            -int(row["rows"]),
        )
    )
    return rows


def _build_fold_comparison(
    *,
    fold_outputs: list[dict[str, object]],
    fold_a_prefix: str,
    fold_b_prefix: str,
    rows_key: str,
    group_columns: list[str],
) -> dict[str, object] | None:
    fold_a = next((fold for fold in fold_outputs if str(fold["name"]).startswith(fold_a_prefix)), None)
    fold_b = next((fold for fold in fold_outputs if str(fold["name"]).startswith(fold_b_prefix)), None)
    if fold_a is None or fold_b is None:
        return None

    frame_a = pd.DataFrame(fold_a[rows_key])
    frame_b = pd.DataFrame(fold_b[rows_key])
    if frame_a.empty or frame_b.empty:
        return None

    compare_columns = group_columns + [
        "rows",
        "score",
        "energy_denominator",
        "cold_start_rows",
        "cold_start_score",
        "cold_start_energy_denominator",
        "warm_start_rows",
        "warm_start_score",
        "warm_start_energy_denominator",
    ]
    merged = frame_a[compare_columns].merge(
        frame_b[compare_columns],
        on=group_columns,
        how="outer",
        suffixes=("_fold_02", "_fold_03"),
    )
    merged["mean_score"] = merged[["score_fold_02", "score_fold_03"]].mean(axis=1)
    merged["score_delta_fold_03_minus_fold_02"] = (
        merged["score_fold_03"] - merged["score_fold_02"]
    )
    merged["combined_energy_denominator"] = (
        merged["energy_denominator_fold_02"].fillna(0.0)
        + merged["energy_denominator_fold_03"].fillna(0.0)
    )
    merged = merged.sort_values(
        by=["mean_score", "combined_energy_denominator"],
        ascending=[True, False],
        kind="stable",
    )
    return {
        "fold_a": str(fold_a["name"]),
        "fold_b": str(fold_b["name"]),
        "rows_key": rows_key,
        "group_columns": group_columns,
        "rows": merged.to_dict(orient="records"),
    }


def _build_markdown_report(
    *,
    baseline_name: str,
    summary_data: dict[str, object],
    top_k: int,
) -> str:
    lines = [
        f"# Validation Error Analysis: {baseline_name}",
        "",
        "## Aggregate",
        "",
        f"- Baseline: `{baseline_name}`",
        f"- Canonical exact concatenated-fold score: `{_fmt(summary_data['aggregate_overall_score'])}`",
        f"- Aggregate cold-start score: `{_fmt(summary_data['aggregate_cold_start_score'])}`",
        f"- Aggregate warm-start score: `{_fmt(summary_data['aggregate_warm_start_score'])}`",
        f"- Aggregate energy denominator: `{_fmt(summary_data['aggregate_energy_denominator'], 3)}`",
        "",
        "## Fold Summary",
        "",
        "| Fold | Overall | Cold | Warm | Cold Rate | Energy Denom |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fold in summary_data["folds"]:
        lines.append(
            "| {name} | {overall} | {cold} | {warm} | {rate:.2%} | {denom} |".format(
                name=fold["name"],
                overall=_fmt(fold["overall_score"]),
                cold=_fmt(fold["cold_start_score"]),
                warm=_fmt(fold["warm_start_score"]),
                rate=fold["cold_start_rate"],
                denom=_fmt(fold["energy_denominator"], 3),
            )
        )

    lines.extend(
        [
            "",
            "## Worst Horizon Slices",
            "",
            "| Fold | Horizon | Score | Cold | Warm | Energy Denom | Rows |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for fold in summary_data["folds"]:
        for row in fold["by_horizon"][:top_k]:
            lines.append(
                f"| {fold['name']} | {row[HORIZON_COLUMN]} | {_fmt(row['score'])} | {_fmt(row['cold_start_score'])} | {_fmt(row['warm_start_score'])} | {_fmt(row['energy_denominator'], 3)} | {row['rows']} |"
            )

    lines.extend(
        [
            "",
            "## Worst Code / Sub-Category / Horizon Slices",
            "",
            "| Fold | Code | Sub-Category | Horizon | Score | Cold | Warm | Energy Denom | Rows |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for fold in summary_data["folds"]:
        for row in fold["by_code_sub_category_horizon"][:top_k]:
            lines.append(
                f"| {fold['name']} | {row[CODE_COLUMN]} | {row[SUB_CATEGORY_COLUMN]} | {row[HORIZON_COLUMN]} | {_fmt(row['score'])} | {_fmt(row['cold_start_score'])} | {_fmt(row['warm_start_score'])} | {_fmt(row['energy_denominator'], 3)} | {row['rows']} |"
            )

    fold_comparison = summary_data.get("fold_02_vs_fold_03_code_sub_category_horizon")
    if fold_comparison and fold_comparison["rows"]:
        lines.extend(
            [
                "",
                "## Fold 02 vs Fold 03",
                "",
                "| Code | Sub-Category | Horizon | Fold 02 | Fold 03 | Cold 02 | Cold 03 | Warm 02 | Warm 03 | Denom 02 | Denom 03 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in fold_comparison["rows"][:top_k]:
            lines.append(
                f"| {row[CODE_COLUMN]} | {row[SUB_CATEGORY_COLUMN]} | {row[HORIZON_COLUMN]} | {_fmt(row['score_fold_02'])} | {_fmt(row['score_fold_03'])} | {_fmt(row['cold_start_score_fold_02'])} | {_fmt(row['cold_start_score_fold_03'])} | {_fmt(row['warm_start_score_fold_02'])} | {_fmt(row['warm_start_score_fold_03'])} | {_fmt(row['energy_denominator_fold_02'], 3)} | {_fmt(row['energy_denominator_fold_03'], 3)} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-path",
        default="outputs/baselines/forward_cv_h360_f4_s360_code_sub_code/summary.json",
    )
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--compare-fold-a", default="fold_02")
    parser.add_argument("--compare-fold-b", default="fold_03")
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    summary = json.loads(summary_path.read_text())
    baseline_name = args.baseline or summary["best_baseline"]
    evaluation_dir = summary_path.parent
    output_dir = Path(args.output_dir) / evaluation_dir.name / baseline_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cold_start_columns = COLD_START_LEVELS[summary["cold_start_level"]]
    first_seen_frame = read_parquet_frame(
        args.train_path,
        columns=[*cold_start_columns, TIME_COLUMN],
    )
    first_seen_lookup = build_first_seen_lookup(
        first_seen_frame,
        key_columns=cold_start_columns,
    )

    aggregate_rows: list[pd.DataFrame] = []
    fold_outputs: list[dict[str, object]] = []
    for fold in summary["folds"]:
        validation_frame = read_parquet_frame(
            args.train_path,
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
                f"{TIME_COLUMN} > {fold['train_end_ts']} "
                f"AND {TIME_COLUMN} <= {fold['validation_end_ts']}"
            ),
        )
        predictions = pd.read_csv(
            evaluation_dir / f"{fold['name']}_{baseline_name}_predictions.csv"
        )
        scored = validation_frame.merge(
            predictions,
            on=[ID_COLUMN, TIME_COLUMN],
            how="left",
            validate="one_to_one",
        )
        if scored["prediction"].isna().any():
            raise ValueError(f"missing predictions for {fold['name']} / {baseline_name}")

        scored = annotate_cold_start_rows(
            scored,
            first_seen_lookup=first_seen_lookup,
            key_columns=cold_start_columns,
            train_end_ts=int(fold["train_end_ts"]),
        )
        aggregate_rows.append(scored)

        cold_frame = scored.loc[scored["is_cold_start"]]
        warm_frame = scored.loc[~scored["is_cold_start"]]
        fold_outputs.append(
            {
                "name": fold["name"],
                **_slice_summary(scored),
                "overall_score": _safe_breakdown(scored)["score"],
                "cold_start_score": _safe_breakdown(cold_frame)["score"],
                "warm_start_score": _safe_breakdown(warm_frame)["score"],
                "cold_start_rate": float(scored["is_cold_start"].mean()),
                "by_horizon": _summarize_groups(scored, group_columns=[HORIZON_COLUMN]),
                "by_sub_category": _summarize_groups(
                    scored,
                    group_columns=[SUB_CATEGORY_COLUMN],
                ),
                "by_sub_category_horizon": _summarize_groups(
                    scored,
                    group_columns=[SUB_CATEGORY_COLUMN, HORIZON_COLUMN],
                ),
                "by_code_sub_category_horizon": _summarize_groups(
                    scored,
                    group_columns=[CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN],
                ),
            }
        )

    aggregate_frame = pd.concat(aggregate_rows, ignore_index=True)
    aggregate_slice = _slice_summary(aggregate_frame)
    aggregate_summary = {
        "baseline": baseline_name,
        "summary_path": str(summary_path),
        "aggregate_overall_score": aggregate_slice["score"],
        "aggregate_cold_start_score": aggregate_slice["cold_start_score"],
        "aggregate_warm_start_score": aggregate_slice["warm_start_score"],
        "aggregate_error_sum": aggregate_slice["error_sum"],
        "aggregate_energy_denominator": aggregate_slice["energy_denominator"],
        "aggregate_ratio": aggregate_slice["ratio"],
        "by_horizon": _summarize_groups(aggregate_frame, group_columns=[HORIZON_COLUMN]),
        "by_sub_category": _summarize_groups(
            aggregate_frame,
            group_columns=[SUB_CATEGORY_COLUMN],
        ),
        "by_sub_category_horizon": _summarize_groups(
            aggregate_frame,
            group_columns=[SUB_CATEGORY_COLUMN, HORIZON_COLUMN],
        ),
        "by_code_sub_category_horizon": _summarize_groups(
            aggregate_frame,
            group_columns=[CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN],
        ),
        "folds": fold_outputs,
    }
    aggregate_summary["fold_02_vs_fold_03_code_sub_category_horizon"] = _build_fold_comparison(
        fold_outputs=fold_outputs,
        fold_a_prefix=args.compare_fold_a,
        fold_b_prefix=args.compare_fold_b,
        rows_key="by_code_sub_category_horizon",
        group_columns=[CODE_COLUMN, SUB_CATEGORY_COLUMN, HORIZON_COLUMN],
    )

    (output_dir / "summary.json").write_text(json.dumps(aggregate_summary, indent=2) + "\n")
    (output_dir / "summary.md").write_text(
        _build_markdown_report(
            baseline_name=baseline_name,
            summary_data=aggregate_summary,
            top_k=args.top_k,
        )
    )
    print(json.dumps(aggregate_summary, indent=2))


if __name__ == "__main__":
    main()
