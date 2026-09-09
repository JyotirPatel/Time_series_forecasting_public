#!/usr/bin/env python
"""Generate a compact profile of the local parquet files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import duckdb

from ts_forecasting.constants import GROUP_COLUMNS, TIME_COLUMN, detect_feature_columns
from ts_forecasting.io import get_columns


def summarize_dataset(path: Path) -> dict[str, object]:
    columns = get_columns(path)
    feature_columns = detect_feature_columns(columns)
    null_expr = ", ".join(
        f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS {column}"
        for column in columns
    )

    with duckdb.connect() as connection:
        rows = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet(?)",
            [str(path.resolve())],
        ).fetchone()[0]
        ts_min, ts_max = connection.execute(
            f"SELECT MIN({TIME_COLUMN}), MAX({TIME_COLUMN}) FROM read_parquet(?)",
            [str(path.resolve())],
        ).fetchone()
        horizons = connection.execute(
            "SELECT DISTINCT horizon FROM read_parquet(?) ORDER BY horizon",
            [str(path.resolve())],
        ).fetchall()
        distinct_counts = connection.execute(
            f"""
            SELECT
                COUNT(DISTINCT code),
                COUNT(DISTINCT sub_code),
                COUNT(DISTINCT sub_category),
                COUNT(DISTINCT code || '|' || sub_code || '|' || sub_category)
            FROM read_parquet(?)
            """,
            [str(path.resolve())],
        ).fetchone()
        null_counts = connection.execute(
            f"SELECT {null_expr} FROM read_parquet(?)",
            [str(path.resolve())],
        ).fetchone()

    top_nulls = sorted(
        (
            {"column": column, "nulls": int(null_count)}
            for column, null_count in zip(columns, null_counts)
            if int(null_count) > 0
        ),
        key=lambda item: item["nulls"],
        reverse=True,
    )[:15]

    return {
        "path": str(path),
        "rows": int(rows),
        "columns": len(columns),
        "feature_columns": len(feature_columns),
        "group_columns": GROUP_COLUMNS,
        "time_range": {"min": int(ts_min), "max": int(ts_max)},
        "horizons": [int(row[0]) for row in horizons],
        "distinct_counts": {
            "code": int(distinct_counts[0]),
            "sub_code": int(distinct_counts[1]),
            "sub_category": int(distinct_counts[2]),
            "group_triplets": int(distinct_counts[3]),
        },
        "top_null_columns": top_nulls,
    }


def write_markdown(summary: dict[str, object], output_path: Path) -> None:
    lines = [
        f"# Profile: {summary['path']}",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Columns: `{summary['columns']}`",
        f"- Feature columns: `{summary['feature_columns']}`",
        f"- Horizons: `{summary['horizons']}`",
        (
            f"- Time range: `{summary['time_range']['min']}`"
            f" to `{summary['time_range']['max']}`"
        ),
        (
            "- Distinct groups: "
            f"`code={summary['distinct_counts']['code']}`, "
            f"`sub_code={summary['distinct_counts']['sub_code']}`, "
            f"`sub_category={summary['distinct_counts']['sub_category']}`, "
            f"`triplets={summary['distinct_counts']['group_triplets']}`"
        ),
        "",
        "## Top Null Columns",
        "",
        "| Column | Nulls |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {item['column']} | {item['nulls']} |"
        for item in summary["top_null_columns"]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--test-path", default="data/test.parquet")
    parser.add_argument("--output-dir", default="outputs/profiles")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary = summarize_dataset(Path(args.train_path))
    test_summary = summarize_dataset(Path(args.test_path))
    combined = {"train": train_summary, "test": test_summary}

    (output_dir / "data_profile.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )
    write_markdown(train_summary, output_dir / "train_profile.md")
    write_markdown(test_summary, output_dir / "test_profile.md")

    print(f"Wrote profile artifacts to {output_dir}")


if __name__ == "__main__":
    main()
