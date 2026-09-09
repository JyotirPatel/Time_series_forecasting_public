#!/usr/bin/env python
"""Audit row-level prediction semantics and train/test overlap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ts_forecasting.semantics import audit_competition_semantics


def write_markdown(summary: dict[str, object], output_path: Path) -> None:
    lines = [
        "# Competition Semantics Audit",
        "",
        "## Row-Level Keys",
        "",
        "| Split | Rows | Distinct ids | Distinct row keys | Duplicate ids | Duplicate row keys | ID mismatches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "test"):
        split_summary = summary[split_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    split_name,
                    f"{split_summary['rows']}",
                    f"{split_summary['distinct_ids']}",
                    f"{split_summary['distinct_row_keys']}",
                    f"{split_summary['duplicate_ids']}",
                    f"{split_summary['duplicate_row_keys']}",
                    f"{split_summary['id_mismatch_rows']}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Train/Test Overlap",
            "",
            "| Level | Columns | Train unique keys | Test unique keys | Overlapping test keys | Unseen test keys | Unseen test rows | Unseen test row rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for level_name, level_summary in summary["overlap"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    level_name,
                    ", ".join(level_summary["columns"]),
                    f"{level_summary['train_unique_keys']}",
                    f"{level_summary['test_unique_keys']}",
                    f"{level_summary['overlapping_test_keys']}",
                    f"{level_summary['unseen_test_keys']}",
                    f"{level_summary['unseen_test_rows']}",
                    f"{level_summary['unseen_test_row_rate']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    lines.extend(f"- {recommendation}" for recommendation in summary["recommendations"])
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--test-path", default="data/test.parquet")
    parser.add_argument("--output-dir", default="outputs/semantics")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = audit_competition_semantics(args.train_path, args.test_path)
    (output_dir / "semantics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, output_dir / "semantics_summary.md")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
