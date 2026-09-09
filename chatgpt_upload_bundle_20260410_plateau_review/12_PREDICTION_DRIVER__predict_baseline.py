#!/usr/bin/env python
"""Generate baseline predictions for the competition test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ts_forecasting.baseline_registry import BASELINE_CONFIGS, build_predictor
from ts_forecasting.constants import CONTEXT_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN
from ts_forecasting.io import read_parquet_frame
from ts_forecasting.sequential import IdentityTransformer, SequentialInferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.parquet")
    parser.add_argument("--test-path", default="data/test.parquet")
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINE_CONFIGS.keys()),
        default="smoothed_stable_group_mean_pw100",
    )
    parser.add_argument("--output", default="outputs/submissions/baseline_submission.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_frame = read_parquet_frame(
        args.train_path,
        columns=[*CONTEXT_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN],
    )
    test_frame = read_parquet_frame(
        args.test_path,
        columns=CONTEXT_COLUMNS,
    )

    config = BASELINE_CONFIGS[args.baseline]
    predictions = SequentialInferencePipeline(
        transformer=IdentityTransformer(),
        predictor=build_predictor(config),
    ).fit(train_frame).predict(test_frame)

    predictions[["id", "prediction"]].to_csv(output_path, index=False)
    print(f"Wrote {args.baseline} predictions to {output_path}")


if __name__ == "__main__":
    main()
