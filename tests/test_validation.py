from __future__ import annotations

import pandas as pd

from ts_forecasting.validation import (
    annotate_cold_start_rows,
    build_expanding_window_folds,
    build_first_seen_lookup,
    summarize_key_coverage,
)


def test_build_expanding_window_folds_returns_oldest_to_newest_order() -> None:
    folds = build_expanding_window_folds(
        max_ts_index=3601,
        holdout_steps=360,
        n_folds=4,
        step_size=360,
        min_train_steps=2000,
    )

    assert [fold.train_end_ts for fold in folds] == [2161, 2521, 2881, 3241]
    assert folds[-1].validation_start_ts == 3242
    assert folds[-1].validation_end_ts == 3601


def test_annotate_cold_start_rows_marks_unseen_keys_for_a_fold() -> None:
    frame = pd.DataFrame(
        {
            "code": ["A", "A", "A", "A", "A"],
            "sub_code": ["s1", "s1", "s2", "s2", "s3"],
            "ts_index": [1, 2, 3, 4, 5],
        }
    )
    first_seen = build_first_seen_lookup(
        frame,
        key_columns=["code", "sub_code"],
    )
    validation = pd.DataFrame(
        {
            "code": ["A", "A"],
            "sub_code": ["s1", "s3"],
        }
    )

    annotated = annotate_cold_start_rows(
        validation,
        first_seen_lookup=first_seen,
        key_columns=["code", "sub_code"],
        train_end_ts=4,
    )

    assert annotated["is_cold_start"].tolist() == [False, True]


def test_summarize_key_coverage_reports_unseen_validation_rows() -> None:
    train_frame = pd.DataFrame(
        {
            "code": ["A", "A", "B"],
            "sub_code": ["s1", "s2", "s3"],
        }
    )
    validation_frame = pd.DataFrame(
        {
            "code": ["A", "B", "B"],
            "sub_code": ["s1", "s3", "s4"],
        }
    )

    summary = summarize_key_coverage(
        train_frame,
        validation_frame,
        key_columns=["code", "sub_code"],
    )

    assert summary["unseen_validation_rows"] == 1
    assert summary["unseen_validation_keys"] == 1
    assert summary["unseen_validation_row_rate"] == 1 / 3
