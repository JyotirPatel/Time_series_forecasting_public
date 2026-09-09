from __future__ import annotations

import pandas as pd

from ts_forecasting.semantics import audit_competition_semantics


def test_audit_competition_semantics_reports_row_level_keys_and_cold_start_overlap(
    tmp_path,
) -> None:
    train_frame = pd.DataFrame(
        {
            "id": [
                "A__S1__C1__1__1",
                "A__S1__C1__1__2",
                "B__S2__C2__3__1",
            ],
            "code": ["A", "A", "B"],
            "sub_code": ["S1", "S1", "S2"],
            "sub_category": ["C1", "C1", "C2"],
            "horizon": [1, 1, 3],
            "ts_index": [1, 2, 1],
            "y_target": [1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    test_frame = pd.DataFrame(
        {
            "id": [
                "A__S1__C1__1__3",
                "A__S9__C1__1__3",
                "B__S2__C2__3__2",
            ],
            "code": ["A", "A", "B"],
            "sub_code": ["S1", "S9", "S2"],
            "sub_category": ["C1", "C1", "C2"],
            "horizon": [1, 1, 3],
            "ts_index": [3, 3, 2],
        }
    )

    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_frame.to_parquet(train_path, index=False)
    test_frame.to_parquet(test_path, index=False)

    summary = audit_competition_semantics(train_path, test_path)

    assert summary["train"]["rows"] == 3
    assert summary["train"]["duplicate_ids"] == 0
    assert summary["test"]["distinct_row_keys"] == 3
    assert summary["test"]["id_mismatch_rows"] == 0

    code_sub_code = summary["overlap"]["code_sub_code"]
    assert code_sub_code["test_unique_keys"] == 3
    assert code_sub_code["overlapping_test_keys"] == 2
    assert code_sub_code["unseen_test_keys"] == 1
    assert code_sub_code["unseen_test_rows"] == 1

    full_group = summary["overlap"]["full_group"]
    assert full_group["unseen_test_rows"] == 1
    assert any("cold-start" in item for item in summary["recommendations"])
