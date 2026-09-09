"""Competition semantics audit helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb

from .constants import ID_COLUMN, TIME_COLUMN


def _path_str(path: str | Path) -> str:
    return str(Path(path).resolve())


def _quoted_path(path: str | Path) -> str:
    return _path_str(path).replace("'", "''")


def _row_key_expression() -> str:
    parts = [
        "code",
        "sub_code",
        "sub_category",
        "CAST(horizon AS VARCHAR)",
        f"CAST({TIME_COLUMN} AS VARCHAR)",
    ]
    return " || '|' || ".join(parts)


def _id_expression() -> str:
    parts = [
        "code",
        "sub_code",
        "sub_category",
        "CAST(horizon AS VARCHAR)",
        f"CAST({TIME_COLUMN} AS VARCHAR)",
    ]
    return " || '__' || ".join(parts)


def dataset_key_audit(path: str | Path) -> dict[str, int]:
    parquet_path = _quoted_path(path)
    row_key_expression = _row_key_expression()
    id_expression = _id_expression()

    query = f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT {ID_COLUMN}) AS distinct_ids,
        COUNT(DISTINCT {row_key_expression}) AS distinct_row_keys,
        SUM(CASE WHEN {ID_COLUMN} != {id_expression} THEN 1 ELSE 0 END) AS id_mismatch_rows
    FROM read_parquet('{parquet_path}')
    """
    duplicate_id_query = f"""
    SELECT COUNT(*)
    FROM (
        SELECT {ID_COLUMN}
        FROM read_parquet('{parquet_path}')
        GROUP BY 1
        HAVING COUNT(*) > 1
    )
    """
    duplicate_row_key_query = f"""
    SELECT COUNT(*)
    FROM (
        SELECT code, sub_code, sub_category, horizon, {TIME_COLUMN}
        FROM read_parquet('{parquet_path}')
        GROUP BY 1, 2, 3, 4, 5
        HAVING COUNT(*) > 1
    )
    """

    with duckdb.connect() as connection:
        rows, distinct_ids, distinct_row_keys, id_mismatch_rows = connection.execute(query).fetchone()
        duplicate_ids = connection.execute(duplicate_id_query).fetchone()[0]
        duplicate_row_keys = connection.execute(duplicate_row_key_query).fetchone()[0]

    return {
        "rows": int(rows),
        "distinct_ids": int(distinct_ids),
        "distinct_row_keys": int(distinct_row_keys),
        "duplicate_ids": int(duplicate_ids),
        "duplicate_row_keys": int(duplicate_row_keys),
        "id_mismatch_rows": int(id_mismatch_rows),
    }


def split_overlap_audit(
    train_path: str | Path,
    test_path: str | Path,
    *,
    columns: list[str],
) -> dict[str, int | float | list[str]]:
    if not columns:
        raise ValueError("columns must not be empty")

    train_parquet_path = _quoted_path(train_path)
    test_parquet_path = _quoted_path(test_path)
    select_list = ", ".join(columns)
    join_condition = " AND ".join(f"t.{column} = tr.{column}" for column in columns)
    null_guard_column = columns[0]

    query = f"""
    WITH
    train_keys AS (
        SELECT DISTINCT {select_list}
        FROM read_parquet('{train_parquet_path}')
    ),
    test_keys AS (
        SELECT DISTINCT {select_list}
        FROM read_parquet('{test_parquet_path}')
    ),
    unseen_test_keys AS (
        SELECT {select_list}
        FROM test_keys
        EXCEPT
        SELECT {select_list}
        FROM train_keys
    ),
    overlapping_test_keys AS (
        SELECT {select_list}
        FROM test_keys
        INTERSECT
        SELECT {select_list}
        FROM train_keys
    ),
    unseen_test_rows AS (
        SELECT COUNT(*) AS row_count
        FROM read_parquet('{test_parquet_path}') AS t
        LEFT JOIN train_keys AS tr
          ON {join_condition}
        WHERE tr.{null_guard_column} IS NULL
    )
    SELECT
        (SELECT COUNT(*) FROM train_keys) AS train_unique_keys,
        (SELECT COUNT(*) FROM test_keys) AS test_unique_keys,
        (SELECT COUNT(*) FROM overlapping_test_keys) AS overlapping_test_keys,
        (SELECT COUNT(*) FROM unseen_test_keys) AS unseen_test_keys,
        (SELECT row_count FROM unseen_test_rows) AS unseen_test_rows,
        (SELECT COUNT(*) FROM read_parquet('{test_parquet_path}')) AS total_test_rows
    """

    with duckdb.connect() as connection:
        (
            train_unique_keys,
            test_unique_keys,
            overlapping_test_keys,
            unseen_test_keys,
            unseen_test_rows,
            total_test_rows,
        ) = connection.execute(query).fetchone()

    unseen_rate = 0.0
    if total_test_rows:
        unseen_rate = float(unseen_test_rows) / float(total_test_rows)

    return {
        "columns": columns,
        "train_unique_keys": int(train_unique_keys),
        "test_unique_keys": int(test_unique_keys),
        "overlapping_test_keys": int(overlapping_test_keys),
        "unseen_test_keys": int(unseen_test_keys),
        "unseen_test_rows": int(unseen_test_rows),
        "total_test_rows": int(total_test_rows),
        "unseen_test_row_rate": unseen_rate,
    }


def audit_competition_semantics(
    train_path: str | Path,
    test_path: str | Path,
) -> dict[str, object]:
    overlap_levels = {
        "code": ["code"],
        "code_sub_category_horizon": ["code", "sub_category", "horizon"],
        "code_sub_code": ["code", "sub_code"],
        "triplet": ["code", "sub_code", "sub_category"],
        "full_group": ["code", "sub_code", "sub_category", "horizon"],
    }
    overlap = {
        name: split_overlap_audit(train_path, test_path, columns=columns)
        for name, columns in overlap_levels.items()
    }

    recommendations = [
        "The prediction target is row-level: every train and test row has a unique id and a unique (code, sub_code, sub_category, horizon, ts_index) key.",
    ]
    if overlap["triplet"]["unseen_test_rows"] > 0:
        recommendations.append(
            "Test contains cold-start (code, sub_code, sub_category) groups, so forward holdout alone is not enough; add a cold-start-aware validation slice before trusting feature or group-history gains."
        )
    if overlap["full_group"]["unseen_test_rows"] > 0:
        recommendations.append(
            "Do not rely on memorizing full (code, sub_code, sub_category, horizon) histories: many test rows live on groups unseen in train."
        )
    if overlap["code_sub_category_horizon"]["unseen_test_rows"] == 0:
        recommendations.append(
            "A coarse fallback keyed by (code, sub_category, horizon) has full test coverage and is a defensible leakage-safe baseline level."
        )

    return {
        "train": dataset_key_audit(train_path),
        "test": dataset_key_audit(test_path),
        "overlap": overlap,
        "recommendations": recommendations,
    }
