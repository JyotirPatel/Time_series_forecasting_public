"""Data access helpers backed by DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .constants import CONTEXT_COLUMNS, TARGET_COLUMN, TIME_COLUMN, WEIGHT_COLUMN


def _path_str(path: str | Path) -> str:
    return str(Path(path).resolve())


def get_columns(path: str | Path) -> list[str]:
    query = "DESCRIBE SELECT * FROM read_parquet(?)"
    with duckdb.connect() as connection:
        rows = connection.execute(query, [_path_str(path)]).fetchall()
    return [row[0] for row in rows]


def get_max_ts_index(path: str | Path) -> int:
    query = f"SELECT MAX({TIME_COLUMN}) FROM read_parquet(?)"
    with duckdb.connect() as connection:
        value = connection.execute(query, [_path_str(path)]).fetchone()[0]
    return int(value)


def read_parquet_frame(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    where: str | None = None,
    order_by: str | None = None,
) -> pd.DataFrame:
    select_list = "*" if columns is None else ", ".join(columns)
    query = f"SELECT {select_list} FROM read_parquet(?)"
    if where:
        query += f" WHERE {where}"
    if order_by:
        query += f" ORDER BY {order_by}"

    with duckdb.connect() as connection:
        return connection.execute(query, [_path_str(path)]).df()


def load_train_validation_split(
    train_path: str | Path,
    *,
    holdout_steps: int,
    extra_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if holdout_steps <= 0:
        raise ValueError("holdout_steps must be positive")

    max_ts_index = get_max_ts_index(train_path)
    split_ts_index = max_ts_index - holdout_steps
    if split_ts_index <= 0:
        raise ValueError("holdout_steps is larger than the available train range")

    base_columns = [*CONTEXT_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN]
    if extra_columns:
        base_columns.extend(extra_columns)

    requested_columns = list(dict.fromkeys(base_columns))

    train_frame, validation_frame = load_train_validation_window(
        train_path,
        train_end_ts=split_ts_index,
        validation_end_ts=max_ts_index,
        extra_columns=extra_columns,
    )
    return train_frame, validation_frame, split_ts_index


def load_train_validation_window(
    train_path: str | Path,
    *,
    train_end_ts: int,
    validation_end_ts: int,
    train_start_ts: int | None = None,
    extra_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an expanding-train / bounded-validation slice from parquet."""

    if train_end_ts <= 0:
        raise ValueError("train_end_ts must be positive")
    if validation_end_ts <= train_end_ts:
        raise ValueError("validation_end_ts must be larger than train_end_ts")
    if train_start_ts is not None and train_start_ts > train_end_ts:
        raise ValueError("train_start_ts must be at or before train_end_ts")

    base_columns = [*CONTEXT_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN]
    if extra_columns:
        base_columns.extend(extra_columns)

    requested_columns = list(dict.fromkeys(base_columns))
    train_where = f"{TIME_COLUMN} <= {train_end_ts}"
    if train_start_ts is not None:
        train_where = f"{TIME_COLUMN} >= {train_start_ts} AND {train_where}"

    train_frame = read_parquet_frame(
        train_path,
        columns=requested_columns,
        where=train_where,
    )
    validation_frame = read_parquet_frame(
        train_path,
        columns=requested_columns,
        where=f"{TIME_COLUMN} > {train_end_ts} AND {TIME_COLUMN} <= {validation_end_ts}",
    )
    return train_frame, validation_frame
