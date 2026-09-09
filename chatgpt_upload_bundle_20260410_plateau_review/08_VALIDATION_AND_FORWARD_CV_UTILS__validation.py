"""Time-aware validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import TIME_COLUMN


@dataclass(frozen=True)
class ExpandingWindowFold:
    """A deterministic train-prefix / future-window validation fold."""

    name: str
    train_end_ts: int
    validation_start_ts: int
    validation_end_ts: int


def build_expanding_window_folds(
    *,
    max_ts_index: int,
    holdout_steps: int,
    n_folds: int = 1,
    step_size: int | None = None,
    min_train_steps: int = 1,
) -> list[ExpandingWindowFold]:
    """Build deterministic expanding-window folds ordered from oldest to newest."""

    if holdout_steps <= 0:
        raise ValueError("holdout_steps must be positive")
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if min_train_steps <= 0:
        raise ValueError("min_train_steps must be positive")

    stride = holdout_steps if step_size is None else step_size
    if stride <= 0:
        raise ValueError("step_size must be positive")

    latest_train_end = max_ts_index - holdout_steps
    if latest_train_end < min_train_steps:
        raise ValueError("holdout_steps leaves no valid training prefix")

    train_end_points = [
        latest_train_end - stride * offset
        for offset in reversed(range(n_folds))
    ]
    if train_end_points[0] < min_train_steps:
        raise ValueError("requested folds violate min_train_steps")

    folds: list[ExpandingWindowFold] = []
    for fold_index, train_end_ts in enumerate(train_end_points, start=1):
        validation_start_ts = train_end_ts + 1
        validation_end_ts = train_end_ts + holdout_steps
        if validation_end_ts > max_ts_index:
            raise ValueError("validation window extends past the available range")
        folds.append(
            ExpandingWindowFold(
                name=(
                    f"fold_{fold_index:02d}_train_{train_end_ts}"
                    f"_valid_{validation_start_ts}_{validation_end_ts}"
                ),
                train_end_ts=train_end_ts,
                validation_start_ts=validation_start_ts,
                validation_end_ts=validation_end_ts,
            )
        )
    return folds


def build_first_seen_lookup(
    frame: pd.DataFrame,
    *,
    key_columns: list[str],
    time_column: str = TIME_COLUMN,
) -> pd.DataFrame:
    """Return the first observed ts_index for each key."""

    required_columns = [*key_columns, time_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing required columns for first-seen lookup: {missing_columns}")

    return (
        frame[required_columns]
        .groupby(key_columns, dropna=False, observed=True)[time_column]
        .min()
        .reset_index(name="first_seen_ts")
    )


def annotate_cold_start_rows(
    frame: pd.DataFrame,
    *,
    first_seen_lookup: pd.DataFrame,
    key_columns: list[str],
    train_end_ts: int,
    output_column: str = "is_cold_start",
) -> pd.DataFrame:
    """Flag validation rows whose key was unseen in the training prefix."""

    annotated = frame.merge(
        first_seen_lookup,
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    if annotated["first_seen_ts"].isna().any():
        raise ValueError("cold-start annotation requires first-seen data for every row")

    annotated[output_column] = annotated["first_seen_ts"] > train_end_ts
    return annotated.drop(columns=["first_seen_ts"])


def summarize_key_coverage(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    key_columns: list[str],
) -> dict[str, object]:
    """Summarize how much of the validation slice is unseen at a given key level."""

    train_keys = train_frame[key_columns].drop_duplicates().assign(_seen_in_train=True)
    validation_keys = validation_frame[key_columns]
    merged = validation_keys.merge(
        train_keys,
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    unseen_mask = ~merged["_seen_in_train"].fillna(False).to_numpy(dtype=bool)
    unseen_rows = int(np.count_nonzero(unseen_mask))
    total_rows = int(len(validation_frame))

    return {
        "columns": key_columns,
        "train_unique_keys": int(len(train_keys)),
        "validation_unique_keys": int(len(validation_keys.drop_duplicates())),
        "unseen_validation_keys": int(len(merged.loc[unseen_mask, key_columns].drop_duplicates())),
        "unseen_validation_rows": unseen_rows,
        "total_validation_rows": total_rows,
        "unseen_validation_row_rate": (
            float(unseen_rows / total_rows) if total_rows else 0.0
        ),
    }
