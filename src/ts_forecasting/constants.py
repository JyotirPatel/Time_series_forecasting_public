"""Shared constants for the forecasting project."""

from __future__ import annotations

ID_COLUMN = "id"
TIME_COLUMN = "ts_index"
TARGET_COLUMN = "y_target"
WEIGHT_COLUMN = "weight"

CODE_COLUMN = "code"
SUB_CODE_COLUMN = "sub_code"
SUB_CATEGORY_COLUMN = "sub_category"
HORIZON_COLUMN = "horizon"

GROUP_COLUMNS = [
    CODE_COLUMN,
    SUB_CODE_COLUMN,
    SUB_CATEGORY_COLUMN,
    HORIZON_COLUMN,
]

CONTEXT_COLUMNS = [
    ID_COLUMN,
    *GROUP_COLUMNS,
    TIME_COLUMN,
]

FEATURE_PREFIX = "feature_"


def detect_feature_columns(columns: list[str] | tuple[str, ...]) -> list[str]:
    """Return feature columns in stable order."""

    return [column for column in columns if column.startswith(FEATURE_PREFIX)]
