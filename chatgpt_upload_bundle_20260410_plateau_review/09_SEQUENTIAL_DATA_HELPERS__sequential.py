"""Sequential-safe inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TIME_COLUMN, detect_feature_columns


class CausalBatchTransformer(Protocol):
    def fit(self, train_frame: pd.DataFrame) -> "CausalBatchTransformer":
        ...

    def transform_batch(self, batch_frame: pd.DataFrame) -> pd.DataFrame:
        ...


class BatchPredictor(Protocol):
    def fit(self, train_frame: pd.DataFrame) -> "BatchPredictor":
        ...

    def predict_batch(self, batch_frame: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class IdentityTransformer:
    """No-op transformer for leakage-safe baselines."""

    def fit(self, train_frame: pd.DataFrame) -> "IdentityTransformer":
        return self

    def transform_batch(self, batch_frame: pd.DataFrame) -> pd.DataFrame:
        return batch_frame.copy()


@dataclass
class FrozenMedianImputer:
    """Fill feature nulls using medians learned on the training frame only."""

    feature_columns: list[str] | None = None
    medians_: dict[str, float] | None = None

    def fit(self, train_frame: pd.DataFrame) -> "FrozenMedianImputer":
        feature_columns = self.feature_columns or detect_feature_columns(list(train_frame.columns))
        medians: dict[str, float] = {}
        for column in feature_columns:
            series = train_frame[column]
            if pd.api.types.is_numeric_dtype(series):
                medians[column] = float(series.median())
        self.feature_columns = feature_columns
        self.medians_ = medians
        return self

    def transform_batch(self, batch_frame: pd.DataFrame) -> pd.DataFrame:
        transformed = batch_frame.copy()
        if not self.medians_:
            return transformed

        for column, value in self.medians_.items():
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(value)
        return transformed


@dataclass
class SequentialInferencePipeline:
    """Predict strictly in non-decreasing ts_index order."""

    transformer: CausalBatchTransformer
    predictor: BatchPredictor
    time_column: str = TIME_COLUMN
    id_column: str = ID_COLUMN

    def fit(self, train_frame: pd.DataFrame) -> "SequentialInferencePipeline":
        transformed_train = self.transformer.fit(train_frame).transform_batch(train_frame)
        self.predictor.fit(transformed_train)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.sort_values(
            [self.time_column, self.id_column],
            kind="stable",
        ).reset_index(drop=True)

        prediction_parts: list[pd.DataFrame] = []
        for _, batch in ordered.groupby(self.time_column, sort=True, observed=True):
            transformed_batch = self.transformer.transform_batch(batch)
            batch_predictions = np.asarray(
                self.predictor.predict_batch(transformed_batch),
                dtype=float,
            )
            if len(batch_predictions) != len(batch):
                raise ValueError("predict_batch must return one prediction per input row")

            output = batch[[self.id_column, self.time_column]].copy()
            output["prediction"] = batch_predictions
            prediction_parts.append(output)

        return pd.concat(prediction_parts, ignore_index=True)
