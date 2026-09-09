"""Sequential-safe inference helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

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

    supports_full_frame_prediction: ClassVar[bool] = True

    def fit(self, train_frame: pd.DataFrame) -> "IdentityTransformer":
        return self

    def transform_batch(self, batch_frame: pd.DataFrame) -> pd.DataFrame:
        return batch_frame.copy()


@dataclass
class FrozenMedianImputer:
    """Fill feature nulls using medians learned on the training frame only."""

    supports_full_frame_prediction: ClassVar[bool] = True
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
class _HistoryState:
    count: int = 0
    total: float = 0.0
    last: float | None = None


@dataclass
class CausalHistoryTransformer:
    """Add leakage-safe rolling history features from prior rows only."""

    supports_full_frame_prediction: ClassVar[bool] = False

    level_columns: list[str]
    source_columns: list[str] | None = None
    history_features: list[str] = field(default_factory=lambda: ["last", "mean", "delta"])
    banned_feature_columns: set[str] = field(default_factory=set)
    time_column: str = TIME_COLUMN
    source_columns_: list[str] | None = None
    history_state_: dict[str, dict[tuple[object, ...], _HistoryState]] | None = None

    def fit(self, train_frame: pd.DataFrame) -> "CausalHistoryTransformer":
        history_features = list(dict.fromkeys(self.history_features))
        unknown_features = set(history_features) - {"last", "mean", "delta"}
        if unknown_features:
            raise ValueError(f"unknown history features: {sorted(unknown_features)}")

        source_columns = self.source_columns or detect_feature_columns(list(train_frame.columns))
        self.source_columns_ = [
            column
            for column in source_columns
            if column in train_frame.columns
            and column not in self.banned_feature_columns
            and pd.api.types.is_numeric_dtype(train_frame[column])
        ]
        self.history_features = history_features
        self.history_state_ = {column: {} for column in self.source_columns_}
        return self

    def transform_batch(self, batch_frame: pd.DataFrame) -> pd.DataFrame:
        if self.source_columns_ is None or self.history_state_ is None:
            raise ValueError("CausalHistoryTransformer must be fit before transform_batch")

        transformed = batch_frame.copy()
        if transformed.empty or not self.source_columns_:
            return transformed

        required_columns = [self.time_column, *self.level_columns, *self.source_columns_]
        missing_columns = [column for column in required_columns if column not in transformed.columns]
        if missing_columns:
            raise ValueError(f"batch_frame is missing required columns: {missing_columns}")

        feature_columns: dict[str, dict[str, str]] = {}
        derived_arrays: dict[str, np.ndarray] = {}
        for source_column in self.source_columns_:
            feature_columns[source_column] = {}
            for history_feature in self.history_features:
                derived_column = f"{source_column}_history_{history_feature}"
                feature_columns[source_column][history_feature] = derived_column
                derived_arrays[derived_column] = np.full(len(transformed), np.nan, dtype=float)

        for _, time_batch in transformed.groupby(self.time_column, sort=True, observed=True):
            batch_index = time_batch.index
            batch_positions = transformed.index.get_indexer(batch_index)
            keys = list(time_batch[self.level_columns].itertuples(index=False, name=None))
            values_by_column = {
                column: pd.to_numeric(time_batch[column], errors="coerce").to_numpy(dtype=float)
                for column in self.source_columns_
            }

            for source_column in self.source_columns_:
                state_map = self.history_state_[source_column]
                last_values = np.full(len(time_batch), np.nan, dtype=float)
                mean_values = np.full(len(time_batch), np.nan, dtype=float)
                delta_values = np.full(len(time_batch), np.nan, dtype=float)

                for row_index, key in enumerate(keys):
                    state = state_map.get(key)
                    if state is None or state.count == 0:
                        continue
                    if "last" in self.history_features and state.last is not None:
                        last_values[row_index] = state.last
                    if "mean" in self.history_features:
                        mean_values[row_index] = state.total / state.count
                    if "delta" in self.history_features and state.last is not None:
                        value = values_by_column[source_column][row_index]
                        if not np.isnan(value):
                            delta_values[row_index] = value - state.last

                if "last" in self.history_features:
                    derived_arrays[feature_columns[source_column]["last"]][batch_positions] = last_values
                if "mean" in self.history_features:
                    derived_arrays[feature_columns[source_column]["mean"]][batch_positions] = mean_values
                if "delta" in self.history_features:
                    derived_arrays[feature_columns[source_column]["delta"]][batch_positions] = delta_values

                for row_index, key in enumerate(keys):
                    value = values_by_column[source_column][row_index]
                    if np.isnan(value):
                        continue
                    state = state_map.setdefault(key, _HistoryState())
                    state.count += 1
                    state.total += float(value)
                    state.last = float(value)

        derived_frame = pd.DataFrame(derived_arrays, index=transformed.index)
        return pd.concat([transformed, derived_frame], axis=1)


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

    def _supports_full_frame_prediction(self) -> bool:
        return bool(
            getattr(self.transformer, "supports_full_frame_prediction", False)
            and getattr(self.predictor, "supports_full_frame_prediction", False)
        )

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        already_sorted: bool = False,
    ) -> pd.DataFrame:
        ordered = (
            frame.reset_index(drop=True).copy()
            if already_sorted
            else frame.sort_values(
                [self.time_column, self.id_column],
                kind="stable",
            ).reset_index(drop=True)
        )

        if self._supports_full_frame_prediction():
            transformed = self.transformer.transform_batch(ordered)
            predictions = np.asarray(
                self.predictor.predict_batch(transformed),
                dtype=float,
            )
            if len(predictions) != len(ordered):
                raise ValueError("predict_batch must return one prediction per input row")

            output = ordered[[self.id_column, self.time_column]].copy()
            output["prediction"] = predictions
            return output

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
