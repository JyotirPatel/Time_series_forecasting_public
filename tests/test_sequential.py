from __future__ import annotations

import warnings

import pandas as pd

from ts_forecasting.sequential import (
    CausalHistoryTransformer,
    FrozenMedianImputer,
    SequentialInferencePipeline,
)


class RecordingTransformer:
    def __init__(self) -> None:
        self.predict_batches: list[int] = []

    def fit(self, train_frame: pd.DataFrame) -> "RecordingTransformer":
        return self

    def transform_batch(self, batch_frame: pd.DataFrame) -> pd.DataFrame:
        self.predict_batches.append(int(batch_frame["ts_index"].iloc[0]))
        return batch_frame.copy()


class EchoPredictor:
    def fit(self, train_frame: pd.DataFrame) -> "EchoPredictor":
        return self

    def predict_batch(self, batch_frame: pd.DataFrame):
        return batch_frame["feature_a"].to_numpy()


class StaticRecordingTransformer:
    supports_full_frame_prediction = True

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def fit(self, train_frame: pd.DataFrame) -> "StaticRecordingTransformer":
        return self

    def transform_batch(self, batch_frame: pd.DataFrame) -> pd.DataFrame:
        self.batch_sizes.append(int(len(batch_frame)))
        return batch_frame.copy()


class StaticEchoPredictor:
    supports_full_frame_prediction = True

    def __init__(self) -> None:
        self.calls = 0

    def fit(self, train_frame: pd.DataFrame) -> "StaticEchoPredictor":
        return self

    def predict_batch(self, batch_frame: pd.DataFrame):
        self.calls += 1
        return batch_frame["feature_a"].to_numpy()


def test_sequential_pipeline_processes_batches_in_time_order() -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["t1"],
            "ts_index": [1],
            "feature_a": [1.0],
        }
    )
    test_frame = pd.DataFrame(
        {
            "id": ["b", "a", "c"],
            "ts_index": [3, 1, 2],
            "feature_a": [30.0, 10.0, 20.0],
        }
    )

    transformer = RecordingTransformer()
    pipeline = SequentialInferencePipeline(
        transformer=transformer,
        predictor=EchoPredictor(),
    ).fit(train_frame)
    transformer.predict_batches.clear()

    predictions = pipeline.predict(test_frame)

    assert transformer.predict_batches == [1, 2, 3]
    assert predictions["id"].tolist() == ["a", "c", "b"]
    assert predictions["prediction"].tolist() == [10.0, 20.0, 30.0]


def test_sequential_pipeline_uses_full_frame_fast_path_when_supported() -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["t1"],
            "ts_index": [1],
            "feature_a": [1.0],
        }
    )
    test_frame = pd.DataFrame(
        {
            "id": ["b", "a", "c"],
            "ts_index": [3, 1, 2],
            "feature_a": [30.0, 10.0, 20.0],
        }
    )

    transformer = StaticRecordingTransformer()
    predictor = StaticEchoPredictor()
    pipeline = SequentialInferencePipeline(
        transformer=transformer,
        predictor=predictor,
    ).fit(train_frame)
    transformer.batch_sizes.clear()
    predictor.calls = 0

    predictions = pipeline.predict(test_frame)

    assert transformer.batch_sizes == [3]
    assert predictor.calls == 1
    assert predictions["id"].tolist() == ["a", "c", "b"]
    assert predictions["prediction"].tolist() == [10.0, 20.0, 30.0]


def test_sequential_pipeline_already_sorted_fast_path_matches_default() -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["t1"],
            "ts_index": [1],
            "feature_a": [1.0],
        }
    )
    sorted_frame = pd.DataFrame(
        {
            "id": ["a", "c", "b"],
            "ts_index": [1, 2, 3],
            "feature_a": [10.0, 20.0, 30.0],
        }
    )

    pipeline = SequentialInferencePipeline(
        transformer=StaticRecordingTransformer(),
        predictor=StaticEchoPredictor(),
    ).fit(train_frame)

    default_predictions = pipeline.predict(sorted_frame)
    fast_predictions = pipeline.predict(sorted_frame, already_sorted=True)

    pd.testing.assert_frame_equal(fast_predictions, default_predictions)


def test_frozen_median_imputer_uses_train_statistics_only() -> None:
    train_frame = pd.DataFrame(
        {
            "feature_a": [1.0, None, 5.0],
            "feature_b": [None, 4.0, 8.0],
        }
    )
    batch_frame = pd.DataFrame(
        {
            "feature_a": [None],
            "feature_b": [None],
        }
    )

    imputer = FrozenMedianImputer().fit(train_frame)
    transformed = imputer.transform_batch(batch_frame)

    assert transformed.loc[0, "feature_a"] == 3.0
    assert transformed.loc[0, "feature_b"] == 6.0


def test_causal_history_transformer_uses_only_prior_time_state() -> None:
    train_frame = pd.DataFrame(
        {
            "id": ["t1", "t2", "t3"],
            "code": ["A", "A", "A"],
            "sub_code": ["X", "X", "X"],
            "sub_category": ["S", "S", "S"],
            "horizon": [1, 1, 1],
            "ts_index": [1, 2, 2],
            "feature_a": [10.0, 20.0, 30.0],
        }
    )
    future_frame = pd.DataFrame(
        {
            "id": ["u1", "u2", "u3"],
            "code": ["A", "A", "A"],
            "sub_code": ["X", "X", "X"],
            "sub_category": ["S", "S", "S"],
            "horizon": [1, 1, 1],
            "ts_index": [3, 3, 4],
            "feature_a": [40.0, 50.0, 60.0],
        }
    )

    transformer = CausalHistoryTransformer(
        level_columns=["code", "sub_code", "horizon"],
        source_columns=["feature_a"],
        history_features=["last", "mean", "delta"],
    )
    transformed_train = transformer.fit(train_frame).transform_batch(train_frame)
    transformed_future = transformer.transform_batch(future_frame)

    assert transformed_train["feature_a_history_last"].isna().tolist() == [True, False, False]
    assert transformed_train["feature_a_history_last"].tolist()[1:] == [10.0, 10.0]
    assert transformed_train["feature_a_history_mean"].isna().tolist() == [True, False, False]
    assert transformed_train["feature_a_history_mean"].tolist()[1:] == [10.0, 10.0]
    assert transformed_train["feature_a_history_delta"].isna().tolist() == [True, False, False]
    assert transformed_train["feature_a_history_delta"].tolist()[1:] == [10.0, 20.0]

    assert transformed_future["feature_a_history_last"].tolist() == [30.0, 30.0, 50.0]
    assert transformed_future["feature_a_history_mean"].tolist() == [20.0, 20.0, 30.0]
    assert transformed_future["feature_a_history_delta"].tolist() == [10.0, 20.0, 10.0]


def test_causal_history_transformer_avoids_fragmentation_warnings() -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "code": ["A", "A"],
            "sub_code": ["X", "X"],
            "sub_category": ["S", "S"],
            "horizon": [1, 1],
            "ts_index": [1, 2],
            **{f"feature_{index:02d}": [float(index), float(index + 1)] for index in range(40)},
        }
    )

    transformer = CausalHistoryTransformer(
        level_columns=["code", "sub_code", "horizon"],
        history_features=["last", "mean", "delta"],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformer.fit(frame).transform_batch(frame)

    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, pd.errors.PerformanceWarning)
    ]
