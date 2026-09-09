from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_forecasting.lgbm_models import (
    BANNED_FEATURES,
    DEFAULT_PARAMS,
    LGBMForecaster,
    build_runtime_lgbm_params,
    build_feature_frame,
    infer_auto_num_threads,
    prepare_feature_frame,
)


def _make_frame(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    codes = ["A", "B", "C", "D"]
    sub_cats = ["C1", "C2"]
    horizons = [1, 3, 10, 25]
    return pd.DataFrame(
        {
            "id": range(n),
            "code": rng.choice(codes, n),
            "sub_code": rng.choice(["S1", "S2", "S3"], n),
            "sub_category": rng.choice(sub_cats, n),
            "horizon": rng.choice(horizons, n),
            "ts_index": np.arange(1, n + 1),
            "y_target": rng.randn(n) * 10,
            "weight": rng.uniform(0.5, 2.0, n),
            "feature_a": rng.randn(n),
            "feature_b": rng.randn(n),
            "feature_c": rng.randn(n),
            "feature_al": rng.randn(n),
        }
    )


def test_build_feature_frame_excludes_banned_and_includes_categoricals():
    frame = _make_frame()
    X, names, cat_maps = build_feature_frame(frame)
    assert "feature_a" in names
    assert "feature_b" in names
    assert "feature_c" in names
    assert "feature_al" not in names
    assert "code" in names
    assert "sub_category" in names
    assert "horizon" in names
    assert "ts_index" in names
    assert "y_target" not in names
    assert "weight" not in names
    assert "id" not in names
    assert "sub_code" not in names
    assert X["code"].dtype.name == "category"
    assert X["sub_category"].dtype.name == "category"
    assert X["horizon"].dtype.name == "category"
    assert "code" in cat_maps
    assert "sub_category" in cat_maps
    assert "horizon" in cat_maps


def test_build_feature_frame_reuses_category_maps():
    train = _make_frame(seed=1)
    _, names, cat_maps = build_feature_frame(train)
    # Build a test frame with a code unseen in train
    test = _make_frame(n=50, seed=99)
    test.loc[0, "code"] = "UNSEEN_CODE"
    X_test, _, _ = build_feature_frame(
        test, feature_names=names, category_maps=cat_maps,
    )
    # Unseen codes become NaN under the fixed category dtype
    assert X_test["code"].dtype.name == "category"
    assert set(X_test["code"].cat.categories) == set(train["code"].unique())


def test_build_feature_frame_uses_provided_feature_names():
    frame = _make_frame()
    _, names, _ = build_feature_frame(frame)
    frame2 = _make_frame(seed=99)
    X2, names2, _ = build_feature_frame(frame2, feature_names=names)
    assert names2 == names
    assert len(X2) == len(frame2)


def test_build_feature_frame_includes_extra_numeric_columns():
    frame = _make_frame()
    frame["incumbent_pred"] = np.random.randn(len(frame))
    X, names, _ = build_feature_frame(
        frame, extra_numeric_columns=["incumbent_pred"],
    )
    assert "incumbent_pred" in names


def test_prepare_feature_frame_preserves_columns_and_category_maps():
    frame = _make_frame()
    prepared = prepare_feature_frame(frame)

    X, names, cat_maps = build_feature_frame(frame)
    prepared_X, prepared_names, prepared_cat_maps = prepared.to_frame()

    assert prepared_names == names
    pd.testing.assert_frame_equal(prepared_X, X)
    assert prepared_cat_maps.keys() == cat_maps.keys()
    for column, dtype in cat_maps.items():
        assert prepared_cat_maps[column] == dtype


def test_prepared_feature_frame_slice_matches_build_feature_frame():
    frame = _make_frame(n=60, seed=7)
    prepared = prepare_feature_frame(frame)

    sliced = prepared.slice_rows(frame.index[:20])
    prepared_X, prepared_names, prepared_cat_maps = sliced.to_frame()
    expected_X, expected_names, expected_cat_maps = build_feature_frame(
        frame.iloc[:20].copy(),
    )

    assert prepared_names == expected_names
    pd.testing.assert_frame_equal(prepared_X, expected_X)
    assert prepared_cat_maps.keys() == expected_cat_maps.keys()


def test_lgbm_forecaster_uses_prepared_frames_with_incumbent_pred(monkeypatch):
    train = _make_frame(n=300, seed=1)
    valid = _make_frame(n=100, seed=2)
    train["incumbent_pred"] = np.random.randn(len(train))
    valid["incumbent_pred"] = np.random.randn(len(valid))

    train_prepared = prepare_feature_frame(train, extra_numeric_columns=["incumbent_pred"])
    valid_prepared = prepare_feature_frame(valid, extra_numeric_columns=["incumbent_pred"])
    train_features, _, train_cat_maps = train_prepared.to_frame()
    valid_features, _, _ = valid_prepared.to_frame(category_maps=train_cat_maps)

    monkeypatch.setattr(
        "ts_forecasting.lgbm_models.build_feature_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_feature_frame should not run for prepared frames")
        ),
    )

    model = LGBMForecaster(
        params={**DEFAULT_PARAMS, "num_leaves": 4, "verbose": -1},
        n_estimators=10,
        extra_numeric_columns=["incumbent_pred"],
    )
    model.fit(
        train,
        eval_frame=valid,
        prepared_train_frame=train_features,
        prepared_eval_frame=valid_features,
    )
    preds = model.predict(valid, prepared_frame=valid_features)

    assert preds.shape == (100,)
    assert "incumbent_pred" in model.feature_names_


def test_infer_auto_num_threads_caps_large_hosts():
    assert infer_auto_num_threads(logical_cpu_count=128) == 64


def test_infer_auto_num_threads_keeps_small_hosts_uncapped():
    assert infer_auto_num_threads(logical_cpu_count=20) == 20


def test_build_runtime_lgbm_params_enables_colwise_on_many_threads():
    params = build_runtime_lgbm_params(logical_cpu_count=128)

    assert params["num_threads"] == 64
    assert params["force_col_wise"] is True


def test_build_runtime_lgbm_params_skips_colwise_on_small_hosts():
    params = build_runtime_lgbm_params(logical_cpu_count=8)

    assert params["num_threads"] == 8
    assert "force_col_wise" not in params


def test_lgbm_forecaster_fit_predict_shape():
    train = _make_frame(n=300, seed=1)
    test = _make_frame(n=100, seed=2)
    model = LGBMForecaster(
        params={**DEFAULT_PARAMS, "num_leaves": 4, "verbose": -1},
        n_estimators=10,
    )
    model.fit(train)
    preds = model.predict(test)
    assert preds.shape == (100,)
    assert not np.any(np.isnan(preds))


def test_lgbm_forecaster_with_early_stopping():
    train = _make_frame(n=300, seed=1)
    val = _make_frame(n=100, seed=2)
    model = LGBMForecaster(
        params={**DEFAULT_PARAMS, "num_leaves": 4, "verbose": -1},
        n_estimators=200,
        early_stopping_rounds=5,
    )
    model.fit(train, eval_frame=val)
    preds = model.predict(val)
    assert preds.shape == (100,)
    assert model.model_.best_iteration <= 200


def test_lgbm_forecaster_with_incumbent_pred():
    train = _make_frame(n=300, seed=1)
    train["incumbent_pred"] = np.random.randn(len(train))
    test = _make_frame(n=100, seed=2)
    test["incumbent_pred"] = np.random.randn(len(test))
    model = LGBMForecaster(
        params={**DEFAULT_PARAMS, "num_leaves": 4, "verbose": -1},
        n_estimators=10,
        extra_numeric_columns=["incumbent_pred"],
    )
    model.fit(train)
    preds = model.predict(test)
    assert preds.shape == (100,)
    assert "incumbent_pred" in model.feature_names_


def test_lgbm_forecaster_unweighted():
    train = _make_frame(n=300, seed=1)
    model = LGBMForecaster(
        params={**DEFAULT_PARAMS, "num_leaves": 4, "verbose": -1},
        n_estimators=10,
        use_sample_weight=False,
    )
    model.fit(train)
    preds = model.predict(train)
    assert preds.shape == (300,)


def test_lgbm_forecaster_raises_if_not_fitted():
    model = LGBMForecaster()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(_make_frame(n=10))
