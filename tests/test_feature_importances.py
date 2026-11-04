"""Feature-importance normalization for parameters without splitting trees."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor


@pytest.mark.parametrize("learner", [NGBRegressor, NGBClassifier])
@pytest.mark.parametrize("col_sample", [1.0, 0.5])
def test_feature_importances_are_zero_for_stumps(learner, col_sample):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    Y = X[:, 0] if learner is NGBRegressor else (X[:, 0] > 0).astype(int)
    model = learner(
        Base=DecisionTreeRegressor(min_samples_split=100),
        n_estimators=3,
        col_sample=col_sample,
        random_state=0,
        verbose=False,
    ).fit(X, Y)

    with np.errstate(invalid="raise", divide="raise"):
        importances = model.feature_importances_
    np.testing.assert_array_equal(
        importances, np.zeros((model.Dist.n_params, X.shape[1]))
    )


@pytest.mark.parametrize("col_sample", [1.0, 0.5])
def test_feature_importances_keep_informative_parameter(col_sample):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    Y = 3 * X[:, 0] + np.sin(X[:, 1])
    model = NGBRegressor(
        Base=[
            DecisionTreeRegressor(max_depth=2),
            DecisionTreeRegressor(min_samples_split=100),
        ],
        n_estimators=3,
        col_sample=col_sample,
        random_state=0,
        verbose=False,
    ).fit(X, Y)

    with np.errstate(invalid="raise", divide="raise"):
        importances = model.feature_importances_
    assert importances.shape == (2, X.shape[1])
    assert np.isfinite(importances).all()
    assert importances[0].sum() == pytest.approx(1)
    np.testing.assert_array_equal(importances[1], np.zeros(X.shape[1]))


@pytest.mark.parametrize("col_sample", [1.0, 0.5])
def test_feature_importances_normalize_splitting_trees(col_sample):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    Y = 3 * X[:, 0] + np.sin(X[:, 1])
    model = NGBRegressor(
        Base=DecisionTreeRegressor(max_depth=2),
        n_estimators=3,
        col_sample=col_sample,
        random_state=0,
        verbose=False,
    ).fit(X, Y)

    with np.errstate(invalid="raise", divide="raise"):
        importances = model.feature_importances_
    assert importances.shape == (2, X.shape[1])
    assert np.isfinite(importances).all()
    np.testing.assert_allclose(importances.sum(axis=1), np.ones(2))
