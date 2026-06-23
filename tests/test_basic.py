import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, Normal, k_categorical


# TODO: This is non-deterministic in the model fitting
def test_classification(breast_cancer_data):
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        log_loss,
        roc_auc_score,
    )

    x_train, x_test, y_train, y_test = breast_cancer_data
    ngb = NGBClassifier(Dist=Bernoulli, verbose=False)
    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    score = roc_auc_score(y_test, preds)

    # loose score requirement so it isn't failing all the time
    assert score >= 0.85

    preds = ngb.predict_proba(x_test)
    score = log_loss(y_test, preds)
    assert score <= 0.30

    score = ngb.score(x_test, y_test)
    assert score <= 0.30

    dist = ngb.pred_dist(x_test)
    assert isinstance(dist, Bernoulli)

    score = roc_auc_score(y_test, preds[:, 1])

    assert score >= 0.85


# TODO: This is non-deterministic in the model fitting
def test_regression(california_housing_data):
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        mean_squared_error,
    )

    x_train, x_test, y_train, y_test = california_housing_data
    ngb = NGBRegressor(verbose=False)
    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    score = mean_squared_error(y_test, preds)
    assert score <= 15

    score = ngb.score(x_test, y_test)
    assert score <= 15

    dist = ngb.pred_dist(x_test)
    assert isinstance(dist, Normal)

    score = mean_squared_error(y_test, preds)
    assert score <= 15


def test_regression_accepts_base_learner_per_distribution_parameter():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    Y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=80)

    ngb = NGBRegressor(
        Dist=Normal,
        Base=[DecisionTreeRegressor(max_depth=1), Ridge(alpha=0.0)],
        n_estimators=2,
        natural_gradient=False,
        verbose=False,
    )

    ngb.fit(X, Y)

    assert len(ngb.base_models) == 2
    for models in ngb.base_models:
        assert len(models) == Normal.n_params
        assert isinstance(models[0], DecisionTreeRegressor)
        assert isinstance(models[1], Ridge)

    preds = ngb.predict(X[:5])
    dist = ngb.pred_dist(X[:5])

    assert preds.shape == (5,)
    assert isinstance(dist, Normal)


def test_classification_accepts_base_learner_per_distribution_parameter():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(90, 3))
    Y = np.argmax(np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]]), axis=1)

    ngb = NGBClassifier(
        Dist=k_categorical(3),
        Base=[DecisionTreeRegressor(max_depth=1), DecisionTreeRegressor(max_depth=2)],
        n_estimators=2,
        natural_gradient=False,
        verbose=False,
    )

    ngb.fit(X, Y)

    for models in ngb.base_models:
        assert len(models) == 2
        assert models[0].max_depth == 1
        assert models[1].max_depth == 2

    assert ngb.predict_proba(X[:5]).shape == (5, 3)


def test_base_learner_sequence_must_match_distribution_parameter_count():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    Y = rng.normal(size=20)

    ngb = NGBRegressor(
        Dist=Normal,
        Base=[DecisionTreeRegressor(max_depth=1)],
        n_estimators=1,
        verbose=False,
    )

    with pytest.raises(ValueError, match="sequence of 2 estimators"):
        ngb.fit(X, Y)


def test_single_base_learner_matches_repeated_base_learner_sequence():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    Y = X[:, 0] + rng.normal(scale=0.1, size=60)
    base = DecisionTreeRegressor(max_depth=2, random_state=0)

    single_base = NGBRegressor(
        Dist=Normal,
        Base=base,
        n_estimators=3,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y)
    repeated_base = NGBRegressor(
        Dist=Normal,
        Base=[base, base],
        n_estimators=3,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y)

    np.testing.assert_allclose(single_base.pred_param(X), repeated_base.pred_param(X))


def test_feature_importances_are_none_for_mixed_base_learners():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    Y = X[:, 0] + rng.normal(scale=0.1, size=40)

    ngb = NGBRegressor(
        Dist=Normal,
        Base=[DecisionTreeRegressor(max_depth=1), Ridge(alpha=0.0)],
        n_estimators=1,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y)

    assert ngb.feature_importances_ is None
