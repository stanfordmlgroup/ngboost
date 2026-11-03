"""pred_param checks the input once and skips constant trees, bit-identical to the loop."""

# pylint: disable=protected-access
import pickle
import warnings

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from ngboost import NGBClassifier, NGBoost, NGBRegressor
from ngboost.distns import Bernoulli, Normal, T, k_categorical
from ngboost.ngboost import NGBoostDegenerateWarning
from ngboost.scores import LogScore


def loop_pred_param(model, X, max_iter=None):
    """The boosting loop pred_param runs without a plan: the reference for every test."""
    params = np.ones((X.shape[0], model.Manifold.n_params)) * model.init_params
    for i, (models, s, col_idx) in enumerate(
        zip(model.base_models, model.scalings, model.col_idxs)
    ):
        if max_iter and i == max_iter:
            break
        resids = np.array([m.predict(X[:, col_idx]) for m in models]).T
        params -= model.learning_rate * resids * s
    return params


def assert_bitwise_equal(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


def toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    Y = 1.5 * (X[:, 0] > 0.3) + rng.normal(scale=0.5 + 0.8 * (X[:, 1] > 0), size=n)
    return X, Y


class NormalFixedInit(Normal):
    """A fixed start, so that constant trees take nonzero steps."""

    @staticmethod
    def fit(Y):
        return np.array([0.0, 0.0])


def constant_tree():
    return DecisionTreeRegressor(max_depth=2, min_samples_leaf=10_000, random_state=0)


def split_tree():
    return DecisionTreeRegressor(max_depth=2, min_samples_leaf=10, random_state=0)


def make_ngb(base, n_estimators=40, **kwargs):
    return NGBoost(
        Dist=NormalFixedInit,
        Score=LogScore,
        Base=base,
        n_estimators=n_estimators,
        learning_rate=0.05,
        verbose=False,
        random_state=0,
        **kwargs,
    )


def fit_quietly(model, X, Y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NGBoostDegenerateWarning)
        return model.fit(X, Y)


def test_all_constant_model_warns_and_calls_no_tree():
    X, Y = toy_data()
    with pytest.warns(NGBoostDegenerateWarning):
        model = make_ngb(constant_tree(), minibatch_frac=0.7, col_sample=0.5).fit(X, Y)
    assert np.abs(model.pred_param(X)[0] - model.init_params).max() > 0.1
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))
    assert not model._pred_plan.steps


def test_no_warning_when_trees_split():
    X, Y = toy_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", NGBoostDegenerateWarning)
        model = make_ngb(split_tree(), n_estimators=5).fit(X, Y)
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))


def test_constant_and_split_trees_per_parameter():
    X, Y = toy_data()
    model = make_ngb([split_tree(), constant_tree()]).fit(X, Y)
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))
    assert {k for _, k, step in model._pred_plan.steps if step is None} == {0}


def test_plan_follows_edited_rounds():
    X, Y = toy_data()
    model = make_ngb([split_tree(), split_tree()], n_estimators=30).fit(X, Y)
    model.pred_param(X)
    for models in model.base_models[10:]:  # constant trees after trees that split
        for k in range(2):
            models[k] = constant_tree().fit(X, np.full(len(X), 0.01 * (k + 1)))
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))
    assert model._pred_plan.n_constant == 40
    model.base_models = model.base_models[:7]
    model.scalings = model.scalings[:7]
    model.col_idxs = model.col_idxs[:7]
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))
    model.learning_rate = 0.1
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))


def test_max_iter_and_staged_predictions():
    X, Y = toy_data()
    model = make_ngb([split_tree(), constant_tree()], n_estimators=20).fit(X, Y)
    for max_iter in (1, 5, 19):
        assert_bitwise_equal(
            model.pred_param(X, max_iter=max_iter),
            loop_pred_param(model, X, max_iter=max_iter),
        )
    assert_bitwise_equal(
        model.staged_pred_dist(X)[-1]._params, model.pred_dist(X)._params
    )


def test_refit_and_partial_fit():
    X, Y = toy_data()
    X2, Y2 = toy_data(seed=1)
    model = fit_quietly(make_ngb(constant_tree(), n_estimators=15), X, Y)
    fit_quietly(model, X2, Y2 + 3.0)
    assert_bitwise_equal(model.pred_param(X2), loop_pred_param(model, X2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NGBoostDegenerateWarning)
        model.partial_fit(X, Y)
    assert len(model.base_models) == 30
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))


def test_plan_is_not_pickled():
    X, Y = toy_data()
    model = fit_quietly(make_ngb(constant_tree(), n_estimators=15), X, Y)
    loaded = pickle.loads(pickle.dumps(model))
    assert getattr(loaded, "_pred_plan", None) is None
    assert_bitwise_equal(loaded.pred_param(X), loop_pred_param(model, X))


@pytest.mark.parametrize("dist", [Normal, T])
def test_regressor_with_early_stopping(dist):
    X, Y = toy_data()
    model = NGBRegressor(
        Dist=dist,
        Base=DecisionTreeRegressor(max_depth=2, min_samples_leaf=60),
        n_estimators=30,
        verbose=False,
        random_state=0,
        minibatch_frac=0.5,
        col_sample=0.5,
        early_stopping_rounds=3,
    ).fit(X, Y)
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))


@pytest.mark.parametrize("dist,K", [(Bernoulli, 2), (k_categorical(3), 3)])
def test_classifiers(dist, K):
    X, Y = toy_data()
    y = np.digitize(Y, np.quantile(Y, np.linspace(0, 1, K + 1)[1:-1]))
    model = NGBClassifier(
        Dist=dist, Base=split_tree(), n_estimators=20, verbose=False, random_state=0
    ).fit(X, y)
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))


@pytest.mark.parametrize(
    "base",
    [
        ExtraTreeRegressor(max_depth=3, random_state=0),
        DecisionTreeRegressor(max_depth=3, monotonic_cst=[1, 0, 0, 0]),
    ],
)
def test_other_tree_settings(base):
    X, Y = toy_data()
    model = NGBRegressor(Base=base, n_estimators=20, verbose=False, random_state=0)
    model.fit(X, Y)
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))


def test_missing_values_follow_the_trees():
    X, Y = toy_data()
    X[::7, 2] = np.nan
    model = NGBRegressor(Base=split_tree(), n_estimators=20, verbose=False).fit(X, Y)
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))
    no_nan = DecisionTreeRegressor(max_depth=2, criterion="absolute_error")
    model = NGBRegressor(Base=no_nan, n_estimators=5, verbose=False)
    model.fit(*toy_data())
    with pytest.raises(ValueError):
        model.pred_param(X)


def test_input_width():
    X, Y = toy_data()
    model = fit_quietly(make_ngb(constant_tree(), n_estimators=10), X, Y)
    wide = np.hstack([X, X])
    assert_bitwise_equal(model.pred_param(wide), loop_pred_param(model, wide))
    with pytest.raises(IndexError):
        model.pred_param(X[:, :2])
    with pytest.raises(ValueError):
        model.pred_param(np.where(np.arange(4) == 1, np.inf, X))


def test_other_learners_and_sparse_input_use_the_loop():
    X, Y = toy_data()
    model = NGBRegressor(Base=Ridge(), n_estimators=10, verbose=False).fit(X, Y)
    assert_bitwise_equal(model.pred_param(X), loop_pred_param(model, X))
    assert model._pred_plan is None
    model = NGBRegressor(Base=split_tree(), n_estimators=10, verbose=False).fit(X, Y)
    X_sparse = sparse.csr_matrix(X)
    assert_bitwise_equal(model.pred_param(X_sparse), loop_pred_param(model, X_sparse))
