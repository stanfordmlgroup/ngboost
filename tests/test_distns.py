from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival, distns
from ngboost.distns import (
    Bernoulli,
    Cauchy,
    ClassificationDistn,
    Distn,
    Exponential,
    Gamma,
    HalfNormal,
    LogNormal,
    MultivariateNormal,
    Normal,
    NormalFixedMean,
    NormalFixedVar,
    RegressionDistn,
    T,
    TFixedDf,
    TFixedDfFixedVar,
    Weibull,
    k_categorical,
)
from ngboost.scores import CRPScore, LogScore, Score

# test all the dist methods and score implementation methods,#
# i.e. they all return proper shapes and sizes and types
# check metric lines up with defaults for lognormal where applicable


Tuple4Array = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Tuple5Array = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="module")
def regression_data():
    data = fetch_california_housing()
    X, y = data["data"][:1000], data["target"][:1000]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def classification_data():
    data = load_breast_cancer()
    X, y = data["data"][:1000], data["target"][:1000]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def is_t_distribution(
    dist, learner, regression_data
):  # pylint: disable=unused-argument
    return dist == T


@pytest.mark.slow
@pytest.mark.parametrize(
    "dist",
    [
        Normal,
        NormalFixedVar,
        NormalFixedMean,
        HalfNormal,
        LogNormal,
        Exponential,
        Gamma,
        T,
        TFixedDf,
        TFixedDfFixedVar,
        Cauchy,
        Weibull,
    ],
)
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
    ],
)
@pytest.mark.xfail(
    condition=is_t_distribution, reason="Known to fail with T distribution"
)
def test_dists_runs_on_examples_logscore(dist: Distn, learner, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # TODO: test early stopping features
    ngb = NGBRegressor(Dist=dist, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # TODO: test properties of output


@pytest.mark.slow
@pytest.mark.parametrize("dist", [Normal, LogNormal, Exponential])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
    ],
)
def test_dists_runs_on_examples_crpscore(dist: Distn, learner, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # TODO: test early stopping features
    ngb = NGBRegressor(Dist=dist, Score=CRPScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # TODO: test properties of output


@pytest.mark.parametrize("dist", [LogNormal, Exponential])
@pytest.mark.parametrize("score", [LogScore, CRPScore])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
    ],
)
def test_survival_runs_on_examples(
    dist: Distn, score: Score, learner, california_housing_survival_data
):
    (
        X_train,
        X_test,
        T_surv_train,
        E_surv_train,
        Y_surv_test,
    ) = california_housing_survival_data
    # test early stopping features
    ngb = NGBSurvival(Dist=dist, Score=score, Base=learner, verbose=False)
    ngb.fit(X_train, T_surv_train, E_surv_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # test properties of output


@pytest.mark.slow
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
    ],
)
def test_bernoulli(learner, classification_data: Tuple4Array):
    X_cls_train, X_cls_test, Y_cls_train, Y_cls_test = classification_data
    # test early stopping features
    # test other args, n_trees, LR, minibatching- args as fixture
    ngb = NGBClassifier(Dist=Bernoulli, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_cls_train, Y_cls_train)
    y_pred = ngb.predict(X_cls_test)
    y_prob = ngb.predict_proba(X_cls_test)
    y_dist = ngb.pred_dist(X_cls_test)
    # test properties of output


@pytest.mark.slow
@pytest.mark.parametrize("k", [2, 4, 7])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
    ],
)
def test_categorical(k: int, learner, classification_data: Tuple4Array):
    X_train, X_test, y_train, _ = classification_data
    dist = k_categorical(k)
    y_train = np.random.randint(0, k, (len(y_train)))
    # test early stopping features
    ngb = NGBClassifier(Dist=dist, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_prob = ngb.predict_proba(X_test)
    y_dist = ngb.pred_dist(X_test)
    # test properties of output


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="squared_error", max_depth=5),
        DecisionTreeRegressor(criterion="squared_error", max_depth=3),
    ],
)
# Ignore the k=1 warning
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multivariatenormal(k: int, learner):
    dist = MultivariateNormal(k)

    # Generate some sample data
    N = 500
    X_train = np.random.randn(N, k)
    y_fns = [np.sin, np.cos, np.exp]
    y_cols = [
        fn(X_train[:, num_col]).reshape(-1, 1) + np.random.randn(N, 1)
        for num_col, fn in enumerate(y_fns[:k])
    ]
    y_train = np.hstack(y_cols)
    X_test = np.random.randn(N, k)

    ngb = NGBRegressor(
        Dist=dist, Score=LogScore, Base=learner, verbose=False, n_estimators=50
    )
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)

    mean = y_dist.mean
    sample = y_dist.rv()
    scipy_list = y_dist.scipy_distribution()


# Slicing and len. Distn.__getitem__ and Distn.__len__ read self._params, so
# every distribution has to store the params it was built from.

_N_ROWS = 6
# Distn.__getitem__ passes a one-dimensional params array to the constructor
# when the key is an integer. Categorical and MultivariateNormal expect a
# two-dimensional array, so dist[0] fails for them. This is separate from
# the missing _params that the NormalFixed classes had.
_NO_INT_INDEX = {"Bernoulli", "k_categorical", "MultivariateNormal"}


def _concrete_distns(int_index=False):
    """The concrete distributions in ngboost.distns.__all__.

    k_categorical and MultivariateNormal are factories, so they are called
    with a small size. Distn, RegressionDistn and ClassificationDistn are
    abstract bases and are left out.
    """
    bases = (Distn, RegressionDistn, ClassificationDistn)
    factories = {
        "k_categorical": lambda: k_categorical(3),
        "MultivariateNormal": lambda: MultivariateNormal(2),
    }
    cases = []
    for name in distns.__all__:
        obj = getattr(distns, name)
        if name in factories:
            obj = factories[name]()
        if obj in bases:
            continue
        marks = []
        if int_index and name in _NO_INT_INDEX:
            marks.append(pytest.mark.xfail(reason="integer key needs 2D params"))
        cases.append(pytest.param(obj, id=name, marks=marks))
    return cases


def _build(Dist):
    rng = np.random.default_rng(0)
    return Dist(rng.normal(scale=0.3, size=(Dist.n_params, _N_ROWS)))


def _rows(value, key):
    # a few distributions keep a scalar in params, such as the n of BetaBinomial
    return value if np.ndim(value) == 0 else value[key]


def test_concrete_distns_cover_all_exports():
    names = [case.id for case in _concrete_distns()]
    assert len(names) == len(set(names))
    assert {"Normal", "NormalFixedVar", "NormalFixedMean", "Bernoulli"} <= set(names)


@pytest.mark.parametrize("Dist", _concrete_distns())
def test_len(Dist):
    assert len(_build(Dist)) == _N_ROWS


@pytest.mark.parametrize("Dist", _concrete_distns())
def test_slice_keeps_class_and_params(Dist):
    dist = _build(Dist)
    sliced = dist[1:4]
    assert sliced.__class__ is Dist
    assert len(sliced) == 3
    assert sliced.params.keys() == dist.params.keys()
    for name, value in dist.params.items():
        np.testing.assert_allclose(sliced.params[name], _rows(value, slice(1, 4)))


@pytest.mark.parametrize("Dist", _concrete_distns(int_index=True))
def test_integer_index_keeps_class_and_params(Dist):
    dist = _build(Dist)
    single = dist[2]
    assert single.__class__ is Dist
    for name, value in dist.params.items():
        np.testing.assert_allclose(single.params[name], _rows(value, 2))


@pytest.mark.parametrize(
    "Dist, expected_loc, expected_scale",
    [
        (NormalFixedVar, lambda p: p[0], lambda p: np.ones(len(p[0]))),
        (NormalFixedMean, lambda p: np.zeros(len(p[0])), lambda p: np.exp(p[0])),
    ],
    ids=["NormalFixedVar", "NormalFixedMean"],
)
def test_normal_fixed_slice_matches_rows(Dist, expected_loc, expected_scale):
    params = np.random.default_rng(1).normal(size=(1, 8))
    dist = Dist(params)
    rows = slice(2, 7)
    sliced = dist[rows]
    assert sliced.__class__ is Dist
    assert len(dist) == 8
    assert len(sliced) == 5
    np.testing.assert_allclose(sliced.loc, expected_loc(params)[rows])
    np.testing.assert_allclose(sliced.scale, expected_scale(params)[rows])
    np.testing.assert_allclose(sliced.params["loc"], sliced.loc)
    np.testing.assert_allclose(sliced.params["scale"], sliced.scale)
    y = np.linspace(-1, 1, 8)
    np.testing.assert_allclose(
        sliced.dist.logpdf(y[rows]), dist.dist.logpdf(y)[rows], rtol=1e-12
    )


@pytest.mark.parametrize("Dist", [NormalFixedVar, NormalFixedMean])
def test_normal_fixed_fitted_prediction_supports_len_and_slice(Dist):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    Y = X[:, 0] + rng.normal(size=60)
    ngb = NGBRegressor(Dist=Dist, n_estimators=5, verbose=False).fit(X, Y)
    pred = ngb.pred_dist(X)
    assert len(pred) == 60
    sliced = pred[0:2]
    assert sliced.__class__ is Dist
    assert len(sliced) == 2
    np.testing.assert_allclose(sliced.loc, pred.loc[0:2])
    np.testing.assert_allclose(sliced.scale, pred.scale[0:2])
    np.testing.assert_allclose(pred[0].loc, pred.loc[0])
