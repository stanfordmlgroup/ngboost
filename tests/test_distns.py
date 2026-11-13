import warnings
from typing import Tuple

import numpy as np
import pytest
from scipy.stats import multivariate_normal
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
from ngboost.manifold import manifold
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
def test_dists_runs_on_examples_logscore(dist: Distn, learner, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # TODO: test early stopping features
    ngb = NGBRegressor(Dist=dist, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # TODO: test properties of output


# Before df was capped, df ran away to infinity on these data within 60
# iterations, exp(log df) overflowed and the fit failed with NaN gradients
@pytest.mark.parametrize("units", [1.0, 1e4])
def test_t_df_stays_finite(units):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(500, 3))
    y = np.sin(np.pi * X[:, 0]) + X[:, 1] + 0.5 * rng.standard_t(5, size=500)
    ngb = NGBRegressor(Dist=T, n_estimators=100, verbose=False, random_state=0)
    ngb.fit(X, units * y)
    df = ngb.pred_dist(X).df
    assert np.all(np.isfinite(df))
    assert np.all(df <= T.max_df)


def test_t_df_is_capped():
    # a log df far above the cap, as boosting steps can give
    D = manifold(LogScore, T)(np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 800.0]]))
    np.testing.assert_allclose(D.df, [np.e, T.max_df])
    y = np.array([0.5, -2.0])
    assert np.all(np.isfinite(D.score(y)))
    assert np.all(np.isfinite(D.grad(y, natural=True)))


def test_t_no_push_above_cap():
    # Both rows are at the cap. Y near loc favors a larger df, an outlier a
    # smaller one. Boosting subtracts the gradient.
    D = manifold(LogScore, T)(np.array([[0.0, 0.0], [0.0, 0.0], [800.0, 800.0]]))
    grad = D.grad(np.array([0.1, 10.0]), natural=True)
    assert grad[0, 2] == 0
    assert grad[1, 2] > 0


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


def _mvn_with_cov(cov, N=4):
    """N MultivariateNormal rows with means 0, ..., p - 1 and covariance cov"""
    p = cov.shape[0]
    # the internal parameters are the means and the lower triangle of L, where
    # inv(cov) = L @ L.T and the diagonal of L is on the log scale
    L = np.linalg.cholesky(np.linalg.inv(cov))
    L[np.diag_indices(p)] = np.log(L[np.diag_indices(p)])
    theta = np.concatenate([np.arange(p, dtype=float), L[np.tril_indices(p)]])
    return MultivariateNormal(p)(np.tile(theta[:, None], (1, N)))


def test_multivariatenormal_params_cov():
    cov = np.array([[2.0, 0.6, 0.0], [0.6, 1.0, -0.3], [0.0, -0.3, 0.5]])
    dist = _mvn_with_cov(cov)
    params = dist.params

    assert list(params) == ["loc", "cov"]
    np.testing.assert_allclose(params["loc"], np.tile([0.0, 1.0, 2.0], (4, 1)))
    # get_chol_factor adds 1e-6 to the diagonal of L, so the match is not exact
    np.testing.assert_allclose(
        params["cov"], np.broadcast_to(cov, (4, 3, 3)), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_array_equal(params["cov"], dist.cov)

    # the names fit scipy.stats.multivariate_normal row by row
    Y = np.random.default_rng(0).normal(size=(4, 3))
    scipy_logpdf = [
        multivariate_normal(mean=params["loc"][i], cov=params["cov"][i]).logpdf(Y[i])
        for i in range(4)
    ]
    np.testing.assert_allclose(scipy_logpdf, dist.logpdf(Y))


def test_multivariatenormal_params_scale_is_deprecated():
    dist = _mvn_with_cov(np.array([[1.0, 0.3], [0.3, 2.0]]))
    params = dist.params

    with pytest.warns(FutureWarning, match=r"Use params\['cov'\]") as record:
        scale = params["scale"]
    assert record[0].filename == __file__
    np.testing.assert_array_equal(scale, params["cov"])

    with pytest.warns(FutureWarning, match=r"Use params\['cov'\]") as record:
        scale = params.get("scale")
    assert record[0].filename == __file__
    np.testing.assert_array_equal(scale, params["cov"])

    # the other keys work as in a plain dict and give no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dict(params.items()).keys() == {"loc", "cov"}
        np.testing.assert_array_equal(params["loc"], dist.loc)
        np.testing.assert_array_equal(params.get("cov"), dist.cov)
        assert params.get("df") is None
        with pytest.raises(KeyError):
            params["df"]  # pylint: disable=pointless-statement
