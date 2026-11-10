import warnings
from typing import Tuple

import numpy as np
import pytest
import scipy.stats
import sympy as sp
from scipy.stats import multivariate_normal
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import (
    Bernoulli,
    Beta,
    Cauchy,
    Distn,
    Exponential,
    Gamma,
    HalfNormal,
    LogNormal,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    NormalFixedMean,
    NormalFixedVar,
    T,
    TFixedDf,
    TFixedDfFixedVar,
    Weibull,
    k_categorical,
)
from ngboost.distns.sympy_utils import make_distribution
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
        # This case fails for both learners, so the marker is strict: if T starts
        # to pass, the run fails and the marker has to go. Only a ValueError
        # counts as the expected failure.
        pytest.param(
            T,
            marks=pytest.mark.xfail(
                reason=(
                    "NGBRegressor(Dist=T) raises a ValueError: the natural "
                    "gradient for log df runs away as df grows, and the "
                    "gradient goes to NaN (#275)"
                ),
                raises=ValueError,
                strict=True,
            ),
        ),
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


def test_negative_binomial_matches_scipy():
    mu, n = np.array([0.5, 4.0, 30.0]), np.array([0.3, 2.0, 50.0])
    dist = manifold(LogScore, NegativeBinomial)(np.log(np.vstack([mu, n])))
    ref = scipy.stats.nbinom(n, n / (n + mu))
    y = np.array([0, 3, 41])
    assert np.allclose(dist.score(y), -ref.logpmf(y))
    assert np.allclose(dist.mean(), mu)
    assert np.allclose(dist.var(), mu + mu**2 / n)
    assert np.array_equal(dist.ppf(0.9), ref.ppf(0.9))


def test_negative_binomial_fit():
    Y = scipy.stats.nbinom(2.0, 0.5).rvs(size=20000, random_state=0)  # mu is 2
    log_mu, log_n = NegativeBinomial.fit(Y)
    assert np.exp(log_mu) == pytest.approx(2.0, rel=0.05)
    assert np.exp(log_n) == pytest.approx(2.0, rel=0.2)
    with pytest.raises(ValueError, match="counts"):
        NegativeBinomial.fit(np.array([0.0, 1.5, 3.0]))


@pytest.mark.slow
def test_negative_binomial_regression():
    np.random.seed(0)  # the Monte Carlo metric samples from the global generator
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    mu = np.exp(1 + X[:, 0])
    Y = scipy.stats.nbinom(2.0, 2.0 / (2.0 + mu)).rvs(random_state=rng)
    ngb = NGBRegressor(Dist=NegativeBinomial, n_estimators=200, verbose=False)
    ngb.fit(X, Y)
    assert np.corrcoef(ngb.predict(X), mu)[0, 1] > 0.9
    assert np.median(ngb.pred_dist(X).params["n"]) == pytest.approx(2.0, rel=0.3)


@pytest.mark.parametrize("dist", [Beta, NegativeBinomial])
def test_factory_sample_shape(dist):
    assert dist(np.zeros((dist.n_params, 3))).sample(4).shape == (4, 3)


def test_factory_scipy_expression_needs_fit_fn():
    mu, n, y = sp.symbols("mu n y", positive=True)
    with pytest.raises(ValueError, match="fit_fn"):
        make_distribution(
            params=[(mu, True), (n, True)],
            y=y,
            score_expr=(y - mu) ** 2 / n,
            scipy_dist_cls=scipy.stats.nbinom,
            scipy_kwarg_map={"n": n, "p": n / (n + mu)},
        )
