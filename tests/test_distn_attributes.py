import copy
import pickle
import warnings

import numpy as np
import pytest
import scipy.stats

from ngboost import NGBRegressor, NGBSurvival
from ngboost.distns import (
    Exponential,
    Gamma,
    HalfNormal,
    Laplace,
    LogNormal,
    Normal,
    Poisson,
    T,
    TFixedDf,
    TFixedDfFixedVar,
    Weibull,
)

SCIPY_DISTS = [
    Normal,
    LogNormal,
    Exponential,
    Gamma,
    HalfNormal,
    Laplace,
    Poisson,
    T,
    TFixedDf,
    TFixedDfFixedVar,
    Weibull,
]


def _dist(Dist, n=4):
    rng = np.random.default_rng(0)
    return Dist(rng.normal(size=(Dist.n_params, n)))


def test_lognormal_params_match_scipy():
    mu, log_sigma = np.array([0.5, -1.0]), np.array([np.log(0.3), np.log(2.0)])
    d = LogNormal(np.array([mu, log_sigma]))
    np.testing.assert_allclose(d.mu, mu)
    np.testing.assert_allclose(d.sigma, np.exp(log_sigma))
    np.testing.assert_allclose(d.params["s"], np.exp(log_sigma))
    np.testing.assert_allclose(d.params["scale"], np.exp(mu))
    y = np.array([1.3, 0.2])
    np.testing.assert_allclose(d.cdf(y), scipy.stats.lognorm(**d.params).cdf(y))


def test_lognormal_loc_and_scale_are_deprecated():
    d = _dist(LogNormal)
    with pytest.warns(FutureWarning, match="LogNormal.mu"):
        np.testing.assert_array_equal(d.loc, d.mu)
    with pytest.warns(FutureWarning, match="params\\['s'\\]"):
        np.testing.assert_array_equal(d.scale, d.sigma)


@pytest.mark.parametrize("score", ["LogScore", "CRPScore"])
def test_lognormal_fit_does_not_use_deprecated_attributes(score):
    from ngboost import scores  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    T_ = np.exp(X[:, 0] + rng.normal(size=200))
    E = rng.random(200) > 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ngb = NGBRegressor(Dist=LogNormal, Score=getattr(scores, score), n_estimators=5)
        ngb.fit(X, T_).pred_dist(X).params
        ngb = NGBSurvival(Dist=LogNormal, Score=getattr(scores, score), n_estimators=5)
        ngb.fit(X, T_, E).pred_dist(X).params


@pytest.mark.parametrize("Dist", SCIPY_DISTS)
def test_unknown_attribute_raises(Dist):
    d = _dist(Dist)
    with pytest.raises(AttributeError):
        d.not_an_attribute  # pylint: disable=pointless-statement
    assert not hasattr(d, "not_an_attribute")
    assert callable(d.mean)


@pytest.mark.parametrize("Dist", SCIPY_DISTS)
def test_pickle_and_deepcopy_distribution(Dist):
    d = _dist(Dist)
    for d2 in (pickle.loads(pickle.dumps(d)), copy.deepcopy(d)):
        for name, value in d.params.items():
            np.testing.assert_array_equal(d2.params[name], value)
        np.testing.assert_array_equal(d2.mean(), d.mean())
