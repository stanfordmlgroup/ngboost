"""Tests that sample weights reach the fit of the starting distribution"""

import numpy as np
import pytest

from ngboost import NGBRegressor, NGBSurvival
from ngboost.distns import (
    Bernoulli,
    Exponential,
    HalfNormal,
    Laplace,
    LogNormal,
    MultivariateNormal,
    Normal,
    Poisson,
    Weibull,
    k_categorical,
)
from ngboost.distns.normal import NormalFixedVar


@pytest.mark.parametrize(
    "dist,draw,atol",
    [
        (Normal, lambda rng, n: rng.normal(size=n), 1e-10),
        (NormalFixedVar, lambda rng, n: rng.normal(size=n), 1e-10),
        (LogNormal, lambda rng, n: rng.lognormal(size=n), 1e-10),
        (Exponential, lambda rng, n: rng.exponential(size=n), 1e-10),
        (HalfNormal, lambda rng, n: np.abs(rng.normal(size=n)), 1e-10),
        (Laplace, lambda rng, n: rng.laplace(size=n), 1e-10),
        # the unweighted fits of Poisson and Weibull are numerical
        (Poisson, lambda rng, n: rng.poisson(3.0, size=n).astype(float), 1e-6),
        (Weibull, lambda rng, n: rng.weibull(1.5, size=n), 1e-4),
        (Bernoulli, lambda rng, n: rng.integers(0, 2, size=n), 1e-10),
        (k_categorical(3), lambda rng, n: rng.integers(0, 3, size=n), 1e-10),
        (MultivariateNormal(2), lambda rng, n: rng.normal(size=(n, 2)), 1e-10),
    ],
)
def test_marginal_fit_weights_match_repeated_rows(dist, draw, atol):
    rng = np.random.default_rng(31)
    Y = draw(rng, 200)
    counts = rng.integers(0, 4, size=200)
    weighted = dist.fit(Y, sample_weight=counts.astype(float))
    repeated = dist.fit(np.repeat(Y, counts, axis=0))
    np.testing.assert_allclose(weighted, repeated, rtol=0, atol=atol)


def test_zero_weight_rows_do_not_move_init_params():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    Y = 3 * X[:, 0] + rng.normal(size=300)
    Y[200:] = 1000.0
    weights = np.r_[np.ones(200), np.zeros(100)]
    options = {
        "n_estimators": 2,
        "verbose": False,
        "random_state": 0,
        "minibatch_frac": 1.0,
    }
    with_weights = NGBRegressor(**options).fit(X, Y, sample_weight=weights)
    rows_removed = NGBRegressor(**options).fit(X[:200], Y[:200])
    np.testing.assert_allclose(with_weights.init_params, rows_removed.init_params)


class NormalWithUnweightedFit(Normal):
    """A custom distribution whose fit does not take sample_weight."""

    # pylint: disable=no-self-argument,arguments-differ
    def fit(Y):
        return Normal.fit(Y)


def test_distribution_without_weighted_fit_drops_zero_weight_rows():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2))
    Y = rng.normal(size=100)
    Y[80:] = 1000.0
    weights = np.r_[np.ones(80), np.zeros(20)]
    options = {
        "Dist": NormalWithUnweightedFit,
        "n_estimators": 2,
        "verbose": False,
        "random_state": 0,
    }
    model = NGBRegressor(**options).fit(X, Y, sample_weight=weights)
    np.testing.assert_allclose(model.init_params, Normal.fit(Y[:80]))


def test_weighted_mle_ignores_zero_weight_rows():
    rng = np.random.default_rng(3)
    Y = rng.weibull(1.5, size=100)
    weights = rng.uniform(0.5, 2.0, size=100)
    weights[80:] = 0.0
    np.testing.assert_allclose(
        Weibull.fit(Y, sample_weight=weights),
        Weibull.fit(Y[:80], sample_weight=weights[:80]),
        rtol=1e-6,
    )


def test_all_zero_sample_weight_raises():
    X = np.random.default_rng(4).normal(size=(20, 2))
    with pytest.raises(ValueError, match="at least one positive entry"):
        NGBRegressor(n_estimators=1, verbose=False).fit(
            X, X[:, 0], sample_weight=np.zeros(20)
        )


def test_survival_init_params_use_weights():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 2))
    T = rng.lognormal(size=120)
    T[100:] = 1e6
    E = np.ones(120, dtype=bool)
    weights = np.r_[np.ones(100), np.zeros(20)]
    options = {
        "n_estimators": 2,
        "verbose": False,
        "random_state": 0,
        "minibatch_frac": 1.0,
    }
    model = NGBSurvival(**options).fit(X, T, E, sample_weight=weights)
    np.testing.assert_allclose(model.init_params, LogNormal.fit(T[:100]))
