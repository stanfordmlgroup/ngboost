# pylint: disable=no-member,no-name-in-module
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from ngboost import NGBRegressor
from ngboost.distns import LogitNormal

SIGMAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
MUS = np.linspace(-10, 10, 9)


def _logitnormal(mu, sigma):
    return LogitNormal(np.array([mu, np.full_like(mu, np.log(sigma))]))


def test_predict_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    Y = expit(X[:, 0] + 0.5 * rng.normal(size=100))
    ngb = NGBRegressor(Dist=LogitNormal, n_estimators=5, verbose=False).fit(X, Y)
    np.testing.assert_array_equal(ngb.predict(X), ngb.predict(X))


@pytest.mark.parametrize("sigma", SIGMAS)
def test_mean_matches_monte_carlo(sigma):
    rng = np.random.default_rng(1)
    draws = expit(MUS + sigma * rng.standard_normal((1_000_000, len(MUS))))
    mc_mean = draws.mean(axis=0)
    mc_se = draws.std(axis=0) / np.sqrt(len(draws))
    mean = _logitnormal(MUS, sigma).mean()
    np.testing.assert_array_less(np.abs(mean - mc_mean), 5 * mc_se + 1e-12)


@pytest.mark.parametrize("sigma", SIGMAS)
def test_mean_matches_adaptive_quadrature(sigma):
    def integrand(z, mu):
        return expit(mu + sigma * z) * norm.pdf(z)

    expected = []
    for mu in MUS:
        # split where expit(mu + sigma * z) changes fastest
        z0 = np.clip(-mu / sigma, -12, 12)
        expected.append(
            quad(integrand, -12, z0, args=(mu,), epsabs=1e-13)[0]
            + quad(integrand, z0, 12, args=(mu,), epsabs=1e-13)[0]
        )
    dist = _logitnormal(MUS, sigma)
    np.testing.assert_allclose(dist.mean(), expected, rtol=0, atol=1e-10)
    # a single row gives the same mean as the full batch
    assert dist[4].mean() == dist.mean()[4]
