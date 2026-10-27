import numpy as np
import pytest

from ngboost import NGBRegressor
from ngboost.distns import (
    Cauchy,
    Normal,
    NormalFixedMean,
    NormalFixedVar,
    T,
    TFixedDf,
    TFixedDfFixedVar,
)
from ngboost.distns.cauchy import CauchyFixedVar

# Internal parameters of three observations, one row for each parameter: loc,
# log scale and, for T, log df. The df of T are above 2, so its variance is finite.
PARAMS = np.array([[0.3, -1.0, 2.0], [0.7, -0.2, 0.1], [1.0, 1.5, 2.5]])

NORMAL_DISTS = [Normal, NormalFixedVar, NormalFixedMean]
T_DISTS = [T, TFixedDf, TFixedDfFixedVar]
DISTS = NORMAL_DISTS + T_DISTS + [Cauchy, CauchyFixedVar]


def name(dist):
    return dist.__name__


@pytest.mark.parametrize("dist", DISTS, ids=name)
def test_var_is_scipy_var(dist):
    d = dist(PARAMS[: dist.n_params])
    # assert_array_equal and assert_allclose treat nan, the Cauchy variance, as equal
    np.testing.assert_array_equal(d.var(), d.dist.var())
    np.testing.assert_allclose(d.var(), d.std() ** 2)


@pytest.mark.parametrize("dist", NORMAL_DISTS, ids=name)
def test_var_of_normal_is_squared_scale(dist):
    d = dist(PARAMS[: dist.n_params])
    np.testing.assert_allclose(d.var(), d.scale**2)


@pytest.mark.parametrize("dist", T_DISTS, ids=name)
def test_var_of_t_is_variance(dist):
    d = dist(PARAMS[: dist.n_params])
    np.testing.assert_allclose(d.var(), d.scale**2 * d.df / (d.df - 2))


def test_pred_dist_var():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    Y = X[:, 0] + rng.normal(size=100)
    ngb = NGBRegressor(n_estimators=5, verbose=False).fit(X, Y)
    pred = ngb.pred_dist(X[:5])
    np.testing.assert_allclose(pred.var(), pred.scale**2)
