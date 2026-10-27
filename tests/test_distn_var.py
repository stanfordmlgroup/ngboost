import inspect
import pickle

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
from ngboost.manifold import manifold
from ngboost.scores import CRPScore, LogScore

# Internal parameters of three observations, one row for each parameter: loc,
# log scale and, for T, log df. The df of T are above 2, so its variance is finite.
PARAMS = np.array([[0.3, -1.0, 2.0], [0.7, -0.2, 0.1], [1.0, 1.5, 2.5]])

NORMAL_DISTS = [Normal, NormalFixedVar, NormalFixedMean]
T_DISTS = [T, TFixedDf, TFixedDfFixedVar, Cauchy, CauchyFixedVar]
DISTS = NORMAL_DISTS + T_DISTS


def name(dist):
    return dist.__name__


@pytest.mark.parametrize("dist", DISTS, ids=name)
def test_var_warns_and_returns_squared_scale(dist):
    d = dist(PARAMS[: dist.n_params])
    with pytest.warns(FutureWarning) as record:
        var, line = d.var, inspect.currentframe().f_lineno
    assert len(record) == 1
    # the warning names the line that reads var, not a line in ngboost
    assert record[0].filename == __file__
    assert record[0].lineno == line
    message = str(record[0].message)
    assert message.startswith(f"{dist.__name__}.var is deprecated.")
    assert "var will be the var() method of scipy.stats" in message
    assert "Use dist.scale ** 2 for the current values." in message
    assert ("It is the squared scale, not the variance." in message) == (
        dist in T_DISTS
    )
    # the values from before the deprecation
    assert isinstance(var, np.ndarray)
    np.testing.assert_array_equal(var, d.scale**2)


FIT_CASES = [(dist, LogScore) for dist in DISTS] + [
    (dist, CRPScore) for dist in NORMAL_DISTS
]


@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize(
    "dist,score",
    FIT_CASES,
    ids=[f"{d.__name__}_{s.__name__}" for d, s in FIT_CASES],
)
def test_ngboost_gives_no_var_warning(dist, score):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    Y = X[:, 0] + rng.normal(size=100)
    scored = manifold(score, dist)(PARAMS[: dist.n_params])
    scored.total_score(Y[:3])
    scored.grad(Y[:3])
    ngb = NGBRegressor(Dist=dist, Score=score, n_estimators=5, verbose=False)
    ngb.fit(X, Y)
    pred = ngb.pred_dist(X[:5])
    loaded = pickle.loads(pickle.dumps(ngb))
    np.testing.assert_array_equal(loaded.pred_dist(X[:5]).loc, pred.loc)
    np.testing.assert_array_equal(loaded.pred_dist(X[:5]).scale, pred.scale)


# This test records what the release that removes the deprecated var attribute
# must give: dist.var() is the var() method of scipy.stats. It fails with
# TypeError while the attribute exists. Remove the xfail mark with the attribute.
@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="dist.var() works once the deprecated var attribute is removed",
)
@pytest.mark.filterwarnings("ignore:.*var is deprecated:FutureWarning")
@pytest.mark.parametrize("dist", DISTS, ids=name)
def test_future_var_is_scipy_var(dist):
    d = dist(PARAMS[: dist.n_params])
    # assert_array_equal and assert_allclose treat nan, the Cauchy variance, as equal
    np.testing.assert_array_equal(d.var(), d.dist.var())
    np.testing.assert_allclose(d.var(), d.std() ** 2)
