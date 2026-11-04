"""Survival log scores must retain information in the right tail."""

import numpy as np
import pytest
import scipy.special
from scipy.stats import expon, lognorm

from ngboost.distns import Exponential, LogNormal
from ngboost.distns.utils import SurvivalDistnClass
from ngboost.helpers import Y_from_censored
from ngboost.manifold import manifold
from ngboost.scores import LogScore


@pytest.mark.parametrize("dist", [Exponential, LogNormal])
@pytest.mark.parametrize("event", [False, True])
def test_survival_logscore_tails(dist, event):  # pylint: disable=too-many-locals
    if dist is Exponential:
        params = np.log([[1.7, 1.7, 1.7, 1.7]])
        time = 1.7 * np.array([0.2, 5, 30, 1000])
        scipy_dist = expon(scale=1.7)
        expected_grad = (event - time / 1.7)[:, None]
    else:
        loc, scale = 0.3, 1.7
        z = np.array([0.2, 5, 10, 40])
        time = np.exp(loc + scale * z)
        params = np.repeat([[loc], [np.log(scale)]], len(z), axis=1)
        scipy_dist = lognorm(s=scale, scale=np.exp(loc))
        # erfcx gives the normal inverse Mills ratio without subtracting a CDF.
        mills = np.sqrt(2 / np.pi) / scipy.special.erfcx(  # pylint: disable=no-member
            z / np.sqrt(2)
        )
        if event:
            expected_grad = np.column_stack((-z / scale, 1 - z**2))
        else:
            expected_grad = np.column_stack((-mills / scale, -z * mills))

    y = Y_from_censored(time, np.full(len(time), event))
    score_dist = manifold(LogScore, SurvivalDistnClass(dist))
    obj = score_dist(params)
    expected_score = -(scipy_dist.logpdf(time) if event else scipy_dist.logsf(time))
    np.testing.assert_allclose(obj.score(y), expected_score, rtol=1e-12)
    np.testing.assert_allclose(obj.d_score(y), expected_grad, rtol=1e-11)

    for i in range(dist.n_params):
        step = np.zeros_like(params)
        step[i] = 1e-5
        numerical = (
            score_dist(params + step).score(y) - score_dist(params - step).score(y)
        ) / 2e-5
        np.testing.assert_allclose(
            obj.d_score(y)[:, i], numerical, rtol=1e-7, atol=1e-7
        )


def test_survival_logscore_mixed_events():
    time = np.array([1.0, 2.0, 1000.0, 1000.0])
    events = np.array([True, False, True, False])
    y = Y_from_censored(time, events)
    for dist, params in [
        (Exponential, np.zeros((1, 4))),
        (LogNormal, np.zeros((2, 4))),
    ]:
        score_dist = manifold(LogScore, SurvivalDistnClass(dist))
        combined = score_dist(params)
        for i in range(len(time)):
            single = score_dist(params[:, i : i + 1])
            np.testing.assert_allclose(
                combined.score(y)[i], single.score(y[i : i + 1])[0]
            )
            np.testing.assert_allclose(
                combined.d_score(y)[i], single.d_score(y[i : i + 1])[0]
            )
