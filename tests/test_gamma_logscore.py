"""The Gamma score gradient must remain exact on its positive support."""

import numpy as np
import pytest
from scipy.stats import gamma

from ngboost.distns import Gamma
from ngboost.manifold import manifold
from ngboost.scores import LogScore


@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_gamma_logscore_positive_tail_gradient(alpha):
    beta = np.array([1.0, 1.0, 1e-100, 1e100, 0.5, 2.0])
    y = np.array([1e-20, 1e-200, 1e-100, 1e-100, 0.8, 3.0])
    params = np.array([np.full(len(y), np.log(alpha)), np.log(beta)])
    score_dist = manifold(LogScore, Gamma)
    obj = score_dist(params)
    # Closed-form digamma values avoid using the implementation's special function.
    psi = 1 - np.euler_gamma if alpha == 2 else -np.euler_gamma - 2 * np.log(2)
    expected = np.column_stack(
        (alpha * (psi - np.log(beta) - np.log(y)), beta * y - alpha)
    )
    np.testing.assert_allclose(obj.score(y), -gamma.logpdf(y, alpha, scale=1 / beta))
    np.testing.assert_allclose(obj.d_score(y), expected, rtol=1e-12, atol=1e-12)
    for i in range(2):
        step = np.zeros_like(params)
        step[i] = 1e-5
        numerical = (
            score_dist(params + step).score(y) - score_dist(params - step).score(y)
        ) / 2e-5
        np.testing.assert_allclose(
            obj.d_score(y)[:, i], numerical, rtol=1e-7, atol=1e-7
        )
