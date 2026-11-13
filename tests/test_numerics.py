import warnings

import numpy as np
import pytest

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.distns.normal import NormalLogScore
from ngboost.scores import CRPScore


def test_rejected_line_search_steps_do_not_warn():
    # issue #352: with plain gradients, CRPScore and a target in the thousands,
    # the first trial step of every line search overflowed in Normal
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = 1000 * (X[:, 0] + rng.normal(size=500))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ngb = NGBRegressor(
            Dist=Normal,
            Score=CRPScore,
            natural_gradient=False,
            n_estimators=20,
            verbose=False,
        ).fit(X, y)

    assert np.isfinite(ngb.pred_param(X)).all()


def test_fit_raises_clear_error_when_gradient_is_not_finite(monkeypatch):
    d_score = NormalLogScore.d_score

    def d_score_with_nan(self, Y):
        D = d_score(self, Y)
        D[0, 1] = np.nan
        return D

    monkeypatch.setattr(NormalLogScore, "d_score", d_score_with_nan)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] + rng.normal(size=100)

    with pytest.raises(ValueError, match="diverged at iteration 0"):
        NGBRegressor(n_estimators=5, verbose=False).fit(X, y)
