"""The NGBoost Logit-Normal distribution — built via the SymPy factory.

Regression on bounded (0, 1) outcomes where the latent process is
approximately Gaussian after a logit transform.

Differs from Beta in that it assumes ``logit(Y) ~ Normal(mu, sigma)``,
giving different tail behavior.  More natural when the generative process
involves a logistic transform of latent features.

Uses a **manual score expression** (no ``sympy.stats`` equivalent) with the
``make_distribution`` factory.  Fisher Information is computed via Monte Carlo.
"""

import numpy as np
import sympy as sp

from ngboost.distns.sympy_utils import make_distribution

_mu, _sigma, _y = sp.symbols("mu sigma y", positive=True)
_logit_y = sp.log(_y / (1 - _y))

_score_expr = (
    sp.Rational(1, 2) * sp.log(2 * sp.pi)
    + sp.log(_sigma)
    + (_logit_y - _mu) ** 2 / (2 * _sigma**2)
    + sp.log(_y)
    + sp.log(1 - _y)
)


def _sample_logitnormal(self, m):
    """Sample by transforming Normal samples through logistic."""
    mu, sigma = np.squeeze(self.mu), np.squeeze(self.sigma)
    return np.array(
        [1.0 / (1.0 + np.exp(-np.random.normal(mu, sigma))) for _ in range(m)]
    )


LogitNormal = make_distribution(
    params=[(_mu, False), (_sigma, True)],  # mu: identity, sigma: log link
    y=_y,
    score_expr=_score_expr,
    sample_fn=_sample_logitnormal,
    name="LogitNormal",
)
