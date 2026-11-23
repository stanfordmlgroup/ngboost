"""The NGBoost Logit-Normal distribution — built via the SymPy factory.

Regression on bounded (0, 1) outcomes where the latent process is
approximately Gaussian after a logit transform.

Differs from Beta in that it assumes ``logit(Y) ~ Normal(mu, sigma)``,
giving different tail behavior.  More natural when the generative process
involves a logistic transform of latent features.

Uses a **manual score expression** (no ``sympy.stats`` equivalent) with the
``make_distribution`` factory.  Fisher Information is computed via Monte Carlo.
The mean has no closed form and is computed by deterministic quadrature.
"""

import numpy as np
import sympy as sp
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, ndtr  # pylint: disable=no-name-in-module

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


# The mean E[expit(mu + sigma * Z)] with Z ~ N(0, 1) has no closed form.  It is
# computed with one of two quadrature rules, chosen by sigma:
#
# - sigma <= 1: 64-node Gauss-Hermite quadrature over Z.
# - sigma > 1: expit(mu + sigma * z) is close to a step in z, and Gauss-Hermite
#   loses accuracy (error 3e-3 at sigma = 10).  Instead write the mean as
#   P(L < mu + sigma * Z) = E[Phi((mu - L) / sigma)] with L standard logistic,
#   and integrate over L with the trapezoidal rule, which converges
#   geometrically for this smooth integrand.
#
# On its own side of sigma = 1, each rule agrees with a high-precision
# reference to about 1e-14.
_GH_NODES, _GH_WEIGHTS = hermegauss(64)
_GH_WEIGHTS = _GH_WEIGHTS / np.sqrt(2 * np.pi)
# step 0.5 on [-40, 40]; the logistic mass outside is 2 * expit(-40), about 8e-18
_LOGIS_NODES = np.linspace(-40, 40, 161)
_LOGIS_WEIGHTS = 0.5 * expit(_LOGIS_NODES) * expit(-_LOGIS_NODES)


def _mean_logitnormal(self):
    """Mean of each row by quadrature (see the note above), so it is deterministic."""
    mu, sigma = np.broadcast_arrays(self.mu, self.sigma)
    mean = np.empty(mu.shape)
    narrow = sigma <= 1
    mu_n, sigma_n = mu[narrow], sigma[narrow]
    mean[narrow] = sum(
        w * expit(mu_n + sigma_n * z) for z, w in zip(_GH_NODES, _GH_WEIGHTS)
    )
    mu_w, sigma_w = mu[~narrow], sigma[~narrow]
    mean[~narrow] = sum(
        w * ndtr((mu_w - t) / sigma_w) for t, w in zip(_LOGIS_NODES, _LOGIS_WEIGHTS)
    )
    return mean


LogitNormal = make_distribution(
    params=[(_mu, False), (_sigma, True)],  # mu: identity, sigma: log link
    y=_y,
    score_expr=_score_expr,
    sample_fn=_sample_logitnormal,
    mean_fn=_mean_logitnormal,
    name="LogitNormal",
)
