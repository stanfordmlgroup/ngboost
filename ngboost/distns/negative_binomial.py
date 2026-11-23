"""The NGBoost negative binomial distribution, built with the SymPy factory.

Use for regression on overdispersed counts, where the variance is larger
than the mean: visits, claims, defects, species counts, words in a
document.  The Poisson distribution has variance equal to the mean.  The
negative binomial adds a dispersion parameter ``n`` and has variance
``mu + mu**2 / n``, so it approaches the Poisson as ``n`` grows.

The parameters are the mean ``mu`` and the dispersion ``n``, both on the
log scale.  ``n`` is the ``n`` of ``scipy.stats.nbinom`` (``size`` in R and
``theta`` in ``MASS::glm.nb``), and the ``p`` of scipy is ``n / (n + mu)``.
The Fisher Information has no closed form in ``n``, so the factory uses
its Monte Carlo metric.
"""

import numpy as np
import scipy.stats
import sympy as sp
from scipy.optimize import minimize

from ngboost.distns.sympy_utils import make_distribution

_mu, _n = sp.symbols("mu n", positive=True)
_y = sp.symbols("y", nonnegative=True, integer=True)

_score_expr = -(
    sp.loggamma(_y + _n)
    - sp.loggamma(_n)
    - sp.loggamma(_y + 1)
    + _n * sp.log(_n / (_n + _mu))
    + _y * sp.log(_mu / (_n + _mu))
)

# Without overdispersion the maximum likelihood estimate of n is infinite,
# so the initial fit keeps n in this range.
_MIN_N, _MAX_N = 1e-4, 1e6


def _negative_binomial_fit(Y):
    """Fit mu and n to the marginal counts by maximum likelihood.

    Starts from the method of moments estimates.
    """
    Y = np.asarray(Y, dtype=float)
    if np.any(Y < 0) or np.any(np.mod(Y, 1) != 0):
        raise ValueError("NegativeBinomial needs counts: integers that are >= 0.")

    mu = max(np.mean(Y), 1e-3)
    excess = np.var(Y) - mu
    n = mu**2 / excess if excess > 0 else _MAX_N
    start = np.log([mu, np.clip(n, _MIN_N, _MAX_N)])

    def nll(log_params):
        mu, n = np.exp(log_params)
        return -np.mean(scipy.stats.nbinom.logpmf(Y, n, n / (n + mu)))

    bounds = [(None, None), (np.log(_MIN_N), np.log(_MAX_N))]
    res = minimize(nll, start, method="L-BFGS-B", bounds=bounds)
    return res.x if np.all(np.isfinite(res.x)) else start


NegativeBinomial = make_distribution(
    params=[(_mu, True), (_n, True)],
    y=_y,
    score_expr=_score_expr,
    scipy_dist_cls=scipy.stats.nbinom,
    scipy_kwarg_map={"n": _n, "p": _n / (_n + _mu)},
    fit_fn=_negative_binomial_fit,
    name="NegativeBinomial",
)
