"""The NGBoost Beta-Binomial distributions — built via the SymPy factory.

Two variants:

- **BetaBinomial** — count regression with overdispersion where ``n`` (number
  of trials) is fixed and known.  Uses ``make_distribution`` with
  ``extra_params=[(_n, 1)]``.
- **BetaBinomialEstN** — same model but ``n`` is also optimised by NGBoost.
  Useful when the total number of trials is unknown and only the count
  outcome is observed.  Prefer ``BetaBinomial`` when ``n`` is known.
"""

import numpy as np
import scipy.stats
import sympy as sp

from ngboost.distns.sympy_utils import make_distribution

# ---- Shared symbols and score expression ----

_alpha, _beta = sp.symbols("alpha beta", positive=True)
_y, _n = sp.symbols("y n", positive=True, integer=True)

_score_expr = -(
    sp.loggamma(_n + 1)
    - sp.loggamma(_y + 1)
    - sp.loggamma(_n - _y + 1)
    + sp.loggamma(_y + _alpha)
    + sp.loggamma(_n - _y + _beta)
    - sp.loggamma(_n + _alpha + _beta)
    + sp.loggamma(_alpha + _beta)
    - sp.loggamma(_alpha)
    - sp.loggamma(_beta)
)


# =====================================================================
# BetaBinomial — fixed n (non-optimised)
# =====================================================================


def _betabinom_fit(Y, n=1):
    """Estimate initial alpha, beta from data via method of moments.

    Uses overdispersion parameter rho = (Var/BinomVar - 1) / (n - 1)
    to recover the concentration alpha + beta, then splits by p_hat.
    """
    Y = np.asarray(Y, dtype=float)
    p_hat = np.clip(np.mean(Y) / n, 0.01, 0.99)
    var_hat = np.var(Y)
    binom_var = n * p_hat * (1 - p_hat)
    if binom_var > 0 and n > 1:
        rho = np.clip((var_hat / binom_var - 1) / (n - 1), 0.01, 0.99)
    else:
        rho = 0.5
    concentration = 1.0 / rho - 1
    alpha = np.clip(p_hat * concentration, 1e-4, 1e4)
    beta = np.clip((1 - p_hat) * concentration, 1e-4, 1e4)
    return np.array([np.log(alpha), np.log(beta)])


def _betabinom_sample(self, m):
    alpha = np.squeeze(self.alpha)
    beta = np.squeeze(self.beta)
    return np.array(
        [np.random.binomial(int(self.n), np.random.beta(alpha, beta)) for _ in range(m)]
    )


def _betabinom_mean(self):
    return self.n * self.alpha / (self.alpha + self.beta)


BetaBinomial = make_distribution(
    params=[(_alpha, True), (_beta, True)],
    y=_y,
    score_expr=_score_expr,
    extra_params=[(_n, 1)],
    scipy_dist_cls=scipy.stats.betabinom,
    scipy_kwarg_map={"n": _n, "a": _alpha, "b": _beta},
    fit_fn=_betabinom_fit,
    sample_fn=_betabinom_sample,
    mean_fn=_betabinom_mean,
    name="BetaBinomial",
)


# =====================================================================
# BetaBinomialEstN — n is also optimised (log-transformed)
# =====================================================================


def _betabinom_estn_fit(Y):
    """Estimate initial alpha, beta, n from data.

    Initialises n well above max(Y) to ensure loggamma(n - y + 1) is
    defined for all observed counts.
    """
    Y = np.asarray(Y, dtype=float)
    n_init = max(2 * np.max(Y), np.max(Y) + 10)
    p_hat = np.clip(np.mean(Y) / n_init, 0.01, 0.99)
    var_hat = np.var(Y)
    binom_var = n_init * p_hat * (1 - p_hat)
    if binom_var > 0 and n_init > 1:
        rho = np.clip((var_hat / binom_var - 1) / (n_init - 1), 0.01, 0.99)
    else:
        rho = 0.5
    concentration = 1.0 / rho - 1
    alpha = np.clip(p_hat * concentration, 1e-4, 1e4)
    beta = np.clip((1 - p_hat) * concentration, 1e-4, 1e4)
    return np.array([np.log(alpha), np.log(beta), np.log(n_init)])


def _betabinom_estn_sample(self, m):
    alpha = np.squeeze(self.alpha)
    beta = np.squeeze(self.beta)
    # Use floor (not round) so sampled y <= floor(n) <= n_continuous,
    # keeping loggamma(n - y + ...) safe during MC metric evaluation.
    n_int = np.maximum(1, np.floor(np.squeeze(self.n)).astype(int))
    return np.array(
        [np.random.binomial(n_int, np.random.beta(alpha, beta)) for _ in range(m)]
    )


def _betabinom_estn_mean(self):
    return self.n * self.alpha / (self.alpha + self.beta)


BetaBinomialEstN = make_distribution(
    params=[(_alpha, True), (_beta, True), (_n, True)],
    y=_y,
    score_expr=_score_expr,
    fit_fn=_betabinom_estn_fit,
    sample_fn=_betabinom_estn_sample,
    mean_fn=_betabinom_estn_mean,
    name="BetaBinomialEstN",
)
