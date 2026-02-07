"""The NGBoost Beta-Bernoulli distribution — built via the SymPy factory.

Binary classification with calibrated uncertainty.  Unlike standard classifiers
that output a single probability, the Beta-Bernoulli gives a full *distribution
over the probability itself* via its Beta(alpha, beta) prior.

The predictive probability for class 1 is ``alpha / (alpha + beta)``.
When alpha and beta are both large, the model is confident.  When they're
small, there's high uncertainty.
"""

import numpy as np
import sympy as sp
import sympy.stats as symstats
from scipy.special import digamma

from ngboost.distns.sympy_utils import make_distribution

_alpha, _beta, _y = sp.symbols("alpha beta y")
_p = _alpha / (_alpha + _beta)


def _beta_bernoulli_fit(Y):
    """Estimate initial alpha, beta from binary data via digamma MLE."""
    p = np.clip(np.mean(Y), 0.01, 0.99)
    a, b = p * 2, (1 - p) * 2
    for _ in range(100):
        ab = a + b
        psi_ab = digamma(ab)
        a = np.clip(
            a * (np.mean(digamma(Y + a)) - psi_ab) / (digamma(a) - psi_ab + 1e-10),
            1e-4,
            1e4,
        )
        b = np.clip(
            b * (np.mean(digamma(1 - Y + b)) - psi_ab) / (digamma(b) - psi_ab + 1e-10),
            1e-4,
            1e4,
        )
    return np.array([np.log(a), np.log(b)])


BetaBernoulli = make_distribution(
    params=[(_alpha, True), (_beta, True)],
    y=_y,
    sympy_dist=symstats.Bernoulli("Y", _p),
    class_prob_exprs=[1 - _p, _p],
    fit_fn=_beta_bernoulli_fit,
    name="BetaBernoulli",
)

BetaBernoulliLogScore = BetaBernoulli.scores[0]
