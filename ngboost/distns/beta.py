"""The NGBoost Beta distribution — built via the SymPy factory.

Use for regression on bounded (0, 1) outcomes: proportions, rates,
probabilities, fractions, percentages.  Examples: click-through rates,
exam scores as fractions, market share, fraction of defective items.

This is the simplest factory pattern: provide a ``sympy.stats``
distribution and a ``scipy.stats`` class, and the factory auto-derives
score, gradient, Fisher Information, fit, sample, quantiles, etc.
"""

import scipy.stats
import sympy as sp
import sympy.stats as symstats

from ngboost.distns.sympy_utils import make_distribution

_alpha, _beta, _y = sp.symbols("alpha beta y", positive=True)

Beta = make_distribution(
    params=[(_alpha, True), (_beta, True)],
    y=_y,
    sympy_dist=symstats.Beta("Y", _alpha, _beta),
    scipy_dist_cls=scipy.stats.beta,
    scipy_kwarg_map={"a": _alpha, "b": _beta},
    name="Beta",
)
