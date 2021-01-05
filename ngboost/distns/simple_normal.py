"""The NGBoost Normal distribution and scores"""
import jax.numpy as np

import scipy as sp
from jax.scipy.stats import norm

from ngboost.distns.distn import RegressionDistn, Parameter
from ngboost.scores import LogScore


class Normal(RegressionDistn):
    """
    Implements the normal distribution for NGBoost.

    The normal distribution has two parameters, loc and scale, which are
    the mean and standard deviation, respectively.
    This distribution has LogScore implemented for it.
    """

    parametrization = {
        "loc": Parameter(),
        "scale": Parameter(min=0),
    }  # instances will be initialized with a dict {loc: [...], scale: [...]}

    @classmethod
    def cdf(cls, Y, loc, scale):
        return norm.cdf(Y, loc=loc, scale=scale)

    def predict(self):  # automate based on self.sample?
        loc, scale = self.params.values()
        return loc

    def sample(self, m):  # automate based on cdf?
        return np.array([sp.stats.norm.rvs(**self.params) for i in range(m)])
