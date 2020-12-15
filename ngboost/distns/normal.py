"""The NGBoost Normal distribution and scores"""
from jax.ops import index_update, index
import jax.numpy as np

import scipy as sp
from jax.scipy.stats import norm

from ngboost.distns.distn import RegressionDistn, Parameter
from ngboost.scores import CRPScore, LogScore


class NormalLogScore(LogScore):
    pass

    @classmethod
    def metric(cls, _params):
        loc, scale = Normal.params_to_user(_params).values()
        var = scale ** 2

        FI = np.zeros((len(var), 2, 2))
        FI = index_update(FI, index[:, 0, 0], 1 / var)
        FI = index_update(FI, index[:, 1, 1], 2)
        return FI


class Normal(RegressionDistn):
    """
    Implements the normal distribution for NGBoost.

    The normal distribution has two parameters, loc and scale, which are
    the mean and standard deviation, respectively.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    scores = [NormalLogScore]

    parametrization = {
        "loc": Parameter(),
        "scale": Parameter(min=0),
    }

    @classmethod
    def cdf(cls, Y, loc, scale):
        return norm.cdf(Y, loc=loc, scale=scale)

    ### Inadvisably automatable?
    def mean(self):  # gives us Normal.mean() required for RegressionDist.predict()
        loc, scale = self.params.values()
        return loc

    ### Automatable?
    def sample(self, m):  # automate based on cdf?
        return np.array([sp.stats.norm.rvs(**self.params) for i in range(m)])

    @classmethod
    def fit(cls, Y):  # automate based on cdf?
        return cls.params_to_internal(*sp.stats.norm.fit(Y))
