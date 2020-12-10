"""The NGBoost Normal distribution and scores"""
from jax import grad
import jax.numpy as np

import scipy as sp
from jax.scipy.stats import norm

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore


class NormalLogScore(LogScore):

    pass

    # def metric(self):
    #     FI = np.zeros((self.var.shape[0], 2, 2))
    #     FI[:, 0, 0] = 1 / self.var
    #     FI[:, 1, 1] = 2
    #     return FI


class Normal(RegressionDistn):
    """
    Implements the normal distribution for NGBoost.

    The normal distribution has two parameters, loc and scale, which are
    the mean and standard deviation, respectively.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    scores = [NormalLogScore]

    ### Parametrization
    @classmethod
    def untransform_params(cls, transformed_params):
        loc = transformed_params[0]
        scale = np.exp(transformed_params[1])
        return dict(loc=loc, scale=scale)

    @classmethod
    def transform_params(cls, loc, scale):
        return np.array([loc, np.log(scale)])

    ### Distribution
    @classmethod
    def cdf(cls, Y, loc, scale):
        return norm.cdf(Y, loc=loc, scale=scale)

    # @classmethod
    # def pdf(cls, Y, loc, scale):
    #     return norm.pdf(Y, loc=loc, scale=scale)

    # @classmethod
    # def logpdf(cls, Y, loc, scale):
    #     return norm.logpdf(Y, loc=loc, scale=scale)

    ### Inadvisably automatable?
    def mean(self):  # gives us Normal.mean() required for RegressionDist.predict()
        loc, scale = self._params
        return loc

    ### Automatable?
    def sample(self, m):  # automate based on cdf?
        return np.array([sp.stats.norm.rvs(**self._params) for i in range(m)])

    @classmethod
    def fit(cls, Y):  # automate based on cdf?
        return cls.transform_params(*sp.stats.norm.fit(Y))
