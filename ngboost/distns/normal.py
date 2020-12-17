"""The NGBoost Normal distribution and scores"""
from jax.ops import index_update, index
import jax.numpy as np
import numpy

import scipy as sp
from jax.scipy.stats import norm

from ngboost.distns.distn import RegressionDistn, Parameter
from ngboost.scores import CRPScore, LogScore


def get_params(_params):
    loc, scale = Normal.params_to_user(_params).values()
    var = scale ** 2
    return loc, scale, var


class NormalLogScore(LogScore):
    pass

    @classmethod
    def _score(cls, _params, Y):
        return -norm.logpdf(Y, **cls.params_to_user(_params))

    @classmethod
    def _d_score(cls, _params, Y):
        loc, scale, var = get_params(_params)

        D = np.zeros((len(Y), 2))
        D = index_update(D, index[:, 0], (loc - Y) / var)
        D = index_update(D, index[:, 1], 1 - ((loc - Y) ** 2) / var)
        return D

    @classmethod
    def _metric(cls, _params):
        loc, scale, var = get_params(_params)

        FI = np.zeros((len(var), 2, 2))
        FI = index_update(FI, index[:, 0, 0], 1 / var)
        FI = index_update(FI, index[:, 1, 1], 2)
        return FI

    @classmethod
    def _fit_marginal(cls, Y):
        loc, scale = sp.stats.norm.fit(Y)
        return cls.params_to_internal(loc=loc, scale=scale)


class NormalCRPScore(CRPScore):
    @classmethod
    def _score(cls, _params, Y):
        loc, scale, var = get_params(_params)

        Z = (Y - loc) / scale
        return scale * (
            Z * (2 * norm.cdf(Z) - 1) + 2 * norm.pdf(Z) - 1 / np.sqrt(np.pi)
        )

    @classmethod
    def _d_score(cls, _params, Y):
        loc, scale, var = get_params(_params)

        Z = (Y - loc) / scale
        D = np.zeros((len(Y), 2))
        D = index_update(D, index[:, 0], -(2 * norm.cdf(Z) - 1))
        D = index_update(D, index[:, 1], cls._score(_params, Y) + (Y - loc) * D[:, 0])
        return D

    @classmethod
    def _metric(cls, _params):
        loc, scale, var = get_params(_params)
        I = numpy.c_[
            2 * np.ones_like(var), np.zeros_like(var), np.zeros_like(var), var,
        ]
        I = I.reshape((var.shape[0], 2, 2))
        I = 1 / (2 * np.sqrt(np.pi)) * I
        return I

    @classmethod
    def _fit_marginal(cls, Y):
        loc, scale = sp.stats.norm.fit(Y)
        return cls.params_to_internal(loc=loc, scale=scale)


class Normal(RegressionDistn):
    """
    Implements the normal distribution for NGBoost.

    The normal distribution has two parameters, loc and scale, which are
    the mean and standard deviation, respectively.
    This distribution has LogScore implemented for it.
    """

    scores = [NormalLogScore, NormalCRPScore]

    parametrization = {
        "loc": Parameter(),
        "scale": Parameter(min=0),
    }

    @classmethod
    def cdf(cls, Y, loc, scale):
        return norm.cdf(Y, loc=loc, scale=scale)

    def predict(self):
        return self.params["loc"]

    def sample(self, m):
        return np.array([sp.stats.norm.rvs(**self.params) for i in range(m)])


class NormalFixedVarLogScore(LogScore):
    def score(Y, loc):
        return -norm.logpdf(Y, loc=loc, scale=1)

    # @classmethod
    # def _fit_marginal(cls, Y):
    #     loc, scale = sp.stats.norm.fit(Y)
    #     return cls.params_to_internal(loc=loc)


class NormalFixedVar(RegressionDistn):
    scores = [NormalFixedVarLogScore]

    parametrization = {
        "loc": Parameter(),
    }

    @classmethod
    def cdf(cls, Y, loc):
        return norm.cdf(Y, loc=loc, scale=1)

    def predict(self):
        return self.params["loc"]

    def sample(self, m):
        return np.array(
            [sp.stats.norm.rvs(loc=self.params["loc"], scale=1) for i in range(m)]
        )
