"""The NGBoost Half-Normal distribution and scores"""
import numpy as np
from scipy.stats import halfnorm as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class HalfNormalLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 1))
        D[:, 0] = (self.scale**2 - Y**2) / self.scale**2  # dL/d(log(scale))
        return D

    def metric(self):
        FI = 2 * np.ones_like(self.scale)
        return FI[:, np.newaxis, np.newaxis]


class HalfNormal(RegressionDistn):
    """
    Implements the Half-Normal distribution for NGBoost.

    The Half-Normal distribution has one parameter, scale.
    The scipy loc parameter is held constant at zero for this implementation.
    LogScore is supported for the Half-Normal distribution.
    """

    n_params = 1
    scores = [HalfNormalLogScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.scale = np.exp(params[0])  # scale (sigma)
        self.dist = dist(loc=0, scale=self.scale)

    def fit(Y):
        _loc, scale = dist.fit(Y, floc=0)  # loc held constant
        return np.array([np.log(scale)])

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": np.zeros(shape=self.scale.shape), "scale": self.scale}
