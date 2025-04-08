"""The NGBoost Weibull distribution and scores"""
import numpy as np
from scipy.stats import weibull_min as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class WeibullLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2))
        shared_term = self.shape * ((Y / self.scale) ** self.shape - 1)
        D[:, 0] = shared_term * np.log(Y / self.scale) - 1
        D[:, 1] = -shared_term

        return D

    def metric(self):
        gamma = 0.5772156649  # Euler's constant
        FI = np.zeros((self.scale.shape[0], 2, 2))
        FI[:, 0, 0] = (np.pi**2 / 6) + (1 - gamma) ** 2
        FI[:, 1, 0] = -self.shape * (1 - gamma)
        FI[:, 0, 1] = FI[:, 1, 0]
        FI[:, 1, 1] = self.shape**2

        return FI


class Weibull(RegressionDistn):
    """
    Implements the Weibull distribution for NGBoost.

    The Weibull distribution has two parameters, shape and scale.
    The scipy loc parameter is held constant for this implementation.
    LogScore is supported for the Weibull distribution.
    """

    n_params = 2
    scores = [WeibullLogScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.shape = np.exp(params[0])  # shape (c)
        self.scale = np.exp(params[1])  # scale (labmda)
        self.dist = dist(c=self.shape, loc=0, scale=self.scale)

    def fit(Y):
        shape, _loc, scale = dist.fit(Y, floc=0)  # hold loc constant
        return np.array([np.log(shape), np.log(scale)])

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"shape": self.shape, "scale": self.scale}
