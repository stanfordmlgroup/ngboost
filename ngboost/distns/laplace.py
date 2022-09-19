"""The NGBoost Laplace distribution and scores"""
import numpy as np
from scipy.stats import laplace as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore


class LaplaceLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2))
        D[:, 0] = np.sign(self.loc - Y) / self.scale
        D[:, 1] = 1 - np.abs(self.loc - Y) / self.scale
        return D

    def metric(self):
        FI = np.zeros((self.loc.shape[0], 2, 2))
        FI[:, 0, 0] = 1 / self.scale ** 2
        FI[:, 1, 1] = 1
        return FI


class LaplaceCRPScore(CRPScore):
    def score(self, Y):
        return (
            np.abs(Y - self.loc)
            + np.exp(-np.abs(Y - self.loc) / self.scale) * self.scale
            - 0.75 * self.scale
        )

    def d_score(self, Y):
        D = np.zeros((len(Y), 2))
        D[:, 0] = np.sign(self.loc - Y) * (
            1 - np.exp(-np.abs(Y - self.loc) / self.scale)
        )
        D[:, 1] = np.exp(-np.abs(Y - self.loc) / self.scale) * (
            self.scale + np.abs(Y - self.loc)
        )
        return D

    def metric(self):
        FI = np.zeros((self.loc.shape[0], 2, 2))
        FI[:, 0, 0] = 0.5 / self.scale
        FI[:, 1, 1] = 0.25 * self.scale
        return FI


class Laplace(RegressionDistn):

    n_params = 2
    scores = [LaplaceLogScore, LaplaceCRPScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.loc = params[0]
        self.logscale = params[1]
        self.scale = np.exp(params[1])
        self.dist = dist(loc=self.loc, scale=self.scale)

    def fit(Y):
        m, s = dist.fit(Y)
        return np.array([m, np.log(s)])

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}
