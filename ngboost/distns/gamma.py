"""The NGBoost Gamma distribution and scores"""
import numpy as np
import scipy as sp
from scipy.stats import gamma as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class GammaLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2))
        # d(-log(PDF))/dalpha
        D[:, 0] = self.alpha * (
            sp.special.digamma(self.alpha) - np.log(self.eps + self.beta * Y)
        )
        # d(-log(PDF))/dbeta
        D[:, 1] = (self.beta * Y) - self.alpha
        return D

    def metric(self):
        FI = np.zeros((self.alpha.shape[0], 2, 2))
        FI[:, 0, 0] = self.alpha**2 * sp.special.polygamma(1, self.alpha)
        FI[:, 1, 1] = self.alpha
        FI[:, 0, 1] = -self.alpha
        FI[:, 1, 0] = -self.alpha
        return FI


class Gamma(RegressionDistn):
    n_params = 2
    scores = [GammaLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.alpha = np.exp(params[0])
        self.beta = np.exp(params[1])
        self.dist = dist(
            a=self.alpha, loc=np.zeros_like(self.alpha), scale=1 / self.beta
        )
        self.eps = 1e-10

    def fit(Y):
        a, _, scale = dist.fit(Y, floc=0)
        return np.array([np.log(a), np.log(1 / scale)])

    def sample(self, m):
        return np.array([self.rvs() for _ in range(m)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"alpha": self.alpha, "beta": self.beta}
