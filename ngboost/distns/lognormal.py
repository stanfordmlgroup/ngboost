"""The NGBoost LogNormal distribution and scores"""
import numpy as np
import scipy as sp
from scipy.stats import lognorm as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore


class LogNormalLogScoreCensored(LogScore):
    def score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        cens = (1 - E) * np.log(1 - self.dist.cdf(T) + self.eps)
        uncens = E * self.dist.logpdf(T)
        return -(cens + uncens)

    def d_score(self, Y):
        E = Y["Event"][:, np.newaxis]
        T = Y["Time"]
        lT = np.log(T)
        Z = (lT - self.loc) / self.scale

        D_uncens = np.zeros((self.loc.shape[0], 2))
        D_uncens[:, 0] = (self.loc - lT) / (self.scale ** 2)
        D_uncens[:, 1] = 1 - ((self.loc - lT) ** 2) / (self.scale ** 2)

        D_cens = np.zeros((self.loc.shape[0], 2))
        D_cens[:, 0] = -sp.stats.norm.pdf(lT, loc=self.loc, scale=self.scale) / (
            1 - self.dist.cdf(T) + self.eps
        )
        D_cens[:, 1] = (
            -Z
            * sp.stats.norm.pdf(lT, loc=self.loc, scale=self.scale)
            / (1 - self.dist.cdf(T) + self.eps)
        )

        return (1 - E) * D_cens + E * D_uncens

    def metric(self):
        FI = np.zeros((self.loc.shape[0], 2, 2))
        FI[:, 0, 0] = 1 / (self.scale ** 2) + self.eps
        FI[:, 1, 1] = 2
        return FI


class LogNormalCRPScoreCensored(CRPScore):
    def score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        lT = np.log(T)
        Z = (lT - self.loc) / self.scale

        crps_uncens = self.scale * (
            Z * (2 * sp.stats.norm.cdf(Z) - 1)
            + 2 * sp.stats.norm.pdf(Z)
            - 1 / np.sqrt(np.pi)
        )
        crps_cens = self.scale * (
            Z * sp.stats.norm.cdf(Z) ** 2
            + 2 * sp.stats.norm.cdf(Z) * sp.stats.norm.pdf(Z)
            - sp.stats.norm.cdf(np.sqrt(2) * Z) / np.sqrt(np.pi)
        )
        return (1 - E) * crps_cens + E * crps_uncens

    def d_score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        lT = np.log(T)
        Z = (lT - self.loc) / self.scale

        D = np.zeros((self.loc.shape[0], 2))
        D[:, 0] = E * -(2 * sp.stats.norm.cdf(Z) - 1) + (1 - E) * -(
            sp.stats.norm.cdf(Z) ** 2
            + 2 * Z * sp.stats.norm.cdf(Z) * sp.stats.norm.pdf(Z)
            + 2 * sp.stats.norm.pdf(Z) ** 2
            - 2 * sp.stats.norm.cdf(Z) * sp.stats.norm.pdf(Z) ** 2
            - np.sqrt(2 / np.pi) * sp.stats.norm.pdf(np.sqrt(2) * Z)
        )
        D[:, 1] = self.score(Y) + (lT - self.loc) * D[:, 0]
        return D

    def metric(self):
        I = np.zeros((self.loc.shape[0], 2, 2))
        I[:, 0, 0] = 2
        I[:, 1, 1] = self.scale ** 2
        I /= 2 * np.sqrt(np.pi)
        return I


class LogNormal(RegressionDistn):

    """
    Implements the log-normal distribution for NGBoost.

    The normal distribution has two parameters, s and scale (see scipy.stats.lognorm)
    This distribution has both LogScore and CRPScore implemented
    for it and both work for right-censored data.
    """

    n_params = 2
    censored_scores = [LogNormalLogScoreCensored, LogNormalCRPScoreCensored]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.dist = dist(s=self.scale, scale=np.exp(self.loc))
        self.eps = 1e-5

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    # should implmenent a `sample()` method

    @property
    def params(self):
        return {"s": self.scale, "scale": np.exp(self.loc)}

    def fit(Y):
        m, s = sp.stats.norm.fit(np.log(Y))
        return np.array([m, np.log(s)])
