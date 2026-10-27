"""The NGBoost LogNormal distribution and scores"""

from warnings import warn

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
        Z = (lT - self.mu) / self.sigma

        D_uncens = np.zeros((self.mu.shape[0], 2))
        D_uncens[:, 0] = (self.mu - lT) / (self.sigma**2)
        D_uncens[:, 1] = 1 - ((self.mu - lT) ** 2) / (self.sigma**2)

        D_cens = np.zeros((self.mu.shape[0], 2))
        D_cens[:, 0] = -sp.stats.norm.pdf(lT, loc=self.mu, scale=self.sigma) / (
            1 - self.dist.cdf(T) + self.eps
        )
        D_cens[:, 1] = -Z * sp.stats.norm.pdf(Z) / (1 - self.dist.cdf(T) + self.eps)

        return (1 - E) * D_cens + E * D_uncens

    def metric(self):
        FI = np.zeros((self.mu.shape[0], 2, 2))
        FI[:, 0, 0] = 1 / (self.sigma**2) + self.eps
        FI[:, 1, 1] = 2
        return FI


class LogNormalCRPScoreCensored(CRPScore):
    def score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        lT = np.log(T)
        Z = (lT - self.mu) / self.sigma

        crps_uncens = self.sigma * (
            Z * (2 * sp.stats.norm.cdf(Z) - 1)
            + 2 * sp.stats.norm.pdf(Z)
            - 1 / np.sqrt(np.pi)
        )
        crps_cens = self.sigma * (
            Z * sp.stats.norm.cdf(Z) ** 2
            + 2 * sp.stats.norm.cdf(Z) * sp.stats.norm.pdf(Z)
            - sp.stats.norm.cdf(np.sqrt(2) * Z) / np.sqrt(np.pi)
        )
        return (1 - E) * crps_cens + E * crps_uncens

    def d_score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        lT = np.log(T)
        Z = (lT - self.mu) / self.sigma

        D = np.zeros((self.mu.shape[0], 2))
        # the censored CRPS is scale * integral of Phi(z)**2 up to Z
        D[:, 0] = E * -(2 * sp.stats.norm.cdf(Z) - 1) + (1 - E) * -(
            sp.stats.norm.cdf(Z) ** 2
        )
        D[:, 1] = self.score(Y) + (lT - self.mu) * D[:, 0]
        return D

    def metric(self):
        I = np.zeros((self.mu.shape[0], 2, 2))
        I[:, 0, 0] = 2
        I[:, 1, 1] = self.sigma**2
        I /= 2 * np.sqrt(np.pi) * self.sigma[:, np.newaxis, np.newaxis]
        return I


class LogNormal(RegressionDistn):
    """
    Implements the log-normal distribution for NGBoost.

    If Y is log-normal, log(Y) is normal with mean mu and standard deviation sigma.
    The params property gives the parameters with the names of scipy.stats.lognorm:
    s = sigma and scale = exp(mu). The attributes mu and sigma give mu and sigma.
    This distribution has both LogScore and CRPScore implemented
    for it and both work for right-censored data.
    """

    n_params = 2
    censored_scores = [LogNormalLogScoreCensored, LogNormalCRPScoreCensored]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params
        self.mu = params[0]
        self.sigma = np.exp(params[1])
        self.dist = dist(s=self.sigma, scale=np.exp(self.mu))
        self.eps = 1e-5

    @property
    def loc(self):
        warn(
            "LogNormal.loc is deprecated and will be removed in a future release. "
            "It is mu, the mean of log(Y). Use LogNormal.mu instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.mu

    @property
    def scale(self):
        warn(
            "LogNormal.scale is deprecated and will be removed in a future release. "
            "It is sigma, the standard deviation of log(Y), and it is not "
            "params['scale'], which is exp(mu) as in scipy.stats.lognorm. "
            "Use LogNormal.sigma or params['s'] instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.sigma

    # should implmenent a `sample()` method

    @property
    def params(self):
        return {"s": self.sigma, "scale": np.exp(self.mu)}

    def fit(Y):
        m, s = sp.stats.norm.fit(np.log(Y))
        return np.array([m, np.log(s)])
