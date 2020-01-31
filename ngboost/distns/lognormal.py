import scipy as sp
import numpy as np
from scipy.stats import lognorm as dist
from ngboost.distns import SurvivalDistn
from ngboost.scores import LogScore, CRPScore


class LogNormalLogScore(LogScore):

    def score(self, Y):
        E = Y['Event']
        T = Y['Time']
        cens = (1-E) * np.log(1 - self.dist.cdf(T) + self.eps)
        uncens = E * self.dist.logpdf(T)
        return -(cens + uncens)

    def d_score(self, Y):
        E = Y['Event'][:,np.newaxis]
        T = Y['Time']
        lT = np.log(T)
        Z = (lT - self.loc) / self.scale

        D_uncens = np.zeros((self.loc.shape[0], 2))
        D_uncens[:, 0] = (self.loc - lT) / (self.scale ** 2)
        D_uncens[:, 1] = 1 - ((self.loc - lT) ** 2) / (self.scale ** 2)

        D_cens = np.zeros((self.loc.shape[0], 2))
        D_cens[:, 0] = -sp.stats.norm.pdf(lT, loc=self.loc, scale=self.scale) / \
                        (1 - self.dist.cdf(T) + self.eps)
        D_cens[:, 1] = -Z * sp.stats.norm.pdf(lT, loc=self.loc, scale=self.scale) / \
                        (1 - self.dist.cdf(T) + self.eps)
        D_cens[:, 0] = -sp.stats.norm.pdf(lT, loc=self.loc, scale=self.scale) / \
                        (1 - self.dist.cdf(T) + self.eps)
        D_cens[:, 1] = -Z * sp.stats.norm.pdf(lT, loc=self.loc, scale=self.scale) / \
                        (1 - self.dist.cdf(T) + self.eps)

        return (1-E) * D_cens + E * D_uncens

    def metric(self):
        FI = np.zeros((self.loc.shape[0], 2, 2))
        FI[:, 0, 0] = 1/(self.scale ** 2) + self.eps
        FI[:, 1, 1] = 2
        return FI

class LogNormalCRPScore(CRPScore):

    def score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        lT = np.log(T)
        Z = (lT - self.loc) / self.scale

        crps_uncens = (self.scale * (Z * (2 * sp.stats.norm.cdf(Z) - 1) + \
                      2 * sp.stats.norm.pdf(Z) - 1 / np.sqrt(np.pi)))
        crps_cens = self.scale * (Z * sp.stats.norm.cdf(Z) ** 2 + \
                    2 * sp.stats.norm.cdf(Z) * sp.stats.norm.pdf(Z) - \
                    sp.stats.norm.cdf(np.sqrt(2) * Z) / np.sqrt(np.pi))
        return (1-E) * crps_cens + E * crps_uncens

    def d_score(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        lT = np.log(T)
        Z = (lT - self.loc) / self.scale

        D = np.zeros((self.loc.shape[0], 2))
        D[:, 0] = E * -(2 * sp.stats.norm.cdf(Z) - 1)
        D[:, 0] = (1-E) * -(sp.stats.norm.cdf(Z) ** 2 + \
                            2 * Z * sp.stats.norm.cdf(Z) * sp.stats.norm.pdf(Z) + \
                            2 * sp.stats.norm.pdf(Z) ** 2 - \
                            2 * sp.stats.norm.cdf(Z) * sp.stats.norm.pdf(Z) ** 2 - \
                            np.sqrt(2/np.pi) * sp.stats.norm.pdf(np.sqrt(2) * Z))
        D[:, 1] = self.crps(Y) + (lT - self.loc) * D[:, 0]
        return D

    def metric(self):
        I = 1/(2*np.sqrt(np.pi)) * np.diag(np.array([1, self.scale ** 2 / 2]))
        return I + 1e-4 * np.eye(2)


class LogNormal(SurvivalDistn):

    n_params = 2
    scores = [LogNormalLogScore, LogNormalCRPScore]

    def __init__(self, params):
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.dist = dist(s=self.scale, scale=np.exp(self.loc))
        self.eps = 1e-5
    
    def __getattr__(self, name): 
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None
    
    @property
    def params(self):
        return {'loc':self.loc, 'scale':self.scale}
    def fit(Y):
        T = Y["Time"]
        m, s = sp.stats.norm.fit(np.log(T))
        return np.array([m, np.log(s)])
