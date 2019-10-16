import scipy as osp
import scipy.stats
import jax.numpy as np
import jax.scipy as sp
import jax.random as random


class LogLaplace(object):

    n_params = 2

    def __init__(self, params, temp_scale = 10.0):
        self.loc = params[0]
        self.scale = np.exp(params[1] / temp_scale) ** 0.5
        self.shp = self.loc.shape
        self.params = params

    def logpdf(self, Y):
        return sp.stats.laplace.logpdf(np.log(Y), loc=self.loc, scale=self.scale)

    def cdf(self, Y):
        return osp.stats.laplace.cdf(np.log(Y), loc=self.loc, scale=self.scale)

    def ppf(self, Q):
        return osp.stats.loglaplace.ppf(Q, c=1., loc=self.loc, scale=self.scale)

    def crps(self, Y):
        Z = (np.log(Y) - self.loc) / self.scale
        A = (1 - np.exp(Z) ** (1 + self.scale)) / (1 + self.scale)
        L = Y * (np.exp(Z) - 1) + np.exp(self.loc) * \
            (self.scale / (4 - self.scale ** 2) + A)
        Z = -(np.log(Y) - self.loc) / self.scale
        A = -(1 - np.exp(Z) ** (1 -  self.scale)) / (1 - self.scale)
        R = Y * (1 - np.exp(Z)) + np.exp(self.loc) * \
            (self.scale / (4 - self.scale ** 2) + A)
        return L * (Y < np.exp(self.loc)) + R * (Y >= np.exp(self.loc))

    def fit(Y):
        m, s = osp.stats.laplace.fit(np.log(Y))
        return np.array([m, np.log(s ** 2)])
