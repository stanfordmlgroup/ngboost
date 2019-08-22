import scipy as osp
import scipy.stats
import jax.numpy as np
import jax.scipy as sp
import jax.random as random


class Laplace(object):

    n_params = 2
    has_fisher_info = True
    has_crps_metric = True

    def __init__(self, params, temp_scale = 1.0):
        self.loc = params[0]
        self.scale = np.exp(params[1] / temp_scale) ** 0.5
        self.var = 2 * self.scale ** 2
        self.shp = self.loc.shape

    def logpdf(self, Y):
        return sp.stats.laplace.logpdf(Y, loc=self.loc, scale=self.scale)

    def cdf(self, Y):
        return sp.stats.laplace.cdf(Y, loc=self.loc, scale=self.scale)

    def ppf(self, Q):
        return osp.stats.laplace.ppf(Q, loc=self.loc, scale=self.scale)

    def crps(self, Y):
        Z = (Y - self.loc) / self.scale
        return self.scale * (np.abs(Z) + np.exp(-np.abs(Z)) - 0.75)

    def crps_metric(self):
        M = np.diag(np.array([1.0 / np.sqrt(np.pi) / self.scale,
                              0.5 / np.sqrt(np.pi) / self.scale]))
        return M + 1e-4 * np.eye(2)

    def fisher_info(self):
        M = np.diag(np.array([0.5 / self.scale, 0.25 / self.scale]))
        return M + 1e-4 * np.eye(2)


class HomoskedasticLaplace(Laplace):

    n_params = 1

    def __init__(self, params):
        self.loc = params[0]
        self.scale = np.ones_like(self.loc)
        self.shape = self.loc.shape

