import scipy as osp
import scipy.stats
import jax.numpy as np
import jax.scipy as sp
import jax.random as random


class LogNormal(object):

    n_params = 2
    has_fisher_info = False
    has_crps_metric = False

    def __init__(self, params, temp_scale = 10.0):
        self.loc = params[0]
        self.scale = np.exp(params[1] / temp_scale) ** 0.5
        self.shp = self.loc.shape
        self.params = params

    def logpdf(self, Y):
        return sp.stats.norm.logpdf(np.log(Y), loc=self.loc, scale=self.scale)

    def cdf(self, Y):
        return sp.stats.norm.cdf(np.log(Y), loc=self.loc, scale=self.scale)

    def sample(self, key):
        logsmp = random.normal(key=key, shape=self.shp,) * self.scale + self.loc
        return np.exp(logsmp)

    def ppf(self, Q):
        return osp.stats.lognorm.ppf(Q, s=self.scale, scale=np.exp(self.loc))

    def crps(self, Y):
        return Y * (2 * self.cdf(Y) - 1) - 2 * \
               np.exp(self.loc + 0.5 * self.scale ** 2) * \
               (sp.stats.norm.cdf((np.log(Y) - self.loc - self.scale ** 2) / \
                self.scale) + \
                sp.stats.norm.cdf(self.scale / np.sqrt(2)) - 1)

    def crps_metric(self):
        I = np.diag(np.array([1 / np.sqrt(np.pi) / self.scale,
                              0.5 / np.sqrt(np.pi) / self.scale]))
        return I + 1e-4 * np.eye(2)



class HomoskedasticLogNormal(LogNormal):

    n_params = 1

    def __init__(self, params):
        self.loc = params[0]
        self.scale = np.ones_like(self.loc)
        self.params = params
