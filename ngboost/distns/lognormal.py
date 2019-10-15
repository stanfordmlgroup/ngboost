import scipy as osp
import numpy as onp
import scipy.stats
import jax.numpy as np
import jax.scipy as sp
import jax.random as random


class LogNormal(object):

    n_params = 2
    has_fisher_info = True
    has_crps_metric = True

    def __init__(self, params, temp_scale = 1.0):
        self.params = params
        self.loc = params[0]
        self.scale = np.exp(params[1] / temp_scale) + 1e-8
        self.var = self.scale ** 2  + 1e-8
        self.shp = self.loc.shape

    def pdf(self, Y):
        return sp.stats.norm.pdf(onp.log(Y), loc=self.loc, scale=self.scale)

    def logpdf(self, Y):
        return sp.stats.norm.logpdf(onp.log(Y), loc=self.loc, scale=self.scale)

    def cdf(self, Y):
        return sp.stats.norm.cdf(onp.log(Y), loc=self.loc, scale=self.scale)

    def sample(self, key):
        return onp.exp(random.normal(key=key, shape=self.shp,) * self.scale + self.loc)

    def ppf(self, Q):
        return onp.exp(osp.stats.norm.ppf(Q, loc=self.loc, scale=self.scale))

    def crps(self, Y):
        Z = (onp.log(Y) - self.loc) / self.scale
        return self.scale * (Z * (2 * sp.stats.norm.cdf(Z) - 1) + \
               2 * sp.stats.norm.pdf(Z) - 1 / np.sqrt(np.pi))

    def crps_metric(self):
        I = 1/(2*np.sqrt(np.pi)) * np.diag(np.array([1, self.var/2]))
        return I + 1e-4 * np.eye(2)

    def fisher_info(self):
        I = np.diag(np.array([1 / self.var, 2]))
        return I + 1e-4 * np.eye(2)

    def fisher_info_cens(self, T):
        nabla = np.array([self.pdf(T),
                          (T - self.loc) / self.scale * self.pdf(T)])
        return np.outer(nabla, nabla) / (self.cdf(T) * (1 - self.cdf(T))) + 1e-2 * np.eye(2)

    def fit(Y):
        m, s = osp.stats.norm.fit(onp.log(Y))
        return onp.array([m, onp.log(s)])
        #return np.array([m, np.log(1e-5)])

    def obj(self):
        return osp.stats.lognormal(self.scale, loc=0, scale=onp.exp(self.loc))

class HomoskedasticLogNormal(LogNormal):

    n_params = 1

    def __init__(self, params):
        self.loc = params[0]
        self.var = np.ones_like(self.loc)
        self.scale = np.ones_like(self.loc)
        self.shape = self.loc.shape

    def fit(Y):
        m, s = osp.stats.norm.fit(onp.log(Y))
        return m

    def crps_metric(self):
        return 1

    def fisher_info(self):
        return 1
