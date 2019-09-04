import scipy as osp
import scipy.stats
import jax.numpy as np
import jax.scipy as sp
import jax.random as random


class Normal(object):

    n_params = 2
    has_fisher_info = True
    has_crps_metric = True

    def __init__(self, params, temp_scale = 1.0):
        self.loc = params[0]
        self.var = np.exp(params[1] / temp_scale) + 1e-20
        self.scale = self.var ** 0.5
        self.shp = self.loc.shape

    def pdf(self, Y):
        return sp.stats.norm.pdf(Y, loc=self.loc, scale=self.scale)

    def logpdf(self, Y):
        return sp.stats.norm.logpdf(Y, loc=self.loc, scale=self.scale)

    def cdf(self, Y):
        return sp.stats.norm.cdf(Y, loc=self.loc, scale=self.scale)

    def sample(self, key):
        return random.normal(key=key, shape=self.shp,) * self.scale + self.loc

    def ppf(self, Q):
        return osp.stats.norm.ppf(Q, loc=self.loc, scale=self.scale)

    def crps(self, Y):
        Z = (Y - self.loc) / self.scale
        return self.scale * (Z * (2 * sp.stats.norm.cdf(Z) - 1) + \
               2 * sp.stats.norm.pdf(Z) - 1 / np.sqrt(np.pi))

    def crps_debug(self, Y):
        print('Y=%.4f loc=%.4f, scale=%.4f' % (Y, self.loc, self.scale))
        Z = (Y - self.loc) / self.scale
        return self.scale * (Z * (2 * sp.stats.norm.cdf(Z) - 1) + \
               2 * sp.stats.norm.pdf(Z) - 1 / np.sqrt(np.pi))

    def crps_metric(self):
        I = np.diag(np.array([1 / np.sqrt(np.pi) / self.scale,
                              0.5 / np.sqrt(np.pi) / self.scale]))
        return I + 1e-4 * np.eye(2)

    def fisher_info(self):
        I = np.diag(np.array([1 / self.var, 0.5 / self.var ** 2]))
        return I + 1e-4 * np.eye(2)

    def fisher_info_cens(self, T):
        nabla = np.array([self.pdf(T),
                          (T - self.loc) / self.scale * self.pdf(T)])
        return np.outer(nabla, nabla) / (self.cdf(T) * (1 - self.cdf(T))) + 1e-2 * np.eye(2)

    def fit(Y):
        m, s = osp.stats.norm.fit(Y)
        print(m, s)
        return np.array([m, np.log(s ** 2)])


class HomoskedasticNormal(Normal):

    n_params = 1

    def __init__(self, params):
        self.loc = params[0]
        self.var = np.ones_like(self.loc)
        self.scale = np.ones_like(self.loc)
        self.shape = self.loc.shape


    def fit(Y):
        m, s = osp.stats.norm.fit(Y)
        return m
