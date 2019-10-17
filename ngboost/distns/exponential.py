import scipy as sp
import numpy as np
from scipy.stats import expon as dist

eps = 1e-5

class Exponential(object):
    n_params = 1

    def __init__(self, params):
        self.loc = 0
        self.scale = np.exp(params[0])
        self.shp = self.scale.shape

        self.dist = dist(scale=self.scale)

    def __getattr__(self, name):
        return getattr(self.dist, name)

    def nll(self, Y):
        try:
            E = Y['Event']
            T = Y['Time']
            cens = (1-E) * np.log(1 - self.dist.cdf(T) + eps)
            uncens = E * self.dist.logpdf(T)
            return -(cens + uncens)
        except:
            return -self.dist.logpdf(Y)

    def D_nll(self, Y):
        try:
            E = Y['Event']
            T = Y['Time']
            cens = (1-E) * T.squeeze() / self.scale
            uncens = E * (-1 + T.squeeze() / self.scale)
            return -(cens + uncens).reshape((-1, 1))
        except:
            return (-1 + Y.squeeze() / self.scale).reshape((-1, 1))

    def fisher_info(self):
        FI = np.ones((self.scale.shape[0], 1, 1))
        return FI

    def crps(self, Y):
        try:
            E = Y['Event']
            T = Y['Time']
            return None
        except:
            return None
        pass

    def D_crps(self, Y):
        pass

    def crps_metric(self):
        pass

    def fit(Y):
        m, s = sp.stats.expon.fit(Y)
        return np.array([np.log(m + s)])
