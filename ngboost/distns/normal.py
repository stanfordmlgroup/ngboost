import scipy as sp
import numpy as np

from scipy.stats import norm as dist

class Normal(object):
    n_params = 2

    def __init__(self, params, temp_scale = 1.0):
        self.loc = params[0]
        self.scale = np.exp(params[1] / temp_scale) + 1e-8
        self.var = self.scale ** 2  + 1e-8
        self.shp = self.loc.shape

        self.dist = dist(loc=self.loc, scale=self.scale)

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def nll(self, Y):
        try:
            E = Y['Event']
            T = Y['Time']
            cens = (1-E) * np.log(1 - self.dist.cdf(T) + eps)
            uncens = E * self.dist.logpdf(T)
            return -(cens + uncens).mean()
        except:
            return -self.dist.logpdf(Y).mean()

    def D_nll(self, Y_):
        Y = Y_.squeeze()
        D = np.zeros((self.var.shape[0], 2))
        D[:, 0] = (self.loc - Y) / self.var
        D[:, 1] = 1 - ((self.loc - Y) ** 2) / self.var
        return D

    def crps(self, Y):
        Z = (Y - self.loc) / self.scale
        return self.scale * (Z * (2 * sp.stats.norm.cdf(Z) - 1) + \
               2 * sp.stats.norm.pdf(Z) - 1 / np.sqrt(np.pi))

    def crps_metric(self):
        I = 1/(2*np.sqrt(np.pi)) * np.diag(np.array([1, self.var/2]))
        return I + 1e-4 * np.eye(2)

    def fisher_info(self):
        FI = np.zeros((self.var.shape[0], 2, 2))
        FI[:, 0, 0] = 1/self.var + 1e-5
        FI[:, 1, 1] = 2
        return FI

    def fisher_info_cens(self, T):
        nabla = np.array([self.pdf(T),
                          (T - self.loc) / self.scale * self.pdf(T)])
        return np.outer(nabla, nabla) / (self.cdf(T) * (1 - self.cdf(T))) + 1e-2 * np.eye(2)

    def fit(Y):
        m, s = sp.stats.norm.fit(Y)
        return np.array([m, np.log(s)])
        #return np.array([m, np.log(1e-5)])

class HomoskedasticNormal(Normal):

    n_params = 1

    def __init__(self, params):
        self.loc = params[0]
        self.var = np.ones_like(self.loc)
        self.scale = np.ones_like(self.loc)
        self.shape = self.loc.shape

    def fit(Y):
        m, s = sp.stats.norm.fit(Y)
        return m
    
    def crps_metric(self):
        return 1

    def fisher_info(self):
        return 1
