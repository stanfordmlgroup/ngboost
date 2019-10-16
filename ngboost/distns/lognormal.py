import scipy as sp
import numpy as np

from scipy.stats import lognorm as dist

class Normal(object):
    n_params = 2

    def __init__(self, params, temp_scale = 1.0):
        mu = params[0]
        logsigma = params[1]
        sigma = np.exp(logsigma)

        self.dist = dist(s=sigma, scale=np.exp(mu))

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def nll(self, Y):
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
        m, s = sp.stats.norm.fit(np.log(Y.squeeze()))
        return np.array([m, np.log(s)])
