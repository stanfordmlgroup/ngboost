import scipy as sp
import numpy as np

from scipy.stats import lognorm as dist

eps = 1e-6

class LogNormal(object):
    n_params = 2

    def __init__(self, params, temp_scale = 1.0):
        mu = params[0]
        logsigma = params[1]
        sigma = np.exp(logsigma)

        self.mu = mu
        self.sigma = sigma

        self.dist = dist(s=sigma, scale=np.exp(mu))

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
            return -(cens + uncens)
        except:
            return -self.dist.logpdf(Y)

    def D_nll(self, Y):
        if True:
            E = Y['Event'].reshape((-1, 1))
            T = Y['Time']
            lT = np.log(T)

            D_uncens = np.zeros((self.mu.shape[0], 2))
            D_uncens[:, 0] = (self.mu - lT) / (self.sigma ** 2)
            D_uncens[:, 1] = 1 - ((self.mu - lT) ** 2) / (self.sigma ** 2)

            D_cens = np.zeros((self.mu.shape[0], 2))
            Z = (lT - self.mu) / self.sigma
            D_cens[:, 0] = sp.stats.norm.pdf(lT, loc=self.mu, scale=self.sigma)/(1 - self.dist.cdf(T) + eps)
            D_cens[:, 0] = Z * sp.stats.norm.pdf(lT, loc=self.mu, scale=self.sigma)/(1 - self.dist.cdf(T) + eps)

            cens = (1-E) * D_cens
            uncens = -(E * D_uncens)
            return -(cens + uncens)
        else:
            Y = Y_.squeeze()
            D = np.zeros((self.mu.shape[0], 2))
            D[:, 0] = (self.mu - np.log(T)) / (self.sigma ** 2)
            D[:, 1] = 1 - ((self.mu - np.log(T)) ** 2) / (self.sigma ** 2)
            return D

    def crps(self, Y_):
        Y = np.log(Y_.squeeze())
        Z = (Y - self.loc) / self.scale
        return self.scale * (Z * (2 * sp.stats.norm.cdf(Z) - 1) + \
               2 * sp.stats.norm.pdf(Z) - 1 / np.sqrt(np.pi))

    def crps_metric(self):
        I = 1/(2*np.sqrt(np.pi)) * np.diag(np.array([1, self.var/2]))
        return I + 1e-4 * np.eye(2)

    def fisher_info(self):
        FI = np.zeros((self.mu.shape[0], 2, 2))
        FI[:, 0, 0] = 1/(self.sigma ** 2) + eps
        FI[:, 1, 1] = 2
        return FI

    def fisher_info_cens(self, T):
        nabla = np.array([self.pdf(T),
                          (T - self.loc) / self.scale * self.pdf(T)])
        return np.outer(nabla, nabla) / (self.cdf(T) * (1 - self.cdf(T))) + 1e-2 * np.eye(2)

    def fit(Y):
        m, s = sp.stats.norm.fit(np.log(Y.squeeze()))
        return np.array([m, np.log(s)])
