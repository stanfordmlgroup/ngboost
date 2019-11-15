import scipy as sp
import numpy as np
from scipy.stats import expon as dist

eps = 1e-5

class Exponential(object):
    n_params = 1

    def __init__(self, params):
        self.scale = np.exp(params[0])
        self.dist = dist(scale=self.scale)

    def __getattr__(self, name):
        return getattr(self.dist, name)

    def nll(self, Y):
        E, T = Y["Event"], Y["Time"]
        cens = (1-E) * np.log(1 - self.dist.cdf(T) + eps)
        uncens = E * self.dist.logpdf(T)
        return -(cens + uncens)

    def D_nll(self, Y):
        E, T = Y["Event"], Y["Time"]
        cens = (1-E) * T.squeeze() / self.scale
        uncens = E * (-1 + T.squeeze() / self.scale)
        return -(cens + uncens).reshape((-1, 1))

    def fisher_info(self):
        FI = 2 * np.ones((self.scale.shape[0], 1, 1))
        return FI

    def crps(self, Y):
        E = Y['Event']
        T = Y['Time']
        c = E == 1
        score = T + self.scale * (2 * np.exp(-T / self.scale) - 1.5)
        score[c] -= 0.5 * self.scale[c] * np.exp(-2 * T[c] / self.scale[c])
        return score

    def D_crps(self, Y):
        E = Y["Event"]
        T = Y["Time"]
        c = E == 1
        d = 2 * np.exp(-T/self.scale) * (self.scale + T) - 1.5 * self.scale
        d[c] -= np.exp(-2 * T[c] / self.scale[c]) * (0.5 * self.scale[c] - T[c])
        return deriv.reshape((-1, 1))

    def crps_metric(self):
        M = 0.5 * self.scale[:, np.newaxis, np.newaxis]
        return M

    def fit(Y):
        m, s = sp.stats.expon.fit(Y["Time"])
        return np.array([np.log(m + s)])
