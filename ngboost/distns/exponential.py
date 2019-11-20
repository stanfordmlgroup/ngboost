import scipy as sp
import numpy as np
from scipy.stats import expon as dist


class Exponential(object):

    n_params = 1
    problem_type = "survival"

    def __init__(self, params):
        self.scale = np.exp(params[0])
        self.dist = dist(scale=self.scale)

    def __getattr__(self, name):
        return getattr(self.dist, name)

    def nll(self, Y):
        E, T = Y["Event"], Y["Time"]
        cens = (1-E) * np.log(1 - self.dist.cdf(T))
        uncens = E * self.dist.logpdf(T)
        return -(cens + uncens)

    def D_nll(self, Y):
        E, T = Y["Event"], Y["Time"]
        cens = (1-E) * T.squeeze() / self.scale
        uncens = E * (-1 + T.squeeze() / self.scale)
        return -(cens + uncens).reshape((-1, 1))

    def fisher_info(self):
        FI = 2 * np.ones_like(self.scale)
        return FI[:, np.newaxis, np.newaxis]

    def crps(self, Y):
        E, T = Y["Event"], Y["Time"]
        score = T + self.scale * (2 * np.exp(-T / self.scale) - 1.5)
        score[E == 1] -= 0.5 * self.scale[E == 1] * np.exp(-2 * T[E == 1] / self.scale[E == 1])
        return score

    def D_crps(self, Y):
        E, T = Y["Event"], Y["Time"]
        deriv = 2 * np.exp(-T / self.scale) * (self.scale + T) - 1.5 * self.scale
        deriv[E == 1] -= np.exp(-2 * T[E == 1] / self.scale[E == 1]) * \
                         (0.5 * self.scale[E == 1] - T[E == 1])
        return deriv.reshape((-1, 1))

    def crps_metric(self):
        M = 0.5 * self.scale[:, np.newaxis, np.newaxis]
        return M

    def fit(Y):
        T = Y["Time"]
        m, s = sp.stats.expon.fit(T)
        return np.array([np.log(m + s)])
