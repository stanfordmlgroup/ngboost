import numpy as np
import scipy as sp
import scipy.special
from scipy.stats import bernoulli as dist


class Bernoulli(object):

    n_params = 1
    problem_type = "classification"

    def __init__(self, params):
        self.logit = params[0]
        self.prob = sp.special.expit(self.logit)
        self.dist = dist(p=self.prob)

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def nll(self, Y):
        return -self.dist.logpmf(Y).mean()

    def D_nll(self, Y):
        D_1 = -np.exp(-self.logit) * self.prob
        D_0 = np.exp(-self.logit) * (self.prob ** 2) / (1 - self.prob)
        return (Y * D_1 + (1 - Y) * D_0)[:,np.newaxis]

    def fisher_info(self):
        FI = (self.prob * np.exp(-self.logit) / (1 - self.prob))
        return FI[:, np.newaxis, np.newaxis]

    def crps(self, Y):
        return ((self.prob - Y) ** 2).mean()

    def D_crps(self, Y):
        D = 2 * (self.prob - Y) * self.prob ** 2 * np.exp(-self.logit)
        return D[:,np.newaxis]

    def crps_metric(self):
        M = 2 * self.prob ** 2 * np.exp(-2 * self.logit) * (1 + (self.prob / (1 - self.prob)) ** 2)
        return M[:, np.newaxis, np.newaxis]

    def fit(Y):
        return np.array([sp.special.logit(np.mean(Y))])
