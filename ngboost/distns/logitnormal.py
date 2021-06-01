"""The NGBoost LogitNormal distribution and scores"""
import numpy as np
import scipy as sp
from scipy.stats import norm as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore

import sympy as sym
from sympy import stats as symstats

mu, logsigma, x = sym.symbols('mu logsigma x')

pdf = 1/(sym.exp(logsigma)*sym.sqrt(2*sym.pi))*sym.exp(-(sym.log(x/(1-x))-mu)**2/(2*sym.exp(logsigma)**2))*1/(x*(1-x))

distr = symstats.ContinuousRV(x, pdf, set=sym.Interval(-sym.oo, sym.oo))
score = -sym.log(pdf)

def neg_loglikelihood_sympy(mu, logsigma, x):
    return score

neg_loglikelihood = sym.lambdify((mu, logsigma, x), neg_loglikelihood_sympy(mu, logsigma, x), 'numpy')
D_0 = sym.lambdify( (mu, logsigma, x), sym.factor(sym.expand(sym.diff(neg_loglikelihood_sympy(mu, logsigma, x), mu))), 'numpy')
D_1 = sym.lambdify( (mu, logsigma, x), sym.factor(sym.expand(sym.diff(neg_loglikelihood_sympy(mu, logsigma, x), logsigma))), 'numpy')
FI_0_0 = sym.lambdify( (mu, logsigma), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, mu), mu))).subs(x, distr)))), 'numpy')
# FI_0_1 = sym.lambdify( (mu, logsigma), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, mu), logsigma))).subs(x, distr)))), 'numpy')
# FI_1_0 = sym.lambdify( (mu, logsigma), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logsigma), mu))).subs(x, distr)))), 'numpy')
# FI_1_1 = sym.lambdify( (mu, logsigma), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logsigma), logsigma))).subs(x, distr)))), 'numpy')

class LogitNormalLogScore(LogScore):

    def score(self, Y):
        return neg_loglikelihood(mu=self.loc, logsigma=np.log(self.scale), x=Y)

    def d_score(self, Y):
        print(D_0(mu=self.loc, logsigma=np.log(self.scale), x=Y))
        print(D_1(mu=self.loc, logsigma=np.log(self.scale), x=Y))
          
        D = np.zeros((len(Y), 2))
        D[:, 0] = D_0(mu=self.loc, logsigma=np.log(self.scale), x=Y)
        D[:, 1] = D_1(mu=self.loc, logsigma=np.log(self.scale), x=Y)
        return D

    # def metric(self):
    #     FI = np.zeros((self.var.shape[0], 2, 2))
    #     FI[:, 0, 0] = FI_0_0(mu=self.loc, logsigma=np.log(self.scale))
    #     FI[:, 0, 1] = 0 # FI_0_1(mu=self.loc, logsigma=np.log(self.scale))
    #     FI[:, 1, 0] = 0 # FI_1_0(mu=self.loc, logsigma=np.log(self.scale))
    #     FI[:, 1, 1] = FI_0_0(mu=self.loc, logsigma=np.log(self.scale)) # FI_1_1(mu=self.loc, logsigma=np.log(self.scale))
    #     return FI

class LogitNormal(RegressionDistn):
    """
    Implements the normal distribution for NGBoost.

    The normal distribution has two parameters, loc and scale, which are
    the mean and standard deviation, respectively.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    n_params = 2
    scores = [LogitNormalLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.var = self.scale ** 2
        self.dist = dist(loc=self.loc, scale=self.scale)

    def fit(Y):
        # m, s = sp.stats.norm.fit(np.log(Y/(1-Y)))
        m, s = 0, 1
        return np.array([m, np.log(s)])

    def sample(self, m):
        return np.array([self.rvs() for i in range(m)])

    def __getattr__(
        self, name
    ):  # gives us Normal.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}
