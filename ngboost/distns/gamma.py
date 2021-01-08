"""The NGBoost Gamma distribution and scores"""
import numpy as np
import scipy as sp
from scipy.stats import gamma as dist
from scipy.special import polygamma
import numpy
import math
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore

import sympy as sym
from sympy import stats as symstats

from sympy.printing.lambdarepr import NumPyPrinter
from sympy.utilities.lambdify import lambdastr

# Need to create str, because non-numpy functions are used: polygamma; and exp is not translated into numpy inside polygamma
def newlambdify(args, funs):
    funcstr = lambdastr(args, funs, printer=NumPyPrinter)
    funcstr = funcstr.replace(
        ' exp', 'numpy.exp'
    )
    return eval(funcstr)

logk, logtheta, x = sym.symbols('logk logtheta x')

distr = symstats.Gamma('dist', sym.exp(logk), sym.exp(logtheta))
score = -sym.log(symstats.density( distr ).pdf(x))
def neg_loglikelihood_sympy(mu, logsigma, x):
    return -sym.log(symstats.density( symstats.Gamma('dist', sym.exp(logk), sym.exp(logtheta)) ).pdf(x))

# Need to vectorize, because FI matrix is calculated by simulation, 
# plus I guess polygamma doesn't support vector operations
neg_loglikelihood = np.vectorize( newlambdify((logk, logtheta, x), neg_loglikelihood_sympy(logk, logtheta, x) ) )
D_0 = np.vectorize( newlambdify( (logk, logtheta, x), sym.diff(neg_loglikelihood_sympy(logk, logtheta, x), logk) ) )
D_1 = np.vectorize( newlambdify( (logk, logtheta, x), sym.diff(neg_loglikelihood_sympy(logk, logtheta, x), logtheta) ) )
FI_0_0 = np.vectorize( newlambdify( (logk, logtheta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logk), logk))).subs(x, distr)))) ) )
FI_0_1 = np.vectorize( newlambdify( (logk, logtheta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logk), logtheta))).subs(x, distr)))) ) )
FI_1_0 = np.vectorize( newlambdify( (logk, logtheta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logtheta), logk))).subs(x, distr)))) ) )
FI_1_1 = np.vectorize( newlambdify( (logk, logtheta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logtheta), logtheta))).subs(x, distr)))) ) )

class GammaLogScore(LogScore):

    def score(self, Y):
        return -dist.logpdf(Y, self.a, loc=np.zeros_like(self.a), scale=self.scale)
        # return neg_loglikelihood(k=self.a, logtheta=self.logscale, x=Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2)) # first col is dS/da, second col is dS/d(log(scale))
        D[:, 0] = D_0(logk=self.loga, logtheta=self.logscale, x=Y)
        D[:, 1] = D_1(logk=self.loga, logtheta=self.logscale, x=Y)
        return D
    
    # no closed form
    # def metric(self):
    #     FI = np.zeros((self.scale.shape[0], 2, 2))
    #     FI[:, 0, 0] = FI_0_0(logk=self.loga, logtheta=self.logscale)
    #     FI[:, 0, 1] = FI_0_1(logk=self.loga, logtheta=self.logscale)
    #     FI[:, 1, 0] = FI_1_0(logk=self.loga, logtheta=self.logscale)
    #     FI[:, 1, 1] = FI_1_1(logk=self.loga, logtheta=self.logscale)

class Gamma(RegressionDistn):

    n_params = 2
    scores = [GammaLogScore]

    def __init__(self, params):
        # save the parameters
        self._params = params

        # create other objects that will be useful later
        self.loga = params[0]
        self.logscale = params[1]
        self.a = np.exp(params[0])
        self.scale = np.exp(params[1]) # since params[1] is log(scale)
        self.dist = dist(a=self.a, loc=0, scale=self.scale)

    def fit(Y):
        a, loc, scale = dist.fit(Y) # use scipy's implementation
        return np.array([np.log(a), np.log(scale)])

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])

    def __getattr__(self, name): # gives us access to Laplace.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {'a':self.a, 'scale':self.scale}
