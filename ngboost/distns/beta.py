import numpy as np
import numpy
import math
import scipy as sp
from scipy.stats import beta as dist
from scipy.special import polygamma, beta

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore

import sympy as sym
from sympy import stats as symstats
from sympy.printing.lambdarepr import NumPyPrinter
from sympy.utilities.lambdify import lambdastr

logalpha, logbeta, x = sym.symbols('logalpha logbeta x')

distr = symstats.Beta('dist', sym.exp(logalpha), sym.exp(logbeta))
score = -sym.log(symstats.density( distr ).pdf(x))
def neg_loglikelihood_sympy(logalpha, logbeta, x):
    return score

# Need to create str, because non-numpy functions are used: polygamma; and exp is not translated into numpy inside polygamma
def newlambdify(args, funs):
    funcstr = lambdastr(args, funs, printer=NumPyPrinter)
    funcstr = funcstr.replace(
        ' exp', 'numpy.exp'
    )
    return eval(funcstr)

neg_loglikelihood = np.vectorize( newlambdify((logalpha, logbeta, x), neg_loglikelihood_sympy(logalpha, logbeta, x)) )
D_0 = np.vectorize( newlambdify( (logalpha, logbeta, x), sym.diff(neg_loglikelihood_sympy(logalpha, logbeta, x), logalpha)) )
D_1 = np.vectorize( newlambdify( (logalpha, logbeta, x), sym.diff(neg_loglikelihood_sympy(logalpha, logbeta, x), logbeta)) )
# FI_0_0 = np.vectorize( newlambdify( (logalpha, logbeta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logalpha), logalpha))).subs(x, distr))))) )
# FI_0_1 = np.vectorize( newlambdify( (logalpha, logbeta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logalpha), logbeta))).subs(x, distr))))) )
# FI_1_0 = np.vectorize( newlambdify( (logalpha, logbeta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logbeta), logalpha))).subs(x, distr))))) )
# FI_1_1 = np.vectorize( newlambdify( (logalpha, logbeta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logbeta), logbeta))).subs(x, distr))))) )



class BetaLogScore(LogScore):

    def score(self, Y):
        return -dist.logpdf(Y, a=self.alpha, b=self.beta, loc=0, scale=1)
        # return neg_loglikelihood(k=self.a, logtheta=self.logscale, x=Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2)) # first col is dS/da, second col is dS/d(log(scale))
        D[:, 0] = D_0(logalpha=self.logalpha, logbeta=self.logbeta, x=Y)
        D[:, 1] = D_1(logalpha=self.logalpha, logbeta=self.logbeta, x=Y)
        return D
    
class Beta(RegressionDistn):

    n_params = 2
    scores = [BetaLogScore]

    def __init__(self, params):
        # save the parameters
        self._params = params

        # create other objects that will be useful later
        self.logalpha = params[0]
        self.logbeta = params[1]
        self.alpha = np.exp(params[0])
        self.beta = np.exp(params[1]) # since params[1] is log(scale)
        self.dist = dist(a=self.alpha, b=self.beta)

    def fit(Y):
        alpha, beta, loc1, scale1  = dist.fit(Y, floc=0, fscale=1) # use scipy's implementation
        return np.array([np.log(alpha), np.log(beta)])

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])

    def __getattr__(self, name): # gives us access to Laplace.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {'alpha':self.alpha, 'beta':self.beta}