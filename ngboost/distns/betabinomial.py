from scipy.stats import betabinom as dist
import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import digamma, polygamma
from array import array
import sys
from scipy.special import binom as binomial
import numpy
import math
import sympy as sym
from sympy import stats as symstats
from sympy.printing.lambdarepr import NumPyPrinter
from sympy.utilities.lambdify import lambdastr
import re

# Need to create str, because non-numpy functions are used: polygamma; and exp is not translated into numpy inside polygamma
def newlambdify(args, funs):
    funcstr = lambdastr(args, funs, printer=NumPyPrinter)
    funcstr = funcstr.replace(
        ' exp', 'numpy.exp'
    ).replace(
        'builtins.sum', 'sum'
    )
    funcstr = re.sub(r'\bk_\d*\b', '_k', funcstr)
    return eval(funcstr)

n, logalpha, logbeta, x = sym.symbols('n, logalpha, logbeta, x')

distr = symstats.BetaBinomial('dist', n, sym.exp(logalpha), sym.exp(logbeta))
score = -sym.log(symstats.density( distr ).pmf(x))
def neg_loglikelihood_sympy(n, logalpha, logbeta, x):
    return score

neg_loglikelihood = np.vectorize( newlambdify( (n, logalpha, logbeta, x), neg_loglikelihood_sympy(n, logalpha, logbeta, x) ) )
D_0 = np.vectorize( newlambdify( (n, logalpha, logbeta, x), sym.diff(neg_loglikelihood_sympy(n, logalpha, logbeta, x), logalpha) ))
D_1 = np.vectorize( newlambdify( (n, logalpha, logbeta, x), sym.diff(neg_loglikelihood_sympy(n, logalpha, logbeta, x), logbeta)))
FI_0_0 = np.vectorize( newlambdify( (n, logalpha, logbeta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logalpha), logalpha))).subs(x, distr)))) ))
FI_0_1 = np.vectorize( newlambdify( (n, logalpha, logbeta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logalpha), logbeta))).subs(x, distr)))) ))
FI_1_0 = np.vectorize( newlambdify( (n, logalpha, logbeta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logbeta), logalpha))).subs(x, distr)))) ))
FI_1_1 = np.vectorize( newlambdify( (n, logalpha, logbeta), sym.factor(sym.expand(symstats.E(sym.factor(sym.expand(sym.diff(sym.diff(score, logbeta), logbeta))).subs(x, distr)))) ))


class BetaBinomialLogScore(LogScore):
    def score(self, Y):
        return neg_loglikelihood(n=self.n, logalpha=self.logalpha, logbeta=np.log(self.logbeta), x=Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2))  # first col is dS/d(log(α)), second col is dS/d(log(β))
        D[:, 0] = D_0(n=self.n, logalpha=self.logalpha, logbeta=np.log(self.logbeta), x=Y)
        D[:, 1] = D_1(n=self.n, logalpha=self.logalpha, logbeta=np.log(self.logbeta), x=Y)
        return D

    # Variance
    def metric(self):
        FI = np.zeros((self.var.shape[0], 2, 2))
        FI[:, 0, 0] = FI_0_0(n=self.n, logalpha=self.logalpha, logbeta=np.log(self.logbeta))
        FI[:, 0, 1] = FI_0_1(n=self.n, logalpha=self.logalpha, logbeta=np.log(self.logbeta))
        FI[:, 1, 0] = FI_1_0(n=self.n, logalpha=self.logalpha, logbeta=np.log(self.logbeta))
        FI[:, 1, 1] = FI_1_1(n=self.n, logalpha=self.logalpha, logbeta=np.log(self.logbeta))
        return FI
 
class BetaBinomial(RegressionDistn):

    n_params = 2
    scores = [BetaBinomialLogScore]

    def __init__(self, params):
        # save the parameters
        self._params = params

        # create other objects that will be useful later
        self.logalpha = params[0]
        self.logbeta = params[1]
        self.alpha = np.exp(self.logalpha)
        self.beta = np.exp(self.logbeta)
        self.dist = dist(n=1, a=self.alpha, b=self.beta)

    def fit(n, Y):
        def fit_alpha_beta_py(trials, success, alpha0=1.5, beta0=5, niter=1000):
            # based on https://github.com/lfiaschi/fastbetabino/blob/master/fastbetabino.pyx

            alpha_old = alpha0
            beta_old = beta0

            for it in range(niter):

                alpha = (
                    alpha_old
                    * (
                        sum(
                            digamma(c + alpha_old) - digamma(alpha_old)
                            for c, i in zip(success, trials)
                        )
                    )
                    / (
                        sum(
                            digamma(i + alpha_old + beta_old)
                            - digamma(alpha_old + beta_old)
                            for c, i in zip(success, trials)
                        )
                    )
                )

                beta = (
                    beta_old
                    * (
                        sum(
                            digamma(i - c + beta_old) - digamma(beta_old)
                            for c, i in zip(success, trials)
                        )
                    )
                    / (
                        sum(
                            digamma(i + alpha_old + beta_old)
                            - digamma(alpha_old + beta_old)
                            for c, i in zip(success, trials)
                        )
                    )
                )

                # print('alpha {} | {}  beta {} | {}'.format(alpha,alpha_old,beta,beta_old))
                sys.stdout.flush()

                if np.abs(alpha - alpha_old) and np.abs(beta - beta_old) < 1e-10:
                    # print('early stop')
                    break

                alpha_old = alpha
                beta_old = beta

            return alpha, beta

        alpha, beta = fit_alpha_beta_py(n, Y)
        return np.array([np.log(alpha), np.log(beta)])

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])

    def __getattr__(self, name):  
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"alpha": self.alpha, "beta": self.beta}
    