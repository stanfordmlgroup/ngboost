from scipy.stats import betabinom as dist
from scipy.stats import beta as betadist
import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import polygamma, gamma, digamma
from scipy.special import beta as betafunction
from fastbetabino import *
from array import array     
class BetaBernoulliLogScore(LogScore): 
    
    def score(self, Y):
        return -self.dist.logpmf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2)) # first col is dS/d(log(α)), second col is dS/d(log(β))
        p = betadist(a=self.alpha, b=self.beta).mean()

        D[:, 0] =   self.alpha * (
                        digamma(self.alpha + self.beta) + 
                        digamma(Y + self.alpha) -
                        digamma(self.alpha + self.beta + 2) -
                        digamma(self.alpha)
                    )
        D[:, 1] =   self.beta * (
                        digamma(self.alpha + self.beta) + 
                        digamma(-Y + self.beta + 2) -
                        digamma(self.alpha + self.beta + 2) -
                        digamma(self.beta)
                    )
        return D

class BetaBernoulli(RegressionDistn):

    n_params = 2
    scores = [BetaBernoulliLogScore]

    def __init__(self, params):
        # save the parameters
        self._params = params
        
        # create other objects that will be useful later
        self.log_alpha = params[0]
        self.log_beta = params[1]
        self.alpha = np.exp(params[0]) # since params[1] is log(alpha)
        self.beta = np.exp(params[1]) # since params[1] is log(beta)
        self.dist = dist(n=1, a=self.alpha, b=self.beta)

    def fit(Y):
        imps = np.ones_like(Y)
        alpha, beta = fit_alpha_beta(imps, Y) # use scipy's implementation
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