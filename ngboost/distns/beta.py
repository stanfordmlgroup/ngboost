from scipy.stats import beta as dist
import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import digamma
                      
class BetaLogScore(LogScore): 
    
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2)) # first col is dS/d(log(α)), second col is dS/d(log(β))
        D[:, 0] = digamma(self.log_alpha + self.log_beta) - digamma(self.log_alpha) + log(Y)
        D[:, 1] = digamma(self.log_alpha + self.log_beta) - digamma(self.log_beta) + log(1-Y)
        return D

class Beta(RegressionDistn):

    n_params = 2
    scores = [BetaLogScore] 

    def __init__(self, params):
        # save the parameters
        self._params = params
        
        # create other objects that will be useful later
        self.log_alpha = params[0]
        self.log_beta = params[1]
        self.alpha = np.exp(params[0]) # since params[1] is log(scale)
        self.beta = np.exp(params[1]) # since params[1] is log(scale)
        self.dist = dist(a=self.alpha, b=self.beta)

    def fit(Y):
        alpha, beta, loc, scale = dist.fit(Y) # use scipy's implementation
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