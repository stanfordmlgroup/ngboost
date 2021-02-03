from scipy.stats import gamma 
from scipy.special import digamma
import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

class GammaLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)
    
    def d_score(self, Y):
        D = np.zeros((len(Y), 3))
        
        D[:, 0] = self.a * (np.log(self.scale) + digamma(self.a) - np.log(Y))
        D[:, 1] = self.loc
        D[:, 2] = self.a + (self.scale * Y)
        return D
    
class Gamma(RegressionDistn):
    n_params = 3
    scores = [GammaLogScore]
    def __init__(self, params):
        # save the parameters
        self._params = params
        # create other objects that will be useful later
        self.loga = params[0]       
        self.logscale = params[2]
        self.loc = params[1]
        self.a = np.exp(params[0])
        self.scale = np.exp(params[2])
        self.dist = gamma(a = self.a, scale=self.scale)
    def fit(Y):
        a, loc, s = gamma.fit(Y) # use scipy's implementation
        return np.array([np.log(a), loc, np.log(s)])
    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])
    def __getattr__(self, name): # gives us access to gamma.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None
    @property
    def params(self):
        return {'loc':self.a, 'scale':self.scale}
