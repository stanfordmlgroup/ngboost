"""The NGBoost NegativeBinomial distribution and scores"""
import numpy as np
from scipy.stats import nbinom as dist
from scipy.special import digamma
from scipy.optimize import Bounds, minimize

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

#helper function because scipy doesn't provide a fit function natively
def negative_log_likelihood(params,k):
    return -dist.logpmf(k = k, n = params[0], p = params[1]).sum()

class NegativeBinomialLogScore(LogScore):

    def score(self, Y):
        return -self.dist.logpmf(Y)
    
    def d_score(self, Y):
        D = np.zeros((len(Y),2))
        D[:,0] = -self.n * (digamma(Y + self.n) + np.log(self.p) - digamma(self.n))
        D[:,1] = (Y * np.exp(self.z) - self.n)/(np.exp(self.z) + 1)
        return D
    
    def metric(self):
        FI = np.zeros((self.n.shape[0], 2, 2))
        FI[:, 0, 0] = (self.n * self.p)/(self.p + 1)
        FI[:, 1, 1] = self.n * self.p
        return FI        

class NegativeBinomial(RegressionDistn):

    n_params = 2
    scores = [NegativeBinomialLogScore]

    def __init__(self,params):
        # save the parameters
        self._params = params

        self.logn = params[0]
        self.n = np.exp(self.logn)
        #z = log(p/(1-p)) => p = 1/(1 + e^(-z))
        self.z = params[1]
        self.p = 1/(1 + np.exp(-self.z))
        self.dist = dist(n = self.n, p = self.p)

    def fit(Y):
        assert np.equal(
            np.mod(Y, 1), 0
        ).all(), "All Negative Binomial target data must be discrete integers"
        assert np.all([y >= 0 for y in Y]), "Count data must be >= 0"

        m = minimize(
            negative_log_likelihood,
            x0=np.array([np.max(Y),.5]),  # initialized value
            args=(Y,),
            bounds=Bounds((1e-8,1e-8),(np.inf,1-1e-8)),

        )
        return np.array([np.log(m.x[0]), np.log(m.x[1]/(1 - m.x[1]))])
    
    def sample(self,m):
        return np.array([self.dist.rvs() for i in range(m)])

    def __getattr__(
        self, name
    ):  # gives us access to NegativeBinomial.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None
    
    @property
    def params(self):
        return {'n':self.n, 'p':self.p}