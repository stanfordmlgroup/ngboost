from ngboost.distns import RegressionDistn
from ngboost.scores import LogScore
import scipy as sp
import numpy as np
from scipy.stats import poisson as dist
from scipy.special import factorial
from scipy.optimize import minimize

### Helpers ####
def negative_log_likelihood(params, data):
    return -dist.logpmf(np.array(data), params[0]).sum()

def isinteger(x):
    return np.equal(np.mod(x, 1), 0)


## NGBoost Classes ##
class PoissonLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpmf(Y)
    
    def d_score(self, Y):
        D = np.zeros((len(Y), 1))
        D[:, 0] = self.mu - Y
        return D
    
class Poisson(RegressionDistn):

    n_params = 1
    scores = [PoissonLogScore] 

    def __init__(self, params):
        # save the parameters
        self._params = params
        
        # create other objects that will be useful later
        self.logmu = params[0]
        self.mu = np.exp(self.logmu)
        self.dist = dist(mu=self.mu)

    def fit(Y):
        assert(np.all([np.equal(np.mod(y, 1), 0) for y in Y])), "All Poisson target data must be discrete integers"
        
        ## alternative method to fit Poisson dist instead of mean ##
#         m = minimize(negative_log_likelihood,
#                      x0=np.ones(1), # initialized value
#                      args=(Y,),       
#                      bounds=(Bounds(0,np.max(Y))),
#                      method='', # minimization method, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#                   )

        ## I have not found a dataset that NLL outperforms just mean ##
        ## would return np.array([np.log(m.x)])
        
        return np.array([np.log(np.mean(Y))])

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])
    
    def __getattr__(self, name): # gives us access to Poisson.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None
    
    @property
    def params(self):
        return {'mu':self.mu}