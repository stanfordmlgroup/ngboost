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

## NGBoost Classes ##
class PoissonLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpmf(Y)
    
    def d_score(self, Y):
        D = np.zeros((len(Y), 1))
        D[:, 0] = self.mu - Y
        return D
    
class Poisson(RegressionDistn):
    """
    Implements the Poisson distribution for NGBoost.
    The Poisson distribution has one parameter, mu, which is the mean number of events per interval.
    This distribution has LogScore implemented for it.
    """
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
        assert(np.equal(np.mod(Y, 1), 0).all), "All Poisson target data must be discrete integers"
        assert((y>=0).all()), "Count data must be >= 0"


        # minimize negative log likelihood 
        m = minimize(negative_log_likelihood,
                     x0=np.ones(1), # initialized value
                     args=(Y,),       
                     bounds=(Bounds(0,np.max(Y))),
                  )

        # another option would be returning just the mean : np.array([np.log(np.mean(Y))])
        # however, I would run into lower bound issues when fitting data this way
        # specifically on the The French Motor Third-Party Liability Claims dataset
        # following this example: https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html
        
        return np.array([np.log(m.x)])

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])
    
    def __getattr__(self, name): # gives us access to Poisson.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None
    
    @property
    def params(self):
        return {'mu':self.mu}
        
