from scipy.stats import betabinom as dist
import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import digamma
from array import array  
import sys
class BetaBernoulliLogScore(LogScore): 
    
    def score(self, Y):
        return -self.dist.logpmf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2)) # first col is dS/d(log(α)), second col is dS/d(log(β))

        D[:, 0] =   -self.alpha * (
                        digamma(self.alpha + self.beta) + 
                        digamma(Y + self.alpha) -
                        digamma(self.alpha + self.beta + 1) -
                        digamma(self.alpha)
                    )
        D[:, 1] =   -self.beta * (
                        digamma(self.alpha + self.beta) + 
                        digamma(-Y + self.beta + 1) -
                        digamma(self.alpha + self.beta + 1) -
                        digamma(self.beta)
                    )
        return D
    
    def metric(self):
        FI = np.zeros((self.alpha.shape[0], 2, 2))
        FI[:, 0, 0] =   ((self.alpha * (
                        digamma(self.alpha + self.beta) + 
                        digamma(0 + self.alpha) -
                        digamma(self.alpha + self.beta + 1) -
                        digamma(self.alpha)
                        ))**2 * self.dist.pmf(0) + 
                        (self.alpha * (
                        digamma(self.alpha + self.beta) + 
                        digamma(1 + self.alpha) -
                        digamma(self.alpha + self.beta + 1) -
                        digamma(self.alpha)
                        ))**2 * self.dist.pmf(1))
        FI[:, 1, 1] =   ((self.beta * (
                        digamma(self.alpha + self.beta) + 
                        digamma(-0 + self.beta + 1) -
                        digamma(self.alpha + self.beta + 1) -
                        digamma(self.beta)
                        ))**2 * self.dist.pmf(0) + 
                        (self.beta * (
                        digamma(self.alpha + self.beta) + 
                        digamma(-1 + self.beta + 1) -
                        digamma(self.alpha + self.beta + 1) -
                        digamma(self.beta)
                        ))**2 * self.dist.pmf(1))
        return FI
class BetaBernoulli(RegressionDistn):

    n_params = 2
    scores = [BetaBernoulliLogScore]

    def __init__(self, params):
        # save the parameters
        self._params = params
        
        # create other objects that will be useful later
        self.log_alpha = params[0]
        self.log_beta = params[1]
        self.alpha = np.exp(self.log_alpha)
        self.beta = np.exp(self.log_beta)
        self.dist = dist(n=1, a=self.alpha, b=self.beta)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def fit(Y):
        
        def fit_alpha_beta_py(impressions, clicks, alpha0=1.5, beta0=5, niter=1000):
            # based on https://github.com/lfiaschi/fastbetabino/blob/master/fastbetabino.pyx

            alpha_old=alpha0
            beta_old=beta0

            for it in range(niter):

                alpha=alpha_old*\
                (sum(digamma(c + alpha_old) - digamma(alpha_old) for c,i in zip(clicks,impressions)))/\
                (sum(digamma(i + alpha_old+beta_old) - digamma(alpha_old+beta_old) for c,i in zip(clicks,impressions)))


                beta=beta_old*\
                (sum(digamma(i-c + beta_old) - digamma(beta_old) for c,i in zip(clicks,impressions)))/\
                (sum(digamma(i + alpha_old+beta_old) - digamma(alpha_old+beta_old) for c,i in zip(clicks,impressions)))


                #print('alpha {} | {}  beta {} | {}'.format(alpha,alpha_old,beta,beta_old))
                sys.stdout.flush()

                if np.abs(alpha-alpha_old) and np.abs(beta-beta_old)<1e-10:
                    #print('early stop')
                    break

                alpha_old=alpha
                beta_old=beta

            return alpha, beta
        
        imps = np.ones_like(Y)
        alpha, beta = fit_alpha_beta_py(imps, Y) # use scipy's implementation
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