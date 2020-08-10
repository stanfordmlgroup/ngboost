from scipy.stats import betabinom as dist
import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import digamma, polygamma
from array import array
import sys

class BetaBernoulliLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpmf(Y)

    def d_alpha(self, Y):
        return self.alpha * (
            digamma(self.alpha + self.beta)
            + digamma(Y + self.alpha)
            - digamma(self.alpha + self.beta + 1)
            - digamma(self.alpha)
        )
    
    def d_beta(self, Y):
        return self.beta * (
            digamma(self.alpha + self.beta)
            + digamma(-Y + self.beta + 1)
            - digamma(self.alpha + self.beta + 1)
            - digamma(self.beta)
        )
    
    def d_alpha_alpha(self, Y):
        return (
            (polygamma(1, Y + self.alpha) +
             polygamma(1, self.alpha + self.beta) -
             polygamma(1, self.alpha + self.beta + 1) -
             polygamma(1, self.alpha)
             ) * self.alpha**2 + self.d_alpha(Y)
        ) 
    
    def d_alpha_beta(self, Y):
        return (
            self.alpha * self.beta * (
                polygamma(1, self.alpha + self.beta) -
                polygamma(1, self.alpha + self.beta + 1)
            )
        )
    
    def d_beta_beta(self, Y):
        return (
            (polygamma(1, -Y + self.beta + 1) + 
             polygamma(1, self.alpha + self.beta) - 
             polygamma(1, self.alpha + self.beta + 1) - 
             polygamma(1, self.beta)
            ) * self.beta**2 + self.d_beta(Y)
        )
    def d_score(self, Y):
        D = np.zeros(
            (len(Y), 2)
        )  # first col is dS/d(log(α)), second col is dS/d(log(β))
        D[:, 0] = -self.d_alpha(Y)
        D[:, 1] = -self.d_beta(Y)
        return D

    # Variance
    def metric(self):
        FI = np.zeros((self.alpha.shape[0], 2, 2))
        FI[:, 0, 0] = (
            self.d_alpha(0)*self.d_alpha(0)*self.dist.pmf(0) +
            self.d_alpha(1)*self.d_alpha(1)*self.dist.pmf(1)
        ) 
        # FI[:, 1, 0] = (
        #     self.d_alpha(0)*self.d_beta(0)*self.dist.pmf(0) +
        #     self.d_alpha(1)*self.d_beta(1)*self.dist.pmf(1)
        # ) 
        # FI[:, 0, 1] = (
        #     self.d_alpha(0)*self.d_beta(0)*self.dist.pmf(0) +
        #     self.d_alpha(1)*self.d_beta(1)*self.dist.pmf(1)
        # ) 
        FI[:, 1, 1] = (
            self.d_beta(0)*self.d_beta(0)*self.dist.pmf(0) +
            self.d_beta(1)*self.d_beta(1)*self.dist.pmf(1)
        ) 
        return FI
    
    # Hessian
    # def metric(self):
    #     FI = np.zeros((self.alpha.shape[0], 2, 2))
    #     FI[:, 0, 0] = self.d_alpha_alpha(0) * self.dist.pmf(0) + self.d_alpha_alpha(1) * self.dist.pmf(1)
    #     # FI[:, 1, 0] = self.d_alpha_beta(0) * self.dist.pmf(0) + self.d_alpha_beta(1) * self.dist.pmf(1)
    #     # FI[:, 0, 1] = self.d_alpha_beta(0) * self.dist.pmf(0) + self.d_alpha_beta(1) * self.dist.pmf(1)
    #     FI[:, 1, 1] = self.d_beta_beta(0) * self.dist.pmf(0) + self.d_beta_beta(1) * self.dist.pmf(1)
    #     return FI

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

    def fit(Y):
        def fit_alpha_beta_py(success, alpha0=1.5, beta0=5, niter=1000):
            # based on https://github.com/lfiaschi/fastbetabino/blob/master/fastbetabino.pyx

            # This optimisation works for Beta-Binomial distribution in general. 
            # For Beta-Bernoulli it's simplified by fixing the trials to 1.
            trials = np.ones_like(Y)

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

        alpha, beta = fit_alpha_beta_py(Y)
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
