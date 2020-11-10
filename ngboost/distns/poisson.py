"""The NGBoost Poisson distribution and scores"""
import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.stats import poisson as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


def negative_log_likelihood(params, data):
    return -dist.logpmf(np.array(data), params[0]).sum()


class PoissonLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpmf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 1))
        D[:, 0] = self.mu - Y
        return D

    def metric(self):
        FI = np.zeros((self.mu.shape[0], 1, 1))
        FI[:, 0, 0] = self.mu
        return FI


class Poisson(RegressionDistn):
    """
    Implements the Poisson distribution for NGBoost.
    The Poisson distribution has one parameter, mu, which is the mean number of events per interval.
    This distribution has LogScore implemented for it.
    """

    n_params = 1
    scores = [PoissonLogScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        # save the parameters
        self._params = params

        # create other objects that will be useful later
        self.logmu = params[0]
        self.mu = np.exp(self.logmu)
        self.dist = dist(mu=self.mu)

    def fit(Y):
        assert np.equal(
            np.mod(Y, 1), 0
        ).all(), "All Poisson target data must be discrete integers"
        assert np.all([y >= 0 for y in Y]), "Count data must be >= 0"

        # minimize negative log likelihood
        m = minimize(
            negative_log_likelihood,
            x0=np.array([np.mean(Y)]),  # initialized value
            args=(Y,),
            bounds=(Bounds(0, np.max(Y))),
        )
        return np.array([np.log(m.x)])

    def sample(self, m):
        return np.array([self.dist.rvs() for i in range(m)])

    def __getattr__(
        self, name
    ):  # gives us access to Poisson.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"mu": self.mu}
