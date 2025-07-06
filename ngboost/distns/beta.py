"""The NGBoost Beta distribution and scores"""
import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import beta as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class BetaLogScore(LogScore):
    """Log score for the Beta distribution."""

    def score(self, Y):
        """Calculate the log score for the Beta distribution."""
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        """Calculate the derivative of the log score with respect to the parameters."""
        D = np.zeros(
            (len(Y), 2)
        )  # first col is dS/d(log(a)), second col is dS/d(log(b))
        D[:, 0] = -self.a * (digamma(self.a + self.b) - digamma(self.a) + np.log(Y))
        D[:, 1] = -self.b * (digamma(self.a + self.b) - digamma(self.b) + np.log(1 - Y))
        return D

    def metric(self):
        """Return the Fisher Information matrix for the Beta distribution."""
        FI = np.zeros((self.a.shape[0], 2, 2))
        trigamma_a_b = polygamma(1, self.a + self.b)
        FI[:, 0, 0] = self.a**2 * (polygamma(1, self.a) - trigamma_a_b)
        FI[:, 0, 1] = -self.a * self.b * trigamma_a_b
        FI[:, 1, 0] = -self.a * self.b * trigamma_a_b
        FI[:, 1, 1] = self.b**2 * (polygamma(1, self.b) - trigamma_a_b)
        return FI


class Beta(RegressionDistn):
    """
    Implements the Beta distribution for NGBoost.

    The Beta distribution has two parameters, a and b.
    The scipy loc and scale parameters are held constant for this implementation.
    LogScore is supported for the Beta distribution.
    """

    n_params = 2
    scores = [BetaLogScore]  # will implement this later

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self._params = params

        # create other objects that will be useful later
        self.log_a = params[0]
        self.log_b = params[1]
        self.a = np.exp(params[0])  # since params[0] is log(a)
        self.b = np.exp(params[1])  # since params[1] is log(b)
        self.dist = dist(a=self.a, b=self.b)

    @staticmethod
    def fit(Y):
        """Fit the distribution to the data."""
        # Use scipy's beta distribution to fit the parameters
        # pylint: disable=unused-variable
        a, b, loc, scale = dist.fit(Y, floc=0, fscale=1)
        return np.array([np.log(a), np.log(b)])

    def sample(self, m):
        """Sample from the distribution."""
        return np.array([self.dist.rvs() for i in range(m)])

    def __getattr__(
        self, name
    ):  # gives us access to Beta.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        """Return the parameters of the Beta distribution."""
        return {"a": self.a, "b": self.b}
