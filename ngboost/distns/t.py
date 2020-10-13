from ngboost.distns import RegressionDistn
from ngboost.scores import LogScore
import numpy as np
from scipy.stats import t as dist
import scipy.stats as ss

ss.t.fit()


class TFixedDFLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def _handle_loc_derivative(self, Y: np.ndarray) -> np.ndarray:
        return (self.loc - Y) / self.var

    def _handle_scale_derivative(self, Y: np.ndarray) -> np.ndarray:
        return 1 - ((self.loc - Y) ** 2) / self.var

    def d_score(self, Y):
        D = np.zeros((len(Y), 2))
        D[:, 0] = self._handle_loc_derivative(Y)
        D[:, 1] = self._handle_scale_derivative(Y)
        return D

    def metric(self):
        FI = np.zeros((self.var.shape[0], 2, 2))
        FI[:, 0, 0] = (1 + self.df) / ((self.df + 3) * self.var)
        FI[:, 1, 1] = (self.df) / (2 * (self.df + 3) * self.var ** 2)
        return FI


class TFixedDF(RegressionDistn):
    """
    Implements the student's t distribution with df=3 for NGBoost.

    The t distribution has two parameters, loc and scale, which are the mean and standard deviation, respectively.
    This distribution only has both LogScore implemented for it.
    """

    n_params = 2
    scores = [TFixedDFLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.var = self.scale ** 2
        # fixed df
        self.df = np.ones_like(self.loc) * 3
        self.dist = dist(loc=self.loc, scale=self.scale, df=self.df)

    def fit(Y):
        _, m, s = dist.fit(Y, fdf=3)
        return np.array([m, np.log(s)])

    def sample(self, m):
        return np.array([self.rvs() for i in range(m)])

    def __getattr__(
        self, name
    ):  # gives us Normal.mean() required for RegressionDist.predict()
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}