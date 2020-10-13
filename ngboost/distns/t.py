import numpy as np
from scipy.stats import t as dist

from ngboost.distns import RegressionDistn
from ngboost.scores import LogScore


class TFixedDFLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def _handle_loc_derivative(self, Y: np.ndarray) -> np.ndarray:
        num = (self.df + 1) * (2 / (self.df * self.var)) * (self.loc - Y)
        den = (2) * (1 + (1 / (self.df * self.var)) * (self.loc - Y) ** 2)
        return -num / den

    def _handle_scale_derivative(self, Y: np.ndarray) -> np.ndarray:
        num = (self.df + 1) * (self.loc - Y) ** 2
        den = (self.df * self.var) + (self.loc - Y) ** 2
        return 1 - (num / den)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2))
        D[:, 0] = self._handle_loc_derivative(Y)
        D[:, 1] = self._handle_scale_derivative(Y)
        return D

    # NOTE: the below metric is wrt scale not log(scale)
    # From https://stats.stackexchange.com/questions/271898/expected-fishers-information-matrix-for-students-t-distribution
    # def metric(self):
    #     FI = np.zeros((self.var.shape[0], 2, 2))
    #     FI[:, 0, 0] = (self.df + 1) / ((self.df + 3) * self.var)
    #     FI[:, 1, 1] = (self.df) / (2 * (self.df + 3) * self.var ** 2)
    #     return FI


class TFixedDF(RegressionDistn):
    """
    Implements the student's t distribution with df=3 for NGBoost.

    The t distribution has two parameters, loc and scale, which are the mean and standard deviation, respectively.
    This distribution only has both LogScore implemented for it.
    """

    n_params = 2
    scores = [TFixedDFLogScore]
    fixed_df = 3.0

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.var = self.scale ** 2
        # fixed df
        self.df = np.ones_like(self.loc) * self.fixed_df
        self.dist = dist(loc=self.loc, scale=self.scale, df=self.df)

    def fit(Y):
        _, m, s = dist.fit(Y, fdf=TFixedDF.fixed_df)
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
