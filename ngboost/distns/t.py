"""The NGBoost Student T distribution and scores"""
import numpy as np
from scipy.special import digamma
from scipy.stats import t as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


class TLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def _handle_loc_derivative(self, Y: np.ndarray) -> np.ndarray:
        num = (self.df + 1) * (Y - self.loc)
        den = (self.df * self.var) + (Y - self.loc) ** 2
        return -(num / den)

    def _handle_scale_derivative(self, Y: np.ndarray) -> np.ndarray:
        num = (self.df + 1) * (Y - self.loc) ** 2
        den = (self.df * self.var) + (Y - self.loc) ** 2
        return 1 - (num / den)

    def _handle_df_derivative(self, Y: np.ndarray) -> np.ndarray:
        term_1 = (self.df / 2) * digamma((self.df + 1) / 2)
        term_2 = (-self.df / 2) * digamma((self.df) / 2)
        term_3 = -1 / 2
        term_4_1 = (-self.df / 2) * np.log(
            1 + ((Y - self.loc) ** 2) / (self.df * self.var)
        )
        term_4_2_num = (self.df + 1) * (Y - self.loc) ** 2
        term_4_2_den = (
            2
            * (self.df * self.var)
            * (1 + ((Y - self.loc) ** 2) / (self.df * self.var))
        )
        return -(term_1 + term_2 + term_3 + term_4_1 + term_4_2_num / term_4_2_den)

    def d_score(self, Y):
        D = np.zeros((len(Y), 3))
        D[:, 0] = self._handle_loc_derivative(Y)
        D[:, 1] = self._handle_scale_derivative(Y)
        D[:, 2] = self._handle_df_derivative(Y)
        return D


class T(RegressionDistn):
    """
    Implements the student's t distribution for NGBoost.

    The t distribution has two parameters, loc and scale, which are
    the mean and standard deviation, respectively.
    This distribution only has LogScore implemented for it.
    """

    n_params = 3
    scores = [TLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.var = self.scale ** 2
        self.df = np.exp(params[2])
        self.dist = dist(loc=self.loc, scale=self.scale, df=self.df)

    def fit(Y):
        df, m, s = dist.fit(Y, fdf=TFixedDf.fixed_df)
        return np.array([m, np.log(s), np.log(df)])

    def sample(self, m):
        return np.array([self.rvs() for i in range(m)])

    def __getattr__(self, name):
        return getattr(self.dist, name, None)

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}


class TFixedDfLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def _handle_loc_derivative(self, Y: np.ndarray) -> np.ndarray:
        num = (self.df + 1) * (Y - self.loc)
        den = (self.df * self.var) + (Y - self.loc) ** 2
        return -(num / den)

    def _handle_scale_derivative(self, Y: np.ndarray) -> np.ndarray:
        num = (self.df + 1) * (Y - self.loc) ** 2
        den = (self.df * self.var) + (Y - self.loc) ** 2
        return 1 - (num / den)

    def d_score(self, Y):
        D = np.zeros((len(Y), 2))
        D[:, 0] = self._handle_loc_derivative(Y)
        D[:, 1] = self._handle_scale_derivative(Y)
        return D

    def metric(self):
        FI = np.zeros((self.var.shape[0], 2, 2))
        FI[:, 0, 0] = (self.df + 1) / ((self.df + 3) * self.var)
        FI[:, 1, 1] = (self.df) / (2 * (self.df + 3) * self.var)
        return FI


class TFixedDf(RegressionDistn):
    """
    Implements the student's t distribution with df=3 for NGBoost.

    The t distribution has two parameters, loc and scale, which are the
    mean and standard deviation, respectively.
    This distribution only has LogScore implemented for it.
    """

    n_params = 2
    scores = [TFixedDfLogScore]
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
        _, m, s = dist.fit(Y, fdf=TFixedDf.fixed_df)
        return np.array([m, np.log(s)])

    def sample(self, m):
        return np.array([self.rvs() for i in range(m)])

    def __getattr__(self, name):
        return getattr(self.dist, name, None)

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}


class TFixedDfFixedVarLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def _handle_loc_derivative(self, Y: np.ndarray) -> np.ndarray:
        num = (self.df + 1) * (2 / (self.df * self.var)) * (Y - self.loc)
        den = (2) * (1 + (1 / (self.df * self.var)) * (Y - self.loc) ** 2)
        return -num / den

    def d_score(self, Y):
        D = np.zeros((len(Y), 1))
        D[:, 0] = self._handle_loc_derivative(Y)
        return D

    def metric(self):
        FI = np.zeros((self.var.shape[0], 1, 1))
        FI[:, 0, 0] = (self.df + 1) / ((self.df + 3) * self.var)
        return FI


class TFixedDfFixedVar(RegressionDistn):
    """
    Implements the student's t distribution with df=3 and var=1 for NGBoost.

    The t distribution has two parameters, loc and scale, which are the
    mean and standard deviation, respectively.
    This distribution only has LogScore implemented for it.
    """

    n_params = 1
    scores = [TFixedDfFixedVarLogScore]
    fixed_df = 3.0

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        # fixed var
        self.scale = np.ones_like(self.loc)
        self.var = np.ones_like(self.loc)
        # fixed df
        self.df = np.ones_like(self.loc) * self.fixed_df
        self.dist = dist(loc=self.loc, scale=self.scale, df=self.df)

    def fit(Y):
        _, m, _ = dist.fit(Y, fdf=TFixedDfFixedVar.fixed_df)
        return m

    def sample(self, m):
        return np.array([self.rvs() for i in range(m)])

    def __getattr__(self, name):
        return getattr(self.dist, name, None)

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}
