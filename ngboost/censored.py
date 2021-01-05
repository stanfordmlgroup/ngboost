import jax.numpy as np


class CensoredOutcome:
    def __init__(self, observed_all, censored_all):
        self.observed_all = observed_all
        self.censored_all = censored_all

        self.ix_cen = np.isnan(self.observed_all)
        self.ix_obs = ~self.ix_cen

        self.observed = self.observed_all[self.ix_obs]
        self.censored = self.censored_all[self.ix_cen, :]

    @property
    def shape(self):
        return self.observed_all.shape

    def __getitem__(self, index):
        return CensoredOutcome(
            observed_all=self.observed_all[index],
            censored_all=self.censored_all[index, :],
        )

    def __len__(self):
        return len(self.observed_all)

    # @classmethod
    # def right_censored(cls, T, E):
    #     return CensoredOutcome([t if e else (t, np.inf) for t, e in zip(T, E)])

    # @classmethod
    # def left_censored(cls, T, E):
    #     return CensoredOutcome([t if e else (-np.inf, t) for t, e in zip(T, E)])

    # @classmethod
    # def censor_administratively(cls, Y, lower=-np.inf, upper=np.inf):
    #     """ For testing """
    #     return CensoredOutcome(
    #         [y if y < lower or upper < y else (float(lower), float(upper)) for y in Y]
    #     )
