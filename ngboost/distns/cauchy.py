"""The NGBoost Cauchy distribution and scores"""

from ngboost.distns.t import (
    TFixedDf,
    TFixedDfFixedVar,
    TFixedDfFixedVarLogScore,
    TFixedDfLogScore,
)

CauchyLogScore = TFixedDfLogScore


class Cauchy(TFixedDf):
    fixed_df = 1.0

    # scipy.stats.cauchy takes loc and scale only, so params has no df
    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}


CauchyFixedVarLogScore = TFixedDfFixedVarLogScore


class CauchyFixedVar(TFixedDfFixedVar):
    fixed_df = 1.0

    # scipy.stats.cauchy takes loc and scale only, so params has no df
    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}
