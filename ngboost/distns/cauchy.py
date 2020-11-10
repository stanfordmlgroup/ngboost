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


CauchyFixedVarLogScore = TFixedDfFixedVarLogScore


class CauchyFixedVar(TFixedDfFixedVar):
    fixed_df = 1.0
