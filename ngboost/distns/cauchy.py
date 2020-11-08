from ngboost.distns.t import TFixedDfFixedVar, TFixedDf, TFixedDfFixedVarLogScore, TFixedDfLogScore

CauchyLogScore = TFixedDfLogScore


class Cauchy(TFixedDf):
    fixed_df = 1.0


CauchyFixedVarLogScore = TFixedDfFixedVarLogScore


class CauchyFixedVar(TFixedDfFixedVar):
    fixed_df = 1.0
