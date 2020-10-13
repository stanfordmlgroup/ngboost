from .t import TFixedDFLogScore, TFixedDF

CauchyLogScore = TFixedDFLogScore


class Cauchy(TFixedDF):
    fixed_df = 1.0
