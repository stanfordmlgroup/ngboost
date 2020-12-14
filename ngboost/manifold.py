def build_distribution(Dist):
    """ Derive/define methods which the developer may not have provided """

    class BuiltDist(Dist):
        if not hasattr(Dist, "_cdf"):
            _cdf = Dist.derive_cdf()

        if not hasattr(Dist, "_pdf"):
            _pdf = Dist.derive_pdf()

        if not hasattr(Dist, "_logpdf"):
            _logpdf = Dist.derive_logpdf()

    return BuiltDist


def build_score(Score, BuiltDist):
    ImplementedScore = BuiltDist.implementation(Score)

    class BuiltScore(ImplementedScore):
        if not hasattr(ImplementedScore, "score"):
            score = ImplementedScore.derive_score(BuiltDist)

        if not hasattr(ImplementedScore, "d_score"):
            d_score = ImplementedScore.derive_d_score(BuiltDist)

    return BuiltScore


def manifold(Score, Dist):
    """
    Mixes a scoring rule and a distribution together to create the resultant
    "Reimannian Manifold" (thus the name of the function). The resulting object has
    all the parameters of the distribution can be sliced and indexed like one, and carries
    the distributions `fit` and `sample` methods, but it also carries the appropriate
    `total_score` and `grad` methods that are inherited through distribution-specific
    inheritence of the relevant implementation of the scoring rule
    """

    # pylint: disable=too-few-public-methods
    BuiltDist = build_distribution(Dist)
    BuiltScore = build_score(Score, BuiltDist)

    class Manifold(BuiltScore, BuiltDist):
        def total_score(self, Y, sample_weight=None):
            return BuiltScore.total_score(Y, self._params, sample_weight)

        def grad(self, Y, natural=True):
            return BuiltScore.grad(Y, self._params, natural)

    return Manifold
