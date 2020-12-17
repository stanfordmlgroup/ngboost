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
    Dist.build()
    ImplementedScore = Dist.find_implementation(Score)
    ImplementedScore.build(Dist)

    class Manifold(Dist, ImplementedScore):
        def total_score(self, Y, sample_weight=None):
            return self._total_score(self._params, Y, sample_weight)

        def grad(self, Y, natural=True):
            return self._grad(self._params, Y, natural)

    return Manifold
