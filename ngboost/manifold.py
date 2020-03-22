from warnings import warn


def manifold(Score, Distribution):
    """
    Mixes a scoring rule and a distribution together to create the resultant "Reimannian Manifold"
    (thus the name of the function). The resulting object has all the parameters of the distribution 
    can be sliced and indexed like one, and carries the distributions `fit` and `sample` methods, but 
    it also carries the appropriate `total_score` and `grad` methods that are inherited through 
    distribution-specific inheritence of the relevant implementation of the scoring rule
    """

    class Manifold(Distribution.implementation(Score), Distribution):
        pass

    return Manifold
