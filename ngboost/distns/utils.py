"""
    Extra functions useful for Distns
"""

from inspect import signature

from ngboost.distns import RegressionDistn

# pylint: disable=too-few-public-methods


def fit_marginal(Dist, Y, sample_weight=None):
    """
    Fits the marginal distribution of Y, for a distribution's initial parameters.

    A distribution whose fit takes sample_weight gets the weights. For any other, rows
    of zero weight are left out and the remaining weights do not affect the fit.
    """
    if sample_weight is None:
        return Dist.fit(Y)
    if "sample_weight" in signature(Dist.fit).parameters:
        return Dist.fit(Y, sample_weight=sample_weight)
    return Dist.fit(Y[sample_weight > 0])


def SurvivalDistnClass(Dist: RegressionDistn):
    """
    Creates a new dist class from a given dist. The new class has its implemented scores

    Parameters:
        Dist (RegressionDistn): a Regression distribution with censored scores implemented.

    Output:
        SurvivalDistn class, this is only used for Survival regression
    """

    class SurvivalDistn(Dist):
        # Stores the original distribution for pickling purposes
        _basedist = Dist
        scores = (
            Dist.censored_scores
        )  # set to the censored versions of the scores implemented for dist

        def fit(Y, sample_weight=None):
            """
            Parameters:
                Y : a object with keys {time, event}, each containing an array
                sample_weight : optional weight of each row
            """
            return fit_marginal(Dist, Y["Time"], sample_weight)

    return SurvivalDistn
