"""
    Extra functions useful for Distns
"""
from ngboost.distns import RegressionDistn

# pylint: disable=too-few-public-methods


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

        def fit(Y):
            """
                Parameters:
                    Y : a object with keys {time, event}, each containing an array
            """
            return Dist.fit(Y["Time"])

    return SurvivalDistn
