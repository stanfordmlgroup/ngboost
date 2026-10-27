"""The NGBoost base distribution"""

from warnings import warn

import numpy as np

from ngboost.helpers import Y_from_censored


class Distn:
    """
    User should define:
    - __init__(params) to hold self.params_ = params
    - X_scoring(self, Y)
    - D_X_scoring(self, Y)
    - sample(self, n)
    - fit(Y)
    - predict(self) mean, mode, whatever (method to call for point prediction
    """

    def __init__(self, params):
        self._params = params

    def __getitem__(self, key):
        return self.__class__(self._params[:, key])

    def __len__(self):
        return self._params.shape[1]

    @classmethod
    def implementation(cls, Score, scores=None):
        """
        Finds the distribution-appropriate implementation of Score
        (using the provided scores if cls.scores is empty)
        """
        if scores is None:
            scores = cls.scores
        if Score in scores:
            warn(
                f"Using Dist={Score.__name__} is unnecessary. "
                "NGBoost automatically selects the correct implementation "
                "when LogScore or CRPScore is used"
            )
            return Score
        try:
            return {S.__bases__[-1]: S for S in scores}[Score]
        except KeyError as err:
            raise ValueError(
                f"The scoring rule {Score.__name__} is not "
                f"implemented for the {cls.__name__} distribution."
            ) from err

    @classmethod
    def uncensor(cls, Score):
        DistScore = cls.implementation(Score, cls.censored_scores)

        class UncensoredScore(DistScore, DistScore.__base__):
            def score(self, Y):
                return super().score(Y_from_censored(Y))

            def d_score(self, Y):
                return super().d_score(Y_from_censored(Y))

        class DistWithUncensoredScore(cls):
            scores = [UncensoredScore]

        return DistWithUncensoredScore


class RegressionDistn(Distn):
    def predict(self):  # predictions for regression are typically conditional means
        return self.mean()


class ClassificationDistn(Distn):
    def predict(self):  # returns class assignments
        return np.argmax(self.class_probs(), 1)


def deprecated_var(note=""):
    """
    Makes the deprecated var property of the Normal and T distributions.

    The property returns self._var, the squared scale that the scores use, and
    gives a FutureWarning. It hides the var() method of scipy.stats that
    __getattr__ would give. Removing it makes dist.var() that method.
    """

    def var(self):
        warn(
            f"{type(self).__name__}.var is deprecated. {note}In a future release, "
            "var will be the var() method of scipy.stats, as for the other "
            "distributions. Use dist.scale ** 2 for the current values.",
            FutureWarning,
            stacklevel=2,
        )
        return self._var  # pylint: disable=protected-access

    return property(var)
