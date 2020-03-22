from warnings import warn

import numpy as np
from ngboost.helpers import Y_from_censored


class Distn(object):
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
                f"Using Dist={Score.__name__} is unnecessary. NGBoost automatically selects the correct implementation when LogScore or CRPScore is used"
            )
            return Score
        else:
            try:
                return {S.__bases__[-1]: S for S in scores}[Score]
            except KeyError as err:
                raise ValueError(
                    f"The scoring rule {Score.__name__} is not implemented for the {cls.__name__} distribution."
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
