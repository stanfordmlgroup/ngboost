"""The NGBoost base distribution"""
from warnings import warn

from jax import grad, jacfwd
import jax.numpy as np
import scipy as sp
from inspect import signature


def n_params(dist):
    return len(signature(dist.transform_params).parameters)


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
        self._params = self.untransform_params(params)

    def __getitem__(self, key):
        return self.__class__(self.transform_params(self._params[:, key]))

    def __len__(self):
        return self._params.shape[1]

    @property
    def params(self):
        return _params

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


class RegressionDistn(Distn):
    def predict(self):  # predictions for regression are typically conditional means
        return self.mean()

    @classmethod
    def pdf(cls, Y, **kwargs):
        return np.diag(jacfwd(cls.cdf)(Y, **kwargs))  # might be inefficient...

    @classmethod
    def logpdf(cls, Y, **kwargs):
        return np.log(cls.pdf(Y, **kwargs))


class ClassificationDistn(Distn):
    def predict(self):  # returns class assignments
        return np.argmax(self.class_probs(), 1)
