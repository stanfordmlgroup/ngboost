"""The NGBoost base distribution"""
from warnings import warn

from jax import grad, vmap
import jax.numpy as np
from toolz.functoolz import compose

import scipy as sp
from inspect import signature


def n_params(dist):
    return len(signature(dist.params_to_internal).parameters)


class Distn:
    # functions that are like _fn operate on the internal array parametrization

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

    @property
    def params(self):
        return self.params_to_user(self._params)

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
    def __init__(self, params):
        super().__init__(params)
        self._cdf = lambda y, params: self.cdf(
            y, **self.params_to_user(params)
        )  # y, params -> quantile (vectorized)
        self._pdf = grad(self._cdf)  # y, params -> likelihood (scalar)
        self._logpdf = compose(
            np.log, self._pdf
        )  # y, params -> log-likelihood (scalar)

    @classmethod
    def derive_cdf(cls):
        return lambda y, params: cls.cdf(y, **cls.params_to_user(params))

    @classmethod
    def derive_pdf(cls):
        cdf = cls.derive_cdf()
        return grad(cdf)

    @classmethod
    def derive_logpdf(cls):
        pdf = cls.derive_pdf()
        return compose(np.log, pdf)

    def predict(self):  # predictions for regression are typically conditional means
        return self.mean()


class ClassificationDistn(Distn):
    def predict(self):  # returns class assignments
        return np.argmax(self.class_probs(), 1)
