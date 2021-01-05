from jax import jit, vmap, grad
import jax.numpy as np

from toolz.functoolz import compose
from scipy.optimize import basinhopping
from warnings import warn

from jax.ops import index_update, index

import pdb


class Score:
    def total_score(self, Y, sample_weight=None):
        return self._total_score(self._params, Y, sample_weight)

    def grad(self, Y, natural=True):
        return self._grad(self._params, Y, natural)

    @classmethod
    def _total_score(cls, _params, Y, sample_weight=None):
        return np.average(cls._score(_params, Y), weights=sample_weight)

    @classmethod
    def _grad(cls, _params, Y, natural=True):
        grad = cls._d_score(_params, Y)
        if natural:
            metric = cls._metric(_params)
            grad = np.linalg.solve(metric, grad)

        return grad

    @classmethod
    def _fit_marginal_obs(cls, y):
        n = len(y)
        return basinhopping(
            func=lambda _params: np.average(
                cls._score(np.ones((n, cls.n_params())) * _params, y)
            ),
            x0=np.ones((cls.n_params(),)) * np.mean(y),
            stepsize=1000,
            niter_success=5,
            minimizer_kwargs=dict(
                jac=lambda _params: np.average(
                    cls._d_score(np.ones((n, cls.n_params())) * _params, y), axis=0,
                )
            ),
        ).x

    @classmethod
    def _fit_marginal(cls, Y):
        return cls._fit_marginal_obs(Y.observed)

    @classmethod
    def has(cls, *attributes):
        return all(hasattr(cls, attribute) for attribute in attributes)


class LogScore(Score):
    """
    Generic class for the log scoring rule.

    The log scoring rule is the same as negative log-likelihood: -log(P̂(y)),
    also known as the maximum likelihood estimator. This scoring rule has a default
    method for calculating the Riemannian metric.
    """

    from_scratch = True  # this score can be auto-generated from distribution methods
    built = False

    @classmethod
    def _score(cls, _params, Y):
        result = np.zeros(Y.shape)

        result = index_update(
            result, index[Y.ix_obs], cls._score_obs(_params[Y.ix_obs, :], Y.observed),
        )
        result = index_update(
            result, index[Y.ix_cen], cls._score_cen(_params[Y.ix_cen, :], Y.censored),
        )

        return result

    @classmethod
    def _d_score(cls, _params, Y):
        result = np.zeros(_params.shape)

        result = index_update(
            result,
            index[Y.ix_obs, :],
            cls._d_score_obs(_params[Y.ix_obs, :], Y.observed),
        )
        result = index_update(
            result,
            index[Y.ix_cen, :],
            cls._d_score_cen(_params[Y.ix_cen, :], Y.censored),
        )

        return result

    @classmethod
    def _metric(cls, _params, n_mc_samples=100):
        grads = np.stack(
            [cls._d_score_obs(_params, Y) for Y in cls(_params).sample(n_mc_samples)]
        )
        return np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)

    @classmethod
    def build(cls):
        Dist, Score = cls.__bases__

        # add support for providing, e.g. `score_obs()` (requires some chain-rule transform)?
        # unclear numerical/speed benefit.

        ### Build necessary methods

        if cls.built:
            return None

        if not cls.has("_cdf"):
            cls._cdf = cls.parametrize_internally(cls.cdf)
            warn(f"Auto-generating _cdf() method from cdf()")

        if not cls.has("pdf"):
            cls.pdf = lambda y, **params: grad(lambda y, params: cls.cdf(y, **params))(
                y, params
            )
            warn(f"Auto-generating pdf() method from cdf()")

        if not cls.has("_pdf"):
            if cls.has("pdf"):
                cls._pdf = cls.parametrize_internally(cls.pdf)
                warn(f"Auto-generating _pdf() method from pdf()")
            else:
                cls._pdf = grad(cls._cdf, 1)  # grad w.r.t. y, not params
                warn(f"Auto-generating _pdf() method from _cdf()")

        if not cls.has("_score_obs"):
            cls._score_obs = compose(lambda x: -x, np.log, cls._pdf)
            warn(f"Auto-generating _score_obs() method from _pdf()")

        if not cls.has("_d_score_obs"):
            cls._d_score_obs = grad(cls._score_obs)
            warn(f"Auto-generating _d_score_obs() method from _score_obs()")

        if not cls.has("_score_cen"):
            cls._score_cen = lambda _params, interval: -np.log(
                cls._cdf(_params, interval[..., 1])
                - cls._cdf(_params, interval[..., 0])
            )
            warn(f"Auto-generating _score_cen() method from _cdf()")

        if not cls.has("_d_score_cen"):
            if cls.has("_d_cdf") and cls.has("_cdf"):
                cls._d_score_cen = lambda _params, interval: -(
                    cls._d_cdf(_params, interval[..., 1])
                    - cls._d_cdf(_params, interval[..., 0])
                ) / (
                    cls._cdf(_params, interval[..., 1])
                    - cls._cdf(_params, interval[..., 0])
                )
                warn(f"Auto-generating _d_score_cen() method from _d_cdf() and _cdf")
            else:
                cls._d_score_cen = grad(cls._score_cen)
                warn(f"Auto-generating _d_score_cen() method from _score_cen()")

        ### Vectorize what needs vectorizing

        test_params = np.zeros((2, cls.n_params()))
        test_y_obs = np.zeros(2)
        test_y_cen = np.array([[-1, 1], [-1, 1]], dtype=float)

        try:
            _ = cls._score_obs(test_params, test_y_obs)
        except TypeError:
            cls._score_obs = jit(vmap(cls._score_obs))
            warn(f"Vectorizing _score_obs")

        try:
            _ = cls._d_score_obs(test_params, test_y_obs)
        except TypeError:
            cls._d_score_obs = jit(vmap(cls._d_score_obs))
            warn(f"Vectorizing _d_score_obs")

        try:
            _ = cls._score_cen(test_params, test_y_cen)
        except TypeError:
            cls._score_cen = jit(vmap(cls._score_cen))
            warn(f"Vectorizing _score_cen")

        try:
            _ = cls._d_score_cen(test_params, test_y_cen)
        except TypeError:
            cls._d_score_cen = jit(vmap(cls._d_score_cen))
            warn(f"Vectorizing _d_score_cen")

        cls.built = True


MLE = LogScore


class CRPScore(Score):
    """
    Generic class for the continuous ranked probability scoring rule.
    """

    from_scratch = False

    @classmethod
    def build(cls, Dist):

        if not cls.has("_d_score"):
            if cls.has("_score"):
                cls._d_score = jit(vmap(grad(cls._score)))
            elif cls.has("score"):
                cls._score = Dist.parametrize_internally(cls.score)
                cls._d_score = jit(vmap(grad(cls._score)))
            else:
                raise ValueError(
                    "Implementations of CRPSScore must have a `_score` or `score` method."
                )


CRPS = CRPScore
