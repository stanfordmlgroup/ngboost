from jax import jit, vmap, grad
import jax.numpy as np
from toolz.functoolz import compose
from scipy.optimize import basinhopping


class Score:
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
    def _fit_marginal(cls, y):
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
    def has(cls, *attributes):
        return all(hasattr(cls, attribute) for attribute in attributes)


class LogScore(Score):
    """
    Generic class for the log scoring rule.

    The log scoring rule is the same as negative log-likelihood: -log(P̂(y)),
    also known as the maximum likelihood estimator. This scoring rule has a default
    method for calculating the Riemannian metric.
    """

    @classmethod
    def _metric(cls, _params, n_mc_samples=100):
        grads = np.stack(
            [cls._d_score(_params, Y) for Y in cls(_params).sample(n_mc_samples)]
        )
        return np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)

    @classmethod
    def build(cls, Dist):

        if not cls.has("_d_score"):
            if cls.has("_score"):
                cls._d_score = jit(vmap(grad(cls._score)))
            elif cls.has("score"):
                cls._score = jit(Dist.parametrize_internally(cls.score))
                cls._d_score = jit(vmap(grad(cls._score)))
            else:
                if Dist.has("_pdf"):
                    _score_scalar = compose(lambda x: -x, np.log, Dist._pdf)
                    cls._score = jit(vmap(_score_scalar))
                    cls._d_score = jit(vmap(grad(_score_scalar)))
                else:
                    raise ValueError(
                        f"Distributions must have _pdf implemented to "
                        f"autogenerate _score and _d_score when using LogScore. "
                        f"{Dist.__name__} has no _pdf method or method from which to "
                        f"generate it (e.g. pdf, _cdf, or cdf)."
                    )


MLE = LogScore


class CRPScore(Score):
    """
    Generic class for the continuous ranked probability scoring rule.
    """

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
