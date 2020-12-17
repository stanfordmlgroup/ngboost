from jax import jit, vmap, grad
import jax.numpy as np
from toolz.functoolz import partial
from scipy.optimize import basinhopping


class Score:
    @classmethod
    def _total_score(cls, Y, _params, sample_weight=None):
        return np.average(cls._score(Y, _params), weights=sample_weight)

    @classmethod
    def _grad(cls, Y, _params, natural=True):
        grad = cls._d_score(Y, _params)
        if natural:
            metric = cls._metric(_params)
            grad = np.linalg.solve(metric, grad)

        return grad

    @classmethod
    def _fit_marginal(cls, y):  # may be generalized or improved with global search
        n = len(y)
        return basinhopping(
            func=lambda params: np.average(
                cls._score(y, np.ones((n, cls.n_params())) * params)
            ),
            x0=np.ones((cls.n_params(),)) * np.mean(y),
            stepsize=1000,
            niter_success=5,
            minimizer_kwargs=dict(
                jac=lambda params: np.average(
                    cls._d_score(y, np.ones((n, cls.n_params())) * params), axis=0,
                )
            ),
        ).x


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
            [cls._d_score(Y, _params) for Y in cls(_params).sample(n_mc_samples)]
        )
        return np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)

    @classmethod
    def build(cls, Dist):
        ImplementedScore = Dist.implementation(cls)

        class BuiltScore(ImplementedScore):
            if not hasattr(ImplementedScore, "_score"):
                _score = jit(vmap(Dist._nll))

            if not hasattr(ImplementedScore, "_d_score"):
                _d_score = jit(vmap(grad(Dist._nll, 1)))

        return BuiltScore


MLE = LogScore


class CRPScore(Score):
    """
    Generic class for the continuous ranked probability scoring rule.
    """

    @classmethod
    def build(cls, Dist):
        ImplementedScore = Dist.implementation(cls)

        class BuiltScore(ImplementedScore):
            if not hasattr(ImplementedScore, "_score"):
                raise ValueError(
                    "Children of CRPSScore must be implemented with a `_score` method."
                )

            if not hasattr(ImplementedScore, "_d_score"):
                _d_score = jit(vmap(grad(cls._score, 1)))

        return BuiltScore


CRPS = CRPScore
