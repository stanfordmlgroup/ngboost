from jax import jit, vmap, grad
import jax.numpy as np
from toolz.functoolz import partial


class Score:
    @classmethod
    def _total_score(cls, Y, _params, sample_weight=None):
        return np.average(cls.score(Y, _params), weights=sample_weight)

    @classmethod
    def _grad(cls, Y, _params, natural=True):
        grad = cls.d_score(Y, _params)
        if natural:
            metric = cls.metric(_params)
            grad = np.linalg.solve(metric, grad)
        return grad


class LogScore(Score):
    """
    Generic class for the log scoring rule.

    The log scoring rule is the same as negative log-likelihood: -log(P̂(y)),
    also known as the maximum likelihood estimator. This scoring rule has a default
    method for calculating the Riemannian metric.
    """

    @classmethod
    def metric(cls, _params, n_mc_samples=100):
        grads = np.stack(
            [cls.d_score(Y, _params) for Y in cls(_params).sample(n_mc_samples)]
        )
        return np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)

    @classmethod
    def build(cls, Dist):
        ImplementedScore = Dist.implementation(cls)

        class BuiltScore(ImplementedScore):
            if not hasattr(ImplementedScore, "score"):
                score = jit(vmap(Dist._nll))

            if not hasattr(ImplementedScore, "d_score"):
                d_score = jit(vmap(grad(Dist._nll, 1)))

        return BuiltScore

    # autofit method from d_score? Simple gradient descent?


MLE = LogScore


class CRPScore(Score):
    """
    Generic class for the continuous ranked probability scoring rule.
    """


CRPS = CRPScore
