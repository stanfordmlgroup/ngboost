from jax import jit, vmap, grad
import jax.numpy as np
from toolz.functoolz import partial

import pdb


class Score:
    @classmethod
    def total_score(cls, Y, _params, sample_weight=None):
        # pdb.set_trace()
        return np.average(cls.score(Y, _params), weights=sample_weight)

    @classmethod
    def grad(cls, Y, _params, natural=True):
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

    # def __init__(self):
    #     self.score = jit(lambda y: vmap(self._logpdf)(y, self._params))
    #     self.d_score = jit(lambda y: vmap(grad(self._logpdf, 1))(y, self._params))

    @classmethod
    def derive_score(cls, Dist):
        return jit(vmap(Dist._logpdf))

    @classmethod
    def derive_d_score(cls, Dist):
        return jit(vmap(grad(Dist._logpdf, 1)))

    @classmethod
    def metric(cls, _params, n_mc_samples=100):
        grads = np.stack([self.d_score(Y, _params) for Y in self.sample(n_mc_samples)])
        return np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)

    # autofit method from d_score?


MLE = LogScore


class CRPScore(Score):
    """
    Generic class for the continuous ranked probability scoring rule.
    """


CRPS = CRPScore
