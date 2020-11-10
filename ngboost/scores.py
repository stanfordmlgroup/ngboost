import numpy as np


class Score:
    def total_score(self, Y, sample_weight=None):
        return np.average(self.score(Y), weights=sample_weight)

    def grad(self, Y, natural=True):
        grad = self.d_score(Y)
        if natural:
            metric = self.metric()
            grad = np.linalg.solve(metric, grad)
        return grad


class LogScore(Score):
    """
    Generic class for the log scoring rule.

    The log scoring rule is the same as negative log-likelihood: -log(P̂(y)),
    also known as the maximum likelihood estimator. This scoring rule has a default
    method for calculating the Riemannian metric.
    """

    def metric(self, n_mc_samples=100):
        grads = np.stack([self.d_score(Y) for Y in self.sample(n_mc_samples)])
        return np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)

    # autofit method from d_score?


MLE = LogScore


class CRPScore(Score):
    """
    Generic class for the continuous ranked probability scoring rule.
    """


CRPS = CRPScore
