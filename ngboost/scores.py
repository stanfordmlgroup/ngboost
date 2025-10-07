import numpy as np


class Score:
    def total_score(self, Y, sample_weight=None):
        return np.average(self.score(Y), weights=sample_weight)

    def grad(self, Y, natural=True):
        grad = self.d_score(Y)
        if natural:
            metric = self.metric()
            grad = self._natural_gradient(grad, metric)
        return grad

    def _natural_gradient(self, grad, metric):
        """
        Compute natural gradient with robust dimension handling.

        Args:
            grad: Gradient array of shape (n_samples, n_params)
            metric: Metric array of shape (n_samples, n_params, n_params)

        Returns:
            Natural gradient array of shape (n_samples, n_params)
        """
        n_samples, n_params = grad.shape

        # Check if dimensions are compatible
        if metric.shape != (n_samples, n_params, n_params):
            raise ValueError(
                f"Metric shape {metric.shape} is incompatible with gradient shape {grad.shape}. "
                f"Expected metric shape: ({n_samples}, {n_params}, {n_params})"
            )

        # Compute natural gradient for each sample
        result = np.zeros_like(grad)

        for i in range(n_samples):
            try:
                # For each sample, solve: metric[i] * x = grad[i]
                result[i] = np.linalg.solve(metric[i], grad[i])
            except np.linalg.LinAlgError:
                # Handle singular matrix case by using pseudo-inverse
                result[i] = np.linalg.pinv(metric[i]) @ grad[i]

        return result


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
