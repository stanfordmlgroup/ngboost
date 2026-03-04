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
        grad = np.asarray(grad)
        metric = np.asarray(metric)

        if grad.ndim != 2:
            raise ValueError(
                f"Expected gradient to be 2D, got shape {grad.shape} with ndim={grad.ndim}"
            )

        if metric.ndim != 3:
            raise ValueError(
                f"Expected metric to be 3D, got shape {metric.shape} with ndim={metric.ndim}"
            )

        # Ensure the batch dimension aligns between grad and metric
        if metric.shape[0] != grad.shape[0]:
            if metric.shape[0] == grad.shape[1] and metric.shape[1:] == (
                grad.shape[0],
                grad.shape[0],
            ):
                grad = np.swapaxes(grad, 0, 1)
            else:
                msg = (
                    f"Metric shape {metric.shape} is incompatible with gradient "
                    f"shape {grad.shape}. Expected metric shape to align with grad "
                    "(n_samples, n_params, n_params)."
                )
                raise ValueError(msg)

        n_samples, n_params = grad.shape

        # Final shape check on the metric tensor
        if metric.shape[1:] != (n_params, n_params):
            raise ValueError(
                f"Metric shape {metric.shape} is incompatible with gradient shape {grad.shape}. "
                f"Expected trailing metric dimensions {(n_params, n_params)}."
            )

        # Fast vectorized path – falls back to per-sample solve if singular
        try:
            solved = np.linalg.solve(metric, grad[..., None])[..., 0]
            return solved
        except np.linalg.LinAlgError:
            pass

        # Fall back to robust per-sample computation
        result = np.zeros_like(grad)
        for i in range(n_samples):
            try:
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


class ScoreMatchingScore(Score):
    """
    Generic class for the Score Matching Score by Aapo Hyv{\"a}rinen
    Reference http://www.jmlr.org/papers/v6/hyvarinen05a.html
    """
