import numpy as np

class Score():
    def total_score(self, Y, sample_weight=None):
        return np.average(self.score(Y.squeeze()), weights=sample_weight)

    def grad(Y, natural=True):
        grad = self.d_score(Y)
        if natural:
            metric = self.metric()
            grad = np.linalg.solve(metric, grad)
        return grad

class LogScore(ScoringRule):
    def metric(self, n_mc_samples=100):
        grads = np.stack([self.d_score(Y) for Y in self.sample(n_mc_samples)])
        return np.mean(np.einsum('sik,sij->sijk', grads, grads), axis=0)
    # autofit method from d_score?
MLE = LogScore

class CRPScore(ScoringRule):
    pass
CRPS = CRPScore