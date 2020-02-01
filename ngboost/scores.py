import numpy as np

def uncensored(Y):
    return {"Event":np.ones_like(Y), "Time":Y}

class Score():
    def total_score(self, Y, sample_weight=None):
        return np.average(self.score(Y), weights=sample_weight)

    def grad(self, Y, natural=True):
        grad = self.d_score(Y)
        if natural:
            metric = self.metric()
            grad = np.linalg.solve(metric, grad)
        return grad

    @classmethod
    def uncensor(DistScore):
        class UncensoredScore(DistScore, DistScore.__base__):
            def score(self, Y):
                return super().score(uncensored(Y))
            def d_score(self, Y):
                return super().d_score(uncensored(Y))
        return UncensoredScore

class LogScore(Score):
    def metric(self, n_mc_samples=100):
        grads = np.stack([self.d_score(Y) for Y in self.sample(n_mc_samples)])
        return np.mean(np.einsum('sik,sij->sijk', grads, grads), axis=0)
    # autofit method from d_score?
MLE = LogScore

class CRPScore(Score):
    pass
CRPS = CRPScore