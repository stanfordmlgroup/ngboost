import numpy as np

class MLE:
    def __init__(self, seed=123):
        pass

    def loss(self, forecast, Y):
        return forecast.nll(Y.squeeze()).mean()

    def grad(self, forecast, Y, natural=True):
        fisher = forecast.fisher_info()
        grad = forecast.D_nll(Y)
        if natural:
            grad = np.linalg.solve(fisher, grad)
        return grad


class CRPS:
    def __init__(self, K=32):
        self.K = K

    def loss(self, forecast, Y):
        return forecast.crps(Y.squeeze()).mean()

    def grad(self, forecast, Y, natural=True):
        metric = forecast.crps_metric()
        grad = forecast.D_crps(Y)
        if natural:
            grad = np.linalg.solve(metric, grad)
        return grad

