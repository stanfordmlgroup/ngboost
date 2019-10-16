import numpy as np

class MLE:
    def __init__(self, seed=123):
        pass

    def loss(self, forecast, Y):
        return forecast.nll(Y.squeeze())

    def natural_grad(self, forecast, Y):
        fisher = forecast.fisher_info()
        grad = forecast.D_nll(Y)
        nat_grad = np.linalg.solve(fisher, grad)
        return nat_grad


class CRPS:
    def __init__(self, K=32):
        self.K = K

    def loss(self, forecast, Y):
        return forecast.crps(Y.squeeze())

    def natural_grad(self, forecast, Y):
        metric = forecast.crps_metric()
        grad = forecast.D_crps(Y)
        nat_grad = np.linalg.solve(metric, grad)
        return nat_grad

