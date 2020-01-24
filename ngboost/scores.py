import numpy as np


class MLE:

    @staticmethod
    def loss(forecast, Y, sample_weight=None):
        return np.average(forecast.nll(Y.squeeze()), weights=sample_weight)

    @staticmethod
    def grad(forecast, Y, natural=True):
        grad = forecast.D_nll(Y)
        if natural:
            fisher = forecast.fisher_info()
            grad = np.linalg.solve(fisher, grad)
        return grad


class CRPS:

    @staticmethod
    def loss(forecast, Y, sample_weight=None):
        return np.average(forecast.crps(Y.squeeze()), weights=sample_weight)

    @staticmethod
    def grad(forecast, Y, natural=True):
        grad = forecast.D_crps(Y)
        if natural:
            metric = forecast.crps_metric()
            grad = np.linalg.solve(metric, grad)
        return grad
