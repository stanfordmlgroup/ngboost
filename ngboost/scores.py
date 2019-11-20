import numpy as np


class MLE:

    @staticmethod
    def loss(forecast, Y):
        return forecast.nll(Y.squeeze()).mean()

    @staticmethod
    def grad(forecast, Y, natural=True):
        fisher = forecast.fisher_info()
        grad = forecast.D_nll(Y)
        if natural:
            grad = np.linalg.solve(fisher, grad)
        return grad


class CRPS:

    @staticmethod
    def loss(forecast, Y):
        return forecast.crps(Y.squeeze()).mean()

    @staticmethod
    def grad(forecast, Y, natural=True):
        metric = forecast.crps_metric()
        grad = forecast.D_crps(Y)
        if natural:
            grad = np.linalg.solve(metric, grad)
        return grad
