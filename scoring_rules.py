import torch
import numpy as np

class MLE_surv(object):
    def __init__(self, K=32):
        self.K = K
        pass

    def loss(self, Forecast, Y, C):
        return - ((1 - C) * Forecast.log_prob(Y) + (1 - Forecast.cdf(Y) + 1e-5).log() * C)

    def __call__(self, Forecast, Y, C):
        return self.loss(Forecast, Y, C)

    def inverse(self, matrix):
        m = int(matrix.shape[0])
        return torch.inverse(matrix + 1e-2 * torch.eye(m))

    def grad(self, params, D):
        grad = [p.grad.clone() for p in params]
        grads = torch.cat([g.reshape(-1, 1) for g in grad], dim=1)
        Forecast = D(params)
        metric = self.metric(params, Forecast)
        nat_grads = torch.cat([torch.mv(self.inverse(m), g).unsqueeze(0) for g, m in zip(grads, metric)], dim=0)
        nat_grad = [ng.reshape(-1) for ng in torch.split(nat_grads, 1, dim=1)]
        return nat_grad

    def metric(self, params, Forecast):
        cov = 0
        m, n = int(params[0].shape[0]), len(params)
        for _ in range(self.K):
            for p in params:
                p.grad.data.zero_()
            X = Forecast.sample()
            score = Forecast.log_prob(X).mean()
            score.backward(retain_graph=True)
            grads = torch.cat([p.grad.clone().reshape(-1, 1) for p in params], dim=1)
            cov += grads.reshape(m, 1, n) * grads.reshape(m, n, 1)
        return cov / self.K

class CRPS_surv(object):
    def __init__(self, K=32):
        self.K = K
        pass

    def loss(self, Forecast, Y, C):
        def I(F, U):
            I_sum = 0.
            for th in np.linspace(0, 1., self.K):
                if th == 0:
                    prev_F = 0.
                    prev_x = 0.
                    continue
                this_x = U * th
                this_F = F(this_x)
                Fdiff = 0.5 * (this_F + prev_F)
                xdiff = this_x - prev_x
                I_sum += (Fdiff * xdiff)
                prev_F = this_F
                prev_x = this_x
            return I_sum

        left = I(lambda y: Forecast.cdf(y).pow(2), Y)
        right = I(lambda y: ((1 - Forecast.cdf(1/y)) / y).pow(2), 1/Y)
        return (left + (1 - C) * right)

    def __call__(self, Forecast, Y, C):
        return self.loss(Forecast, Y, C)

    def grad(self, params, D):
        grads = [torch.Tensor(p.grad) for p in params]
        Forecast = D(params)
        metrics = self.metric(params, Forecast)
        
        return grads

    def metric(self, params, Forecast):
        return 0
