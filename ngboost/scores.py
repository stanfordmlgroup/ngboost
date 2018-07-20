import torch
import numpy as np


class Score(object):

    def grad(self, params, D, natural_gradient=True, second_order=True):
        grad = [p.grad.clone() for p in params]
        if not natural_gradient and not second_order:
            return grad

        grads = torch.cat([g.reshape(-1, 1) for g in grad], dim=1)

        if second_order:
            M, P = len(grads), len(params)
            hessians = torch.zeros((M, P, P))
            for i in range(P):
                second_derivs = torch.autograd.grad(grads[:,i].split(1), params, retain_graph=True)
                for j, g in enumerate(second_derivs):
                    hessians[:,i,j] = g
            for k in range(M):
                L, U = torch.eig(hessians[k,:,:], eigenvectors=True)
                L = torch.diag(torch.abs(L[:,0]))
                hessians[k,:,:] = U @ L @ torch.transpose(U, 1, 0)
            grads = torch.cat([torch.mv(self.inverse(m), g).unsqueeze(0) for g, m in zip(grads, hessians)], dim=0)

        if natural_gradient:
            Forecast = D(params)
            metric = self.metric(params, Forecast)
            grads = torch.cat([torch.mv(self.inverse(m), g).unsqueeze(0) for g, m in zip(grads, metric)], dim=0)

        grad = [ng.reshape(-1) for ng in torch.split(grads, 1, dim=1)]
        return grad

    def inverse(self, matrix):
        m = int(matrix.shape[0])
        return torch.inverse(matrix + 1e-2 * torch.eye(m))


class MLE(Score):

    def __init__(self, K=32):
        self.K = K

    def __call__(self, Forecast, Y):
        return self.loss(Forecast, Y)

    def loss(self, Forecast, Y):
        return -Forecast.log_prob(Y)

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


class MLE_surv(MLE):

    def loss(self, Forecast, Y, C):
        return - ((1 - C) * Forecast.log_prob(Y) + (1 - Forecast.cdf(Y) + 1e-5).log() * C)

    def __call__(self, Forecast, Y, C):
        return self.loss(Forecast, Y, C)


class CRPS(Score):

    def __init__(self, K=32):
        self.K = K

    def __call__(self, Forecast, Y):
        return self.loss(Forecast, Y)

    def I(self, F, U):
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

    def loss(self, Forecast, Y):

        left = self.I(lambda y: Forecast.cdf(y).pow(2), Y)
        right = self.I(lambda y: ((1 - Forecast.cdf(1/y)) / y).pow(2), 1/Y)
        return left + right

    def metric(self, params, Forecast):
        m, n = int(params[0].shape[0]), len(params)
        I_sum = 0.
        for th in np.linspace(0, 1., self.K):
            if th == 0:
                prev_F = 0.
                prev_x = 0.
                continue
            this_x = 1. / th
            xdiff = this_x - prev_x

            loss = Forecast.cdf(torch.Tensor([this_x])).mean()
            loss.backward(retain_graph=True)

            grads = torch.cat([p.grad.clone().reshape(-1, 1) for p in params], dim=1)
            this_F = grads.reshape(m, 1, n) * grads.reshape(m, n, 1) / (th ** 2)
            Fdiff = 0.5 * (this_F + prev_F)
            I_sum += Fdiff * 1./self.K

            prev_F = this_F
            prev_x = this_x
        return 2 * I_sum


class CRPS_surv(CRPS):

    def loss(self, Forecast, Y, C):
        left = self.I(lambda y: Forecast.cdf(y).pow(2), Y)
        right = self.I(lambda y: ((1 - Forecast.cdf(1/y)) / y).pow(2), 1/Y)
        return (left + (1 - C) * right)

    def __call__(self, Forecast, Y, C):
        return self.loss(Forecast, Y, C)

