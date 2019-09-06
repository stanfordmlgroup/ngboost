import jax.numpy as np
import numpy as onp
import jax.random as random
import jax.scipy as sp
from jax import grad, vmap, jacfwd, jit
from jax.lax import stop_gradient

from ngboost.distns import Normal, LogNormal


class Score(object):

    def __init__(self, seed=123):
        self.key = random.PRNGKey(seed=seed)

    def __call__(self, Forecast, Y):
        raise NotImplementedError

    def setup_distn(self, distn):
        self.distn = distn


class MLE(Score):

    def __init__(self, seed=123, K=128):
        super().__init__(seed=seed)
        self.metric_fn = jit(vmap(lambda p: self.distn(p).fisher_info()))
        self.sample_grad_fn = jit(vmap(grad(self._loglik_fn)))
        self.outer_product_fn = jit(vmap(lambda v: np.outer(v, v)))
        self.K = K

    def __call__(self, forecast, Y):
        return -forecast.logpdf(Y.squeeze())

    def metric(self, params, Y):
        if self.distn.has_fisher_info:
            return self.metric_fn(params)
        batch_size = len(params)
        var = 0
        for _ in range(self.K):
            self.key, subkey = random.split(self.key)
            grad = self.sample_grad_fn(params, random.split(subkey, batch_size))
            var += self.outer_product_fn(grad)
        return var / self.K

    def _loglik_fn(self, params, key):
        sample = stop_gradient(self.distn(params).sample(key=key))
        return self.distn(params).logpdf(sample)


class MLE_SURV(MLE):

    def __init__(self, seed=123):
        super().__init__(seed=seed)

    def __call__(self, forecast, Y, eps=1e-5):
        E = Y['Event']
        T = Y['Time']
        cens = (1-E) * np.log(1 - forecast.cdf(T) + eps)
        uncens = E * forecast.logpdf(T)
        return -(cens + uncens)

class CRPS(Score):

    def __init__(self, K=32):
        super().__init__()
        self.metric_fn = jit(vmap(lambda p: self.distn(p).crps_metric()))
        self.K = K
        self.I_pos = self._I_pos
        self.I_neg = self._I_neg
        #self.I_pos = jit(self._I_pos)
        #self.I_neg = jit(self._I_neg)

    def _I_pos(self, params, U):
        axis = np.outer(np.linspace(0, 1, self.K)[1:], U)
        evals = self.distn(params).cdf(axis) ** 2
        return 0.5 * np.sum(evals[:self.K - 2] + evals[1:]) * U / self.K

    def _I_neg(self, params, U):
        axis = np.outer(np.linspace(0, 1, self.K)[1:], U)
        evals = (1 - self.distn(params).cdf(1 / axis) ** 2) / axis ** 2
        return 0.5 * np.sum(evals[:self.K - 2] + evals[1:]) * U / self.K

    def I_normal(self, Forecast, Y):
        S = Forecast.scale
        Y_std = (Y - Forecast.loc) / Forecast.scale
        norm2 = Normal([Forecast.loc, Forecast.scale / np.sqrt(2.)])
        ncdf = Forecast.cdf(Y)
        npdf = np.exp(Forecast.logpdf(Y))
        n2cdf = norm2.cdf(Y)
        return S * (Y_std * np.power(ncdf, 2) + 2 * ncdf * npdf * S -
               n2cdf / np.sqrt(np.pi))

    def __call__(self, forecast, Y):
        return forecast.crps(Y.squeeze())

    def metric(self, params, Y):
        if self.distn.has_crps_metric:
            return self.metric_fn(params)
        raise NotImplementedError


class CRPS_SURV(CRPS):

    def __call__(self, forecast, Y):
        E = Y['Event']
        T = Y['Time']
        if isinstance(forecast, LogNormal):
            left = self.I_normal(Normal((forecast.loc, forecast.scale)), onp.log(T))
            right = self.I_normal(Normal((-forecast.loc, forecast.scale)), -onp.log(T))
        else:
            left = self.I_pos(forecast.params, T)
            right = self.I_neg(forecast.params, 1/T)
        return (left + E * right)

