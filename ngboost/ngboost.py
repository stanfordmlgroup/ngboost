import numpy as np
import scipy
import torch

from sklearn.tree import DecisionTreeRegressor
from torch.distributions.constraint_registry import transform_to
from torch.distributions.log_normal import LogNormal
from torch.optim import LBFGS

from experiments.evaluation import calculate_concordance_naive
from ngboost.scores import MLE_surv, CRPS_surv


class NGBoost(object):

    def __init__(self, Dist=LogNormal, Score=MLE_surv, Base=DecisionTreeRegressor,
                 n_estimators=1000, learning_rate=0.1, minibatch_frac=1.0,
                 natural_gradient=True, second_order=True,
                 quadrant_search=False, nu_penalty=0.0001, verbose=True):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.natural_gradient = natural_gradient
        self.second_order = second_order
        self.do_quadrant_search = quadrant_search
        self.nu_penalty = nu_penalty
        self.verbose = verbose
        self.Dist = Dist
        self.D = lambda args: Dist(*[transform_to(constraint)(arg) for (param, constraint), arg in zip(Dist.arg_constraints.items(), args)])
        self.Score = Score()
        self.Base = Base
        self.init_params = []
        self.base_models = []
        self.scalings = []

    def pred_param(self, X):
        m, n = X.shape
        params = [p * np.ones(m) for p in self.init_params]

        for models, scalings in zip(self.base_models, self.scalings):
            base_params = [model.predict(X) for model in models]
            params = [p - self.learning_rate * b for p, b in zip(params, self.mul(base_params, scalings))]

        return [torch.tensor(p, requires_grad=True, dtype=torch.float32) for p in params]

    def sample(self, X, Y):
        sample_size = int(self.minibatch_frac * len(Y))
        idxs = np.random.choice(np.arange(len(Y)), sample_size, replace=False)
        return idxs, X[idxs,:], torch.Tensor(Y[idxs])

    def fit_base(self, X, grads):
        models = [self.Base().fit(X, g.detach().numpy()) for g in grads]
        self.base_models.append(models)
        fitted = [torch.Tensor(m.predict(X)) for m in models]
        return fitted

    def mul(self, array, num):
        return [a * n for (a, n) in zip(array, num)]

    def sub(self, A, B):
        return [a - b for a, b in zip(A, B)]

    def norm(self, grads):
        return np.linalg.norm([float(torch.norm(g)) for g in grads])

    def line_search(self, fn, start, resids):
        loss_init = float(fn(start))
        scale = [10. for _ in resids]
        half = [0.5 for _ in resids]
        while True:
            new_loss = float(fn(self.sub(start, self.mul(resids, scale))))
            if new_loss < loss_init or self.norm(self.mul(resids, scale)) < 1e-5:
                break
            scale = self.mul(scale, half)
        self.scalings.append(scale)
        return scale

    def quadrant_search(self, fn, start, resids):
        def lossfn(lognu):
            nu = [float(n) for n in np.exp(lognu)]
            return float(fn(self.sub(start, self.mul(resids, nu)))) + self.nu_penalty * np.linalg.norm(nu)**2
        lognu0 = np.array([0. for _ in resids])
        res = scipy.optimize.minimize(lossfn, lognu0, method='Nelder-Mead', tol=1e-6)
        scale = [float(np.exp(f)) for f in res.x]
        self.scalings.append(scale)
        return scale

    def fit(self, X, Y):

        S = lambda p: self.Score(self.D(p), torch.tensor(Y, dtype=torch.float32)).mean()
        self.fit_init_params_to_marginal(S)
        for itr in range(self.n_estimators):
            idxs, X_batch, Y_batch = self.sample(X, Y)

            S = lambda p: self.Score(self.D(p), Y_batch).mean()
            params = self.pred_param(X_batch)
            score = S(params)

            if self.verbose:
                print('[iter %d] loss=%f' % (itr, float(score)))
            if float(score) == float('-inf'):
                break
            if str(float(score)) == 'nan':
                print(params)
                print(S(params))
                break

            score.backward(retain_graph=True, create_graph=True)
            grads = self.Score.grad(params, self.D, natural_gradient=self.natural_gradient, second_order=self.second_order)
            resids = self.fit_base(X_batch, grads)

            if self.do_quadrant_search:
                scale = self.quadrant_search(S, params, resids)
            else:
                scale = self.line_search(S, params, resids)

            if self.norm(self.mul(resids, scale)) < 1e-5:
                break

    def fit_init_params_to_marginal(self, S):

        init_params = [torch.tensor(0., requires_grad=True) for _ in  self.Dist.arg_constraints]
        opt = LBFGS(init_params, lr=1.)

        if self.verbose:
            print("Fitting marginal distribution, until convergence...")

        prev_loss = 0.

        for i in range(10**10):

            opt.zero_grad()
            loss = S(init_params)
            loss.backward(retain_graph=True)
            opt.step(lambda: loss)
            curr_loss = loss.data.numpy()
            if np.abs(prev_loss - curr_loss) < 1e-5:
                break
            prev_loss = curr_loss

        self.init_params = [p.detach().numpy() for p in init_params]

        if self.verbose:
            print("Initial params: ", self.init_params)


    def pred_dist(self, X):
        params = self.pred_param(X)
        dist = self.D(params)
        return dist

    def pred_mean(self, X):
        dist = self.pred_dist(X)
        return dist.mean.data.numpy()

    def pred_median(self, X):
        dist = self.pred_dist(X)
        return dist.icdf(torch.tensor(0.5)).data.numpy()


class SurvNGBoost(NGBoost):

    def sample(self, X, Y, C):
        sample_size = int(self.minibatch_frac * len(Y))
        idxs = np.random.choice(np.arange(len(Y)), sample_size, replace=False)
        return idxs, X[idxs, :], torch.Tensor(Y[idxs]), torch.Tensor(C[idxs])

    def fit(self, X, Y, C):
        S = lambda p: self.Score(self.D(p), torch.tensor(Y, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)).mean()
        self.fit_init_params_to_marginal(S)
        for itr in range(self.n_estimators):
            idxs, X_batch, Y_batch, C_batch = self.sample(X, Y, C)

            S = lambda p: self.Score(self.D(p), Y_batch, C_batch).mean()
            params = self.pred_param(X_batch)
            score = S(params)

            print('[iter %d] loss=%f' % (itr, float(score)))
            if float(score) == float('-inf'):
                break
            if str(float(score)) == 'nan':
                print(params)
                print(S(params))
                break

            score.backward(retain_graph=True, create_graph=True)
            grads = self.Score.grad(params, self.D, natural_gradient=self.natural_gradient, second_order=self.second_order)
            resids = self.fit_base(X_batch, grads)

            if self.do_quadrant_search:
                scale = self.quadrant_search(S, params, resids)
            else:
                scale = self.line_search(S, params, resids)

            if self.norm(self.mul(resids, scale)) < 1e-5:
                break


def main():
    m, n = 100, 50
    X = np.random.rand(m, n).astype(np.float32)
    Y = np.random.rand(m).astype(np.float32) * 2 + 1
    C = (np.random.rand(m) > 0.5).astype(np.float32)

    sb = SurvNGBoost(Base = lambda : DecisionTreeRegressor(criterion='mse'),
                     Dist = LogNormal,
                     Score = CRPS_surv,
                     n_estimators = 12,
                     learning_rate = 0.1,
                     natural_gradient = True,
                     second_order = True,
                     quadrant_search = False,
                     nu_penalty=1e-5)
    sb.fit(X, Y, C)
    preds = sb.pred_mean(X)

    print("Train/DecTree:", calculate_concordance_naive(preds, Y, C))
    print('Pred_mean: %f, True_mean: %f' % (np.mean(preds), np.mean(Y)))


if __name__ == '__main__':
    main()
