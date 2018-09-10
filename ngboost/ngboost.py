import numpy as np
import scipy as sp
import torch
import pickle

from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.constraint_registry import transform_to
from torch.distributions.transforms import AffineTransform
from torch.optim import LBFGS

from ngboost.scores import MLE, MLE_surv
from ngboost.learners import default_tree_learner


class NGBoost(object):

    def __init__(self, Dist=Normal, Score=MLE, Base=default_tree_learner,
                 n_estimators=1000, learning_rate=0.1, minibatch_frac=1.0,
                 natural_gradient=True, second_order=True,
                 quadrant_search=False, nu_penalty=0.0001,
                 normalize_inputs=True, normalize_outputs=True, verbose=True):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.natural_gradient = natural_gradient
        self.second_order = second_order
        self.do_quadrant_search = quadrant_search
        self.nu_penalty = nu_penalty
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.verbose = verbose
        self.Dist = Dist
        self.D = lambda args: \
            Dist(*[transform_to(constraint)(arg) for (param, constraint), arg
                   in zip(Dist.arg_constraints.items(), args)])
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
            params = [p - self.learning_rate * b for p, b
                      in zip(params, self.mul(base_params, scalings))]
        return [torch.tensor(p, requires_grad=True, dtype=torch.float32) for p
                in params]

    def sample(self, X, Y):
        sample_size = int(self.minibatch_frac * len(Y))
        idxs = np.random.choice(np.arange(len(Y)), sample_size, replace=False)
        return idxs, X[idxs,:], Y[idxs]

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
            loss = float(fn(self.sub(start, self.mul(resids, scale))))
            if loss < loss_init or self.norm(self.mul(resids, scale)) < 1e-5:
                break
            scale = self.mul(scale, half)
        self.scalings.append(scale)
        return scale

    def quadrant_search(self, fn, start, resids):
        def lossfn(lognu):
            nu = [float(n) for n in np.exp(lognu)]
            return float(fn(self.sub(start, self.mul(resids, nu)))) + \
                   self.nu_penalty * np.linalg.norm(nu) ** 2
        lognu0 = np.array([0. for _ in resids])
        res = sp.optimize.minimize(lossfn, lognu0, method='Nelder-Mead',
                                   tol=1e-6)
        scale = [float(np.exp(f)) for f in res.x]
        self.scalings.append(scale)
        return scale

    def fit(self, X, Y):

        if self.normalize_inputs:
            self.X_scaler = StandardScaler(copy=True)
            X = self.X_scaler.fit_transform(X)
        if self.normalize_outputs:
            self.Y_scaler = StandardScaler(copy=True)
            Y = self.Y_scaler.fit_transform(Y[:,np.newaxis])[:,0]

        loss_list = []
        Y = torch.tensor(Y, dtype=torch.float32)
        S_full = lambda p: self.Score(self.D(p), Y).mean()
        self.fit_init_params_to_marginal(S_full)

        for itr in range(self.n_estimators):

            idxs, X_batch, Y_batch = self.sample(X, Y)
            S_batch = lambda p: self.Score(self.D(p), Y_batch).mean()

            params = self.pred_param(X_batch)
            score = S_batch(params)

            if self.verbose:
                print("[iter %d] loss=%.4f" % (itr, float(score)))
            if float(score) == float('-inf'):
                raise ValueError("Score of -inf occurred.")
            if str(float(score)) == 'nan':
                raise ValueError("Score of nan occurred.")

            grads = torch.autograd.grad(score, params, create_graph=True)
            grads = self.Score.grad(self.D, params, grads,
                                    natural_gradient=self.natural_gradient,
                                    second_order=self.second_order)
            resids = self.fit_base(X_batch, grads)

            if self.do_quadrant_search:
                scale = self.quadrant_search(S_batch, params, resids)
            else:
                scale = self.line_search(S_batch, params, resids)

            loss_list.append(score.detach().numpy())
            if self.norm(self.mul(resids, scale)) < 1e-5:
                break

        return loss_list

    def fit_init_params_to_marginal(self, S, lbfgs_lr=0.1):
        init_params = [torch.tensor(0., requires_grad=True) for _
                       in self.Dist.arg_constraints]
        opt = LBFGS(init_params, lr=lbfgs_lr, max_iter=20)
        prev_loss = float("inf")
        if self.verbose:
            print("Fitting marginal distribution, until convergence...")
        while True:
            opt.zero_grad()
            loss = S(init_params)
            curr_loss = loss.data.numpy()
            if np.isnan(curr_loss):
                lbfgs_lr /= 10
                opt = LBFGS(init_params, lr=lbfgs_lr, max_iter=20)
                init_params = [torch.tensor(0., requires_grad=True) for _
                               in self.Dist.arg_constraints]
                continue
            loss.backward(retain_graph=True)
            opt.step(lambda: loss)
            if np.abs(prev_loss - curr_loss) < 1e-5:
                break
            prev_loss = curr_loss
        self.init_params = [p.detach().numpy() for p in init_params]

    def pred_dist(self, X):
        if self.normalize_inputs:
            X = self.X_scaler.transform(X)
        params = self.pred_param(X)
        dist = self.D(params)
        breakpoint()
        if self.normalize_outputs:
            transform = AffineTransform(self.Y_scaler.mean_[0],
                                        self.Y_scaler.scale_[0])
            dist = TransformedDistribution(dist, [transform])
        return dist

    # def pred_mean(self, X):
    #     dist = self.pred_dist(X)
    #     return dist.mean.data.numpy()

    # def pred_median(self, X):
    #     dist = self.pred_dist(X)
    #     return dist.icdf(torch.tensor(0.5)).data.numpy()

    def write_to_disk(self, filename):
        if not self.base_models:
            raise ValueError("NGBoost model has not yet been fit!")
        file = open(filename, "wb")
        pickle.dump({
            "base_models": self.base_models,
            "scalings": self.scalings,
            "learning_rate": self.learning_rate,
            "init_params": self.init_params,
        }, file)
        file.close()

    def load_from_disk(self, filename):
        file = open(filename, "rb")
        properties = pickle.load(file)
        self.base_models = properties["base_models"]
        self.scalings = properties["scalings"]
        self.learning_rate = properties["learning_rate"]
        self.init_params = properties["init_params"]


class SurvNGBoost(NGBoost):

    def sample(self, X, Y, C):
        sample_size = int(self.minibatch_frac * len(Y))
        idxs = np.random.choice(np.arange(len(Y)), sample_size, replace=False)
        return idxs, X[idxs, :], Y[idxs], C[idxs]

    def fit(self, X, Y, C):

        loss_list = []
        Y = torch.tensor(Y, dtype=torch.float32)
        C = torch.tensor(C, dtype=torch.float32)
        S_full = lambda p: self.Score(self.D(p), Y, C).mean()
        self.fit_init_params_to_marginal(S_full)

        for itr in range(self.n_estimators):

            idxs, X_batch, Y_batch, C_batch = self.sample(X, Y, C)

            S_batch = lambda p: self.Score(self.D(p), Y_batch, C_batch).mean()
            params = self.pred_param(X_batch)
            score = S_batch(params)

            if self.verbose:
                print("[iter %d] loss=%.4f" % (itr, float(score)))
            if float(score) == float('-inf'):
                raise ValueError("Score of -inf occurred.")
            if str(float(score)) == 'nan':
                raise ValueError("Score of nan occurred.")

            grads = torch.autograd.grad(score, params, create_graph=True)
            grads = self.Score.grad(self.D, params, grads,
                                    natural_gradient=self.natural_gradient,
                                    second_order=self.second_order)
            resids = self.fit_base(X_batch, grads)

            if self.do_quadrant_search:
                scale = self.quadrant_search(S_batch, params, resids)
            else:
                scale = self.line_search(S_batch, params, resids)

            loss_list.append(score.detach().numpy())
            if self.norm(self.mul(resids, scale)) < 1e-5:
                break

        return loss_list

