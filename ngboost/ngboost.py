import jax.numpy as np
import numpy as onp
import jax.scipy as sp
import scipy.optimize as optim
import pickle
import numpy.random as np_rnd

from jax import jit, grad, vmap, jacfwd, jacrev
from ngboost.distns import Normal

from ngboost.scores import MLE, MLE_SURV, CRPS_SURV, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.distns.normal import Normal


class NGBoost(object):

    def __init__(self, Dist=Normal, Score=MLE(),
                 Base=default_tree_learner, natural_gradient=False,
                 n_estimators=100, learning_rate=0.1, minibatch_frac=1.0,
                 verbose=True, tol=1e-4):
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.verbose = verbose
        self.init_params = None
        self.base_models = []
        self.scalings = []
        self.tol = tol
        self.loss_fn = lambda P, Y: self.Score(self.Dist(P.T), Y).sum()
        self.grad_fn = grad(self.loss_fn)
        #self.grad_fn = jit(vmap(grad(self.loss_fn)))
        self.hessian_fn = jit(vmap(jacrev(grad(self.loss_fn))))
        #self.loss_fn = jit(vmap(self.loss_fn))
        self.Score.setup_distn(self.Dist)
        if isinstance(self.Score, CRPS_SURV):
            self.marginal_score = MLE_SURV()
        elif isinstance(self.Score, CRPS):
            self.marginal_score = MLE()
        else:
            self.marginal_score = self.Score
        self.marginal_loss = lambda P, Y: self.marginal_score(self.Dist(P), Y)
        self.marginal_grad = jit(vmap(grad(self.marginal_loss)))
        self.marginal_loss = jit(vmap(self.marginal_loss))
        self.matmul_inv_fn = jit(vmap(lambda A, b: np.linalg.solve(A, b)))

    def pred_param(self, X, max_iter=None):
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
        return params

    def sample(self, X, Y, params):
        sample_size = int(self.minibatch_frac * len(Y))
        idxs = np_rnd.choice(np.arange(len(Y)), sample_size, replace=False)
        return idxs, X[idxs,:], Y[idxs], params[idxs, :]

    def fit_base(self, X, grads):
        models = [self.Base().fit(X, g) for g in grads.T]
        fitted = np.array([m.predict(X) for m in models]).T
        self.base_models.append(models)
        return fitted

    def line_search(self, resids, start, Y, scale_init=1):
        loss_init = self.loss_fn(start, Y).mean()
        scale = scale_init
        while True:
            scaled_resids = resids * scale
            loss = self.loss_fn(start - scaled_resids, Y).mean()
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if not np.isnan(loss) and (loss < loss_init or norm < self.tol) and\
               np.linalg.norm(scaled_resids, axis=1).mean() < 5.0:
                break
            scale = scale * 0.5
        self.scalings.append(scale)
        return scale

    def fit(self, X, Y, X_val = None, Y_val = None):

        loss_list = []
        val_loss_list = []
        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)
        if X_val is not None and Y_val is not None:
            val_params = self.pred_param(X_val)

        for itr in range(self.n_estimators):
            _, X_batch, Y_batch, P_batch = self.sample(X, Y, params)

            losses = self.loss_fn(P_batch, Y_batch)
            loss = losses.mean()
            if np.isinf(loss) or np.isnan(loss):
                for i, (p, l, y) in enumerate(zip(P_batch, losses, Y_batch)):
                    if np.isinf(l) or np.isnan(l):
                        print('[%d] Params=[%.4f,%.4f], loss=%.4f, y=%.4f' % (i, p[0], p[1], l, y))
                        print('Loss=%.4f' % self.Dist(p).crps_debug(y))
                breakpoint()

            grads = self.grad_fn(P_batch, Y_batch)

            if self.natural_gradient:
                grads = self.Score.naturalize(P_batch, grads)

            #scale = self.line_search(grads, P_batch, Y_batch, scale_init=1)
            #grads = grads * scale

            if np.any(np.isnan(grads)) or np.any(np.isinf(grads)):
                print(grads)
                grads = self.grad_fn(P_batch, Y_batch)
                print('recalculated')
                print(grads)
                print('params')
                print(grads)
                pass

            proj_grad = self.fit_base(X_batch, grads)
            scale = self.line_search(proj_grad, P_batch, Y_batch)

            loss_list += [loss]

            params -= self.learning_rate * scale * np.array([m.predict(X) for m in self.base_models[-1]]).T

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= self.learning_rate * scale * np.array([m.predict(X_val) for m in self.base_models[-1]]).T
                val_loss = self.loss_fn(val_params, Y_val).mean()
                val_loss_list += [val_loss]
                if np.mean(np.array(val_loss_list[-5:])) > \
                   np.mean(np.array(val_loss_list[-10:-5])):
                    if self.verbose:
                        print(f"== Quitting at iteration / VAL {itr} (val_loss={val_loss:.4f})")
                    break

            if self.verbose:
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                      f"norm={grad_norm:.4f}")

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        return loss_list, val_loss_list

    def fit_init_params_to_marginal(self, Y, iters=1000):
        try:
            E = Y['Event']
            T = Y['Time'].reshape((-1, 1))[E == 1]
        except:
            T = Y
        self.init_params = self.Dist.fit(T)
        return


    def pred_dist(self, X, max_iter=None):
        params = onp.asarray(self.pred_param(X, max_iter))
        dist = self.Dist(params.T)
        return dist

    def predict(self, X):
        dist = self.pred_dist(X)
        return list(dist.loc.flatten())

    def staged_predict(self, X, max_iter=None):
        predictions = []
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
            dists = self.Dist(onp.asarray(params).T)
            predictions.append(dists.loc.flatten())
        return predictions

    def staged_pred_dist(self, X, max_iter=None):
        predictions = []
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
            dists = self.Dist(onp.asarray(params).T)
            predictions.append(dists)
        return predictions
