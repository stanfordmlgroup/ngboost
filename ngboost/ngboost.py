import numpy as np
import scipy as sp
from ngboost.distns import Normal
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.distns.normal import Normal
from sklearn.base import clone

class NGBoost(object):

    def __init__(self, Dist=Normal, Score=MLE,
                 Base=default_tree_learner, natural_gradient=True,
                 n_estimators=500, learning_rate=0.01, minibatch_frac=1.0,
                 verbose=True, verbose_eval=100, tol=1e-4):
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.init_params = None
        self.base_models = []
        self.scalings = []
        self.tol = tol

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
        if self.minibatch_frac == 1.0:
            return np.arange(len(Y)), X, Y, params
        sample_size = int(self.minibatch_frac * len(Y))
        idxs = np.random.choice(np.arange(len(Y)), sample_size, replace=False)
        return idxs, X[idxs,:], Y[idxs], params[idxs, :]

    def fit_base(self, X, grads):
        models = [clone(self.Base).fit(X, g) for g in grads.T]
        fitted = np.array([m.predict(X) for m in models]).T
        self.base_models.append(models)
        return fitted

    def line_search(self, resids, start, Y, scale_init=1):
        S = self.Score
        D_init = self.Dist(start.T)
        loss_init = S.loss(D_init, Y)
        scale = scale_init

        # first scale up
        while True:
            scaled_resids = resids * scale
            D = self.Dist((start - scaled_resids).T)
            loss = S.loss(D, Y)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if not np.isfinite(loss) or loss > loss_init:
                break
            scale = scale * 2

        # then scale down
        while True:
            scaled_resids = resids * scale
            D = self.Dist((start - scaled_resids).T)
            loss = S.loss(D, Y)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if np.isfinite(loss) and (loss < loss_init or norm < self.tol) and\
               np.linalg.norm(scaled_resids, axis=1).mean() < 5.0:
                break
            scale = scale * 0.5
        self.scalings.append(scale)
        return scale

    def fit(self, X, Y, 
            X_val = None, Y_val = None, 
            train_loss_monitor = None, val_loss_monitor = None, 
            early_stopping_rounds = None):

        loss_list = []
        val_loss_list = []

        if early_stopping_rounds is not None:
            best_val_loss = np.inf
            best_val_loss_itr = 0

        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)
        if X_val is not None and Y_val is not None:
            val_params = self.pred_param(X_val)

        S = self.Score

        if not train_loss_monitor:
            train_loss_monitor = S.loss

        if not val_loss_monitor:
            val_loss_monitor = S.loss

        for itr in range(self.n_estimators):
            _, X_batch, Y_batch, P_batch = self.sample(X, Y, params)

            D = self.Dist(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch)]
            loss = loss_list[-1]
            grads = S.grad(D, Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads)
            scale = self.line_search(proj_grad, P_batch, Y_batch)

            params -= self.learning_rate * scale * np.array([m.predict(X) for m in self.base_models[-1]]).T

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= self.learning_rate * scale * np.array([m.predict(X_val) for m in self.base_models[-1]]).T
                val_loss = val_loss_monitor(self.Dist(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if early_stopping_rounds is not None:
                    if val_loss < best_val_loss:
                        best_val_loss, best_val_loss_itr = val_loss, itr
                    if best_val_loss < np.min(np.array(val_loss_list[-early_stopping_rounds:])):
                        if self.verbose:
                            print(f"== Early stopping achieved.")
                            print(f"== Best iteration / VAL {best_val_loss_itr} (val_loss={best_val_loss:.4f})")
                        break

            if self.verbose and int(self.verbose_eval) > 0 and itr % int(self.verbose_eval) == 0:
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                      f"norm={grad_norm:.4f}")

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break


        self.evals_result = {}
        metric = self.Score.__name__.upper()
        self.evals_result['train'] = {metric: loss_list}
        if X_val is not None and Y_val is not None:
            self.evals_result['val'] = {metric: val_loss_list}

        return self

    def fit_init_params_to_marginal(self, Y, iters=1000):
        self.init_params = self.Dist.fit(Y)
        return

    def pred_dist(self, X, max_iter=None):
        params = np.asarray(self.pred_param(X, max_iter))
        dist = self.Dist(params.T)
        return dist

    def predict(self, X): # move this and staged_predict to the api
        dist = self.pred_dist(X)
        return list(dist.loc.flatten())

    def score(self, X, Y):
        return self.Score.loss(self.pred_dist(X), Y)

    def staged_predict(self, X, max_iter=None):
        predictions = []
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
            dists = self.Dist(np.asarray(params).T)
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
            dists = self.Dist(np.asarray(params).T)
            predictions.append(dists)
        return predictions

