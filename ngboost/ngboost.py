import numpy as np
import numpy.random as np_rnd
from sklearn.base import BaseEstimator
from sklearn.utils import column_or_1d

from ngboost.scores import MLE
from ngboost.learners import default_tree_learner
from ngboost.distns.normal import Normal


class NGBoost(BaseEstimator):

    def __init__(self, Dist=Normal, Score=MLE(),
                 Base=default_tree_learner, natural_gradient=True,
                 n_estimators=500, learning_rate=0.01, subsample=1.0,
                 verbose=True, verbose_eval=100, tol=1e-4):
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
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

    def fit_base(self, X, grads, sample_weight):
        models = [self.Base().fit(X, g, sample_weight=sample_weight) for g in grads.T]
        fitted = np.array([m.predict(X) for m in models]).T
        self.base_models.append(models)
        return fitted

    def line_search(self, resids, start, Y, scale_init=1):
        S = self.Score
        D_init = self.Dist(start.T)
        loss_init = S.loss(D_init, Y)
        scale = scale_init
        while True:
            scaled_resids = resids * scale
            D = self.Dist((start - scaled_resids).T)
            loss = S.loss(D, Y)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if not np.isnan(loss) and (loss < loss_init or norm < self.tol) and\
               np.linalg.norm(scaled_resids, axis=1).mean() < 5.0:
                break
            scale = scale * 0.5
        self.scalings.append(scale)
        return scale

    def _random_sample_mask(self, n_total_samples, n_in_batch):
        rand = np_rnd.rand(n_total_samples)
        sample_mask = np.zeros(n_total_samples, dtype=np.bool)
        n_batch = 0

        for i in range(n_total_samples):
            if rand[i] * (n_total_samples - i) < (n_in_batch - n_batch):
                sample_mask[i] = 1
                n_batch += 1

        return sample_mask

    def fit(self, X, Y, X_val = None, Y_val = None, train_loss_monitor = None, val_loss_monitor = None,
                        sample_weight=None):

        n_samples = X.shape[0]

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)

        do_subsample = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=np.bool)
        n_inbatch = max(1, int(self.subsample * n_samples))

        loss_list = []
        val_loss_list = []

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
            sample_weight_batch = sample_weight

            if do_subsample:
                sample_mask = self._random_sample_mask(n_samples, n_inbatch)
                sample_weight_batch = sample_weight * sample_mask.astype(np.float64)

            X_batch, Y_batch, P_batch = X[sample_mask], Y[sample_mask], params[sample_mask, :]

            D = self.Dist(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch)]
            loss = loss_list[-1]
            grads = S.grad(D, Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads, sample_weight=sample_weight_batch[sample_mask])
            scale = self.line_search(proj_grad, P_batch, Y_batch)
            params -= self.learning_rate * scale * np.array([m.predict(X) for m in self.base_models[-1]]).T

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= self.learning_rate * scale * np.array([m.predict(X_val) for m in self.base_models[-1]]).T
                val_loss = val_loss_monitor(self.Dist(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if len(val_loss_list) > 10 and np.mean(np.array(val_loss_list[-5:])) > \
                   np.mean(np.array(val_loss_list[-10:-5])):
                    if self.verbose:
                        print(f"== Quitting at iteration / VAL {itr} (val_loss={val_loss:.4f})")
                    break

            if self.verbose and int(self.verbose_eval) > 0 and itr % int(self.verbose_eval) == 0:
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                      f"norm={grad_norm:.4f}")

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        return self

    def fit_init_params_to_marginal(self, Y, iters=1000):
        self.init_params = self.Dist.fit(Y)
        return

    def pred_dist(self, X, max_iter=None):
        params = np.asarray(self.pred_param(X, max_iter))
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
