import numpy as np
import scipy as sp
from ngboost.distns import Normal
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.distns.normal import Normal
from sklearn.base import clone

# import pdb

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
        self.best_val_loss_itr = None

    def fit_init_params_to_marginal(self, Y, sample_weight=None, iters=1000):
        self.init_params = self.Dist.fit(Y) # would be best to put sample weights here too
        return

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

    def fit_base(self, X, grads, sample_weight=None):
        models = [clone(self.Base).fit(X, g, sample_weight=sample_weight) for g in grads.T]
        fitted = np.array([m.predict(X) for m in models]).T
        self.base_models.append(models)
        return fitted

    def line_search(self, resids, start, Y, sample_weight=None, scale_init=1): 
        S = self.Score
        D_init = self.Dist(start.T)
        loss_init = S.loss(D_init, Y, sample_weight)
        scale = scale_init

        # first scale up
        while True:
            scaled_resids = resids * scale
            D = self.Dist((start - scaled_resids).T)
            loss = S.loss(D, Y, sample_weight)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if not np.isfinite(loss) or loss > loss_init or scale > 256:
                break
            scale = scale * 2

        # then scale down
        while True:
            scaled_resids = resids * scale
            D = self.Dist((start - scaled_resids).T)
            loss = S.loss(D, Y, sample_weight)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if np.isfinite(loss) and (loss < loss_init or norm < self.tol) and\
               np.linalg.norm(scaled_resids, axis=1).mean() < 5.0:
                break
            scale = scale * 0.5
        self.scalings.append(scale)
        return scale

    def fit(self, X, Y, 
            X_val = None, Y_val = None, 
            sample_weight = None, val_sample_weight = None,
            train_loss_monitor = None, val_loss_monitor = None, 
            early_stopping_rounds = None):

        loss_list = []
        val_loss_list = []

        if early_stopping_rounds is not None:
            best_val_loss = np.inf

        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)
        if X_val is not None and Y_val is not None:
            val_params = self.pred_param(X_val)

        S = self.Score

        if not train_loss_monitor:
            train_loss_monitor = lambda D,Y: S.loss(D, Y, sample_weight=sample_weight)

        if not val_loss_monitor:
            val_loss_monitor = lambda D,Y: S.loss(D, Y, sample_weight=val_sample_weight)

        for itr in range(self.n_estimators):
            _, X_batch, Y_batch, P_batch = self.sample(X, Y, params)

            D = self.Dist(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch)]
            loss = loss_list[-1]
            grads = S.grad(D, Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads, sample_weight)
            scale = self.line_search(proj_grad, P_batch, Y_batch, sample_weight)

            # pdb.set_trace()
            params -= self.learning_rate * scale * np.array([m.predict(X) for m in self.base_models[-1]]).T

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= self.learning_rate * scale * np.array([m.predict(X_val) for m in self.base_models[-1]]).T
                val_loss = val_loss_monitor(self.Dist(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if early_stopping_rounds is not None:
                    if val_loss < best_val_loss:
                        best_val_loss, self.best_val_loss_itr = val_loss, itr
                    if best_val_loss < np.min(np.array(val_loss_list[-early_stopping_rounds:])):
                        if self.verbose:
                            print(f"== Early stopping achieved.")
                            print(f"== Best iteration / VAL {self.best_val_loss_itr} (val_loss={best_val_loss:.4f})")
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

    def score(self, X, Y):
        return self.Score.loss(self.pred_dist(X), Y)

    def pred_dist(self, X, max_iter=None):
        if max_iter is not None: # get prediction at a particular iteration if asked for
            dist = self.staged_pred_dist(X, max_iter=max_iter)[-1]
        elif self.best_val_loss_itr is not None: # this will exist if there's a validation set 
            dist = self.staged_pred_dist(X, max_iter=self.best_val_loss_itr)[-1]
        else: 
            params = np.asarray(self.pred_param(X, max_iter))
            dist = self.Dist(params.T)
        return dist

    def staged_pred_dist(self, X, max_iter=None):
        predictions = []
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
            dists = self.Dist(np.asarray(params).T)
            predictions.append(dists)
            if max_iter and i == max_iter:
                break
        return predictions

    # these methods won't work unless the model is either an NGBRegressor, NGBClassifier, or NGBSurvival object,
    # each of which have the dist_to_prediction() method defined in their own specific way
    def predict(self, X): 
        return self.dist_to_prediction(self.pred_dist(X))

    def staged_predict(self, X, max_iter=None):
        return [self.dist_to_prediction(dist) for dist in self.staged_pred_dist(X, max_iter=None)]

    def get_shap_tree_explainer(self, param_idx=0, **kwargs):
        """
        Return the tree explainer for the param_idx-th parameter in the distribution

        Parameters
        ----------
        param_idx : int
            The index of parameter.
        **kwargs :
            Additional arguments to be passed to shap.TreeExplainer
            (See https://shap.readthedocs.io/en/latest/#shap.TreeExplainer)

        Returns
        -------
        explainer : TreeExplainer object
            Shap TreeExplainer object
        """
        import copy, shap
        assert self.base_models, "Model has empty `base_models`! Have you called `model.fit`?"
        assert str(type(self.base_models[0][param_idx])).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"), "You must use default_tree_learner!"
        temp_model = copy.deepcopy(self)
        temp_model.shap_trees = [trees[param_idx] for trees in temp_model.base_models]
        explainer = shap.TreeExplainer(temp_model, **kwargs)
        return explainer

    @property
    def feature_importances_(self):
        """
        Return the feature importances for all parameters in the distribution
            (the higher, the more important the feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_params, n_features]
            The summation along second axis of this array is an array of ones, 
            unless all trees are single node trees consisting of only the root 
            node, in which case it will be an array of zeros.
        """
        # Check whether the model is fitted
        if not self.base_models:
            return None
        # Check whether the base model is DecisionTreeRegressor
        if not 'sklearn.tree.tree.DecisionTreeRegressor' in str(type(self.base_models[0][0])):
            return None
        # Reshape the base_models
        params_trees = zip(*self.base_models)
        # Get the feature_importances_ for all the params and all the trees
        all_params_importances = [[getattr(tree, 'feature_importances_') 
            for tree in trees if tree.tree_.node_count > 1] 
                for trees in params_trees]

        if not all_params_importances:
            return np.zeros(len(self.base_models[0]),self.base_models[0][0].n_features_, dtype=np.float64)
        # Weighted average of importance by tree scaling factors
        all_params_importances = np.average(all_params_importances,
                                  axis=1, weights=self.scalings)
        return all_params_importances / np.sum(all_params_importances,axis=1,keepdims=True)
