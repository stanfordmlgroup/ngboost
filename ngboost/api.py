import numpy as np
from ngboost.ngboost import NGBoost
from ngboost.distns import Bernoulli, Normal, LogNormal
from ngboost.scores import MLE
from ngboost.learners import default_tree_learner
from sklearn.base import BaseEstimator


class NGBRegressor(NGBoost, BaseEstimator):

    def __init__(self,
                 Dist=Normal,
                 Score=MLE,
                 Base=default_tree_learner,
                 natural_gradient=True,
                 n_estimators=500,
                 learning_rate=0.01,
                 minibatch_frac=1.0,
                 verbose=True,
                 verbose_eval=100,
                 tol=1e-4):
        assert Dist.problem_type == "regression"
        super().__init__(Dist, Score, Base, natural_gradient, n_estimators, learning_rate,
                         minibatch_frac, verbose, verbose_eval, tol)

    def dist_to_prediction(self, dist): # predictions for regression are typically conditional means
        return dist.mean()

class NGBClassifier(NGBoost, BaseEstimator):

    def __init__(self,
                 Dist=Bernoulli,
                 Score=MLE,
                 Base=default_tree_learner,
                 natural_gradient=True,
                 n_estimators=500,
                 learning_rate=0.01,
                 minibatch_frac=1.0,
                 verbose=True,
                 verbose_eval=100,
                 tol=1e-4):
        assert Dist.problem_type == "classification"
        super().__init__(Dist, Score, Base, natural_gradient, n_estimators, learning_rate,
                         minibatch_frac, verbose, verbose_eval, tol)

    def dist_to_prob(self, dist): 
        num_classes = 2 # will need to change this when categorical distribution is added
        p_hat_1 = dist.prob
        p_hat = np.zeros((len(p_hat_1), num_classes))
        p_hat[:, 1] = dist.prob
        p_hat[:, 0] = 1 - dist.prob
        return p_hat

    def predict_proba(self, X, max_iter=None):
        return self.dist_to_prob(self.pred_dist(X, max_iter=max_iter))

    def staged_predict_proba(self, X, max_iter=None):
        return [self.dist_to_prob(dist) for dist in self.staged_pred_dist(X, max_iter=max_iter)]

    def dist_to_prediction(self, dist): # returns class assignments
        return np.argmax(self.dist_to_prob(dist), 1)

class NGBSurvival(NGBoost, BaseEstimator):

    def __init__(self,
                 Dist=LogNormal,
                 Score=MLE,
                 Base=default_tree_learner,
                 natural_gradient=True,
                 n_estimators=500,
                 learning_rate=0.01,
                 minibatch_frac=1.0,
                 verbose=True,
                 verbose_eval=100,
                 tol=1e-4):
        assert Dist.problem_type == "survival"
        super().__init__(Dist, Score, Base, natural_gradient, n_estimators, learning_rate,
                         minibatch_frac, verbose, verbose_eval, tol)

    def dist_to_prediction(self, dist): # predictions for regression are typically conditional means
        return dist.mean()
