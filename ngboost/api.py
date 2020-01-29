import numpy as np
from ngboost.ngboost import NGBoost
from ngboost.distns import RegressionDistn, ClassificationDistn
from ngboost.distns import Bernoulli, Normal, LogNormal
from ngboost.scores import LogScore
from ngboost.learners import default_tree_learner
from sklearn.base import BaseEstimator


class NGBRegressor(NGBoost, BaseEstimator):

    def __init__(self,
                 Dist=Normal,
                 Score=LogScore,
                 Base=default_tree_learner,
                 natural_gradient=True,
                 n_estimators=500,
                 learning_rate=0.01,
                 minibatch_frac=1.0,
                 verbose=True,
                 verbose_eval=100,
                 tol=1e-4,
                 random_state=None):
        assert issubclass(Dist, RegressionDistn), f'{Dist.__name__} is not useable for regression.'
        super().__init__(Dist, Score, Base, natural_gradient, n_estimators, learning_rate,
                         minibatch_frac, verbose, verbose_eval, tol, random_state)

class NGBClassifier(NGBoost, BaseEstimator):

    def __init__(self,
                 Dist=Bernoulli,
                 Score=LogScore,
                 Base=default_tree_learner,
                 natural_gradient=True,
                 n_estimators=500,
                 learning_rate=0.01,
                 minibatch_frac=1.0,
                 verbose=True,
                 verbose_eval=100,
                 tol=1e-4,
                 random_state=None):
        assert issubclass(Dist, ClassificationDistn), f'{Dist.__name__} is not useable for classification.'
        super().__init__(Dist, Score, Base, natural_gradient, n_estimators, learning_rate,
                         minibatch_frac, verbose, verbose_eval, tol, random_state)

    def predict_proba(self, X, max_iter=None):
        return self.pred_dist(X, max_iter=max_iter).class_probs()

    def staged_predict_proba(self, X, max_iter=None):
        return [dist.class_probs() for dist in self.staged_pred_dist(X, max_iter=max_iter)]

class NGBSurvival(NGBoost, BaseEstimator):

    def __init__(self,
                 Dist=LogNormal,
                 Score=LogScore,
                 Base=default_tree_learner,
                 natural_gradient=True,
                 n_estimators=500,
                 learning_rate=0.01,
                 minibatch_frac=1.0,
                 verbose=True,
                 verbose_eval=100,
                 tol=1e-4):
        # do something else here to check survival
        super().__init__(Dist, Score, Base, natural_gradient, n_estimators, learning_rate,
                         minibatch_frac, verbose, verbose_eval, tol, random_state)

