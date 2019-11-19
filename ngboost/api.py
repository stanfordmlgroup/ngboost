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
