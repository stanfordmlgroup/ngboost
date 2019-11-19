import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

from ngboost.distns import Bernoulli, Normal
from ngboost.learners import default_tree_learner
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE

np.random.seed(1)


class TestBasic(unittest.TestCase):
    def test_basic(self):
        pass

    def test_classification(self):
        data, target = load_breast_cancer(True)
        x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=0.2,
                                                            random_state=42)
        ngb = NGBoost(Base=default_tree_learner, Dist=Bernoulli, Score=MLE,
                      verbose=False)
        ngb.fit(x_train, y_train)
        preds = ngb.pred_dist(x_test)
        score = roc_auc_score(y_test, preds.prob)
        assert score >= 0.95

    def test_regression(self):
        data, target = load_boston(True)
        x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=0.2,
                                                            random_state=42)
        ngb = NGBoost(Base=default_tree_learner, Dist=Normal, Score=MLE,
                      natural_gradient=True, verbose=False)
        ngb.fit(x_train, y_train)
        preds = ngb.predict(x_test)
        score = mean_squared_error(y_test, preds)
        assert score <= 8.0
