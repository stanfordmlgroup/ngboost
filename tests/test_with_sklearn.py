import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from ngboost.distns import Bernoulli
from ngboost.sklearn_api import NGBClassifier, NGBRegressor

np.random.seed(1)


class TestSklearn(unittest.TestCase):
    def test_classification(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import roc_auc_score
        data, target = load_breast_cancer(True)
        x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=0.2,
                                                            random_state=42)
        ngb = NGBClassifier(Dist=Bernoulli, verbose=False)
        ngb.fit(x_train, y_train)
        preds = ngb.predict(x_test)
        score = roc_auc_score(y_test, preds)
        assert score >= 0.95

    def test_regression(self):
        from sklearn.datasets import load_boston
        from sklearn.metrics import mean_squared_error
        data, target = load_boston(True)
        x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=0.2,
                                                            random_state=42)
        ngb = NGBRegressor(verbose=False)
        ngb.fit(x_train, y_train)
        preds = ngb.predict(x_test)
        score = mean_squared_error(y_test, preds)
        assert score <= 8.0
