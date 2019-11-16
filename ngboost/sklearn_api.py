import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from ngboost.ngboost import NGBoost


class NGBRegressor(NGBoost, RegressorMixin):
    """NGBoost for regression with Sklean API."""

    pass


class NGBClassifier(NGBoost, ClassifierMixin):
    """NGBoost for classification with Sklean API.

    Warning:
        Dist need to be Bernoulli.
        You can use this model for only binary classification.
    """

    def predict(self, X):
        dist = self.pred_dist(X)
        return np.round(dist.prob)

    def predict_proba(self, X):
        num_classes = 2
        y_pred = np.zeros((len(X), num_classes))
        dist = self.pred_dist(X)
        y_pred[:, 1] = dist.prob
        y_pred[:, 0] = 1 - dist.prob
        return y_pred
