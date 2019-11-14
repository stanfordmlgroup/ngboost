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
