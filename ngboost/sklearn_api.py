import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from ngboost.ngboost import NGBoost
from ngboost.distns import Bernoulli, Normal


class NGBRegressor(NGBoost, RegressorMixin):
    """NGBoost for regression with Sklean API."""
    def __init__(self, *args, **kwargs):
        super(NGBRegressor, self).__init__(Dist=Normal, *args, **kwargs)


class NGBClassifier(NGBoost, ClassifierMixin):
    """NGBoost for classification with Sklean API.

    Warning:
        Dist need to be Bernoulli.
        You can use this model for only binary classification.
    """
    def __init__(self, *args, **kwargs):
        super(NGBClassifier, self).__init__(Dist=Bernoulli, *args, **kwargs)

    def predict(self, X):
        dist = self.pred_dist(X)
        return np.round(dist.prob)
