import pytest

import numpy as np
from sklearn.model_selection import train_test_split

from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, Normal


def test_classification():
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_auc_score, log_loss

    data, target = load_breast_cancer(True)
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )
    ngb = NGBClassifier(Dist=Bernoulli, verbose=False)
    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    score = roc_auc_score(y_test, preds)
    assert score >= 0.95

    preds = ngb.predict_proba(x_test)
    score = log_loss(y_test, preds)
    assert score <= 0.20

    score = ngb.score(x_test, y_test)
    assert score <= 0.20

    dist = ngb.pred_dist(x_test)
    assert isinstance(dist, Bernoulli)

    score = roc_auc_score(y_test, preds[:, 1])
    assert score >= 0.95


def test_regression():
    from sklearn.datasets import load_boston
    from sklearn.metrics import mean_squared_error

    data, target = load_boston(True)
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )
    ngb = NGBRegressor(verbose=False)
    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    score = mean_squared_error(y_test, preds)
    assert score <= 8.0

    score = ngb.score(x_test, y_test)
    assert score <= 8.0

    dist = ngb.pred_dist(x_test)
    assert isinstance(dist, Normal)

    score = mean_squared_error(y_test, preds)
    assert score <= 8.0
