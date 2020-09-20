from itertools import product
from typing import List, Iterable, Tuple


import numpy as np
import pytest
from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import (
    Bernoulli,
    Distn,
    Exponential,
    LogNormal,
    Normal,
    k_categorical,
)
from ngboost.scores import CRPScore, LogScore, Score
from sklearn.tree import DecisionTreeRegressor

# test all the dist methods and score implementation methods, i.e. they all return proper shapes and sizes and types
# check metric lines up with defaults for lognormal where applicable


Tuple4Array = Tuple[np.array, np.array, np.array, np.array]
Tuple5Array = Tuple[np.array, np.array, np.array, np.array, np.array]


def product_list(*its: Iterable) -> List:
    """Convenience to create a list of the cartesian product of input iterables

    This is mostly so the parametrized functions below can be a bit cleaner.
    """
    return list(product(*its))


@pytest.mark.slow
@pytest.mark.parametrize(
    ["dist", "score", "learner"],
    product_list(
        [Normal, LogNormal, Exponential],
        [LogScore, CRPScore],
        [
            DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
            DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
        ],
    ),
)
def test_dists_runs_on_examples(
    dist: Distn, score: Score, learner, boston_data: Tuple4Array
):
    X_train, X_test, y_train, y_test = boston_data
    # TODO: test early stopping features
    ngb = NGBRegressor(Dist=dist, Score=score, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # TODO: test properties of output


@pytest.mark.slow
@pytest.mark.parametrize(
    ["dist", "score", "learner"],
    product_list(
        [LogNormal, Exponential],
        [LogScore, CRPScore],
        [
            DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
            DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
        ],
    ),
)
def test_survival_runs_on_examples(
    dist: Distn, score: Score, learner, boston_survival_data: Tuple5Array
):
    X_train, X_test, T_surv_train, E_surv_train, Y_surv_test = boston_survival_data
    # test early stopping features
    ngb = NGBSurvival(Dist=dist, Score=score, Base=learner, verbose=False)
    ngb.fit(X_train, T_surv_train, E_surv_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # test properties of output


@pytest.mark.slow
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
    ],
)
def test_bernoulli(learner, breast_cancer_data: Tuple4Array):
    X_cls_train, X_cls_test, Y_cls_train, Y_cls_test = breast_cancer_data
    # test early stopping features
    # test other args, n_trees, LR, minibatching- args as fixture
    ngb = NGBClassifier(Dist=Bernoulli, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_cls_train, Y_cls_train)
    y_pred = ngb.predict(X_cls_test)
    y_prob = ngb.predict_proba(X_cls_test)
    y_dist = ngb.pred_dist(X_cls_test)
    # test properties of output


@pytest.mark.slow
@pytest.mark.parametrize(
    ["k", "learner"],
    product_list(
        [2, 4, 7],
        [
            DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
            DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
        ],
    ),
)
def test_categorical(k: int, learner, breast_cancer_data: Tuple4Array):
    X_train, X_test, y_train, y_test = breast_cancer_data
    dist = k_categorical(k)
    y_train = np.random.randint(0, k, (len(y_train)))
    # test early stopping features
    ngb = NGBClassifier(Dist=dist, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_prob = ngb.predict_proba(X_test)
    y_dist = ngb.pred_dist(X_test)
    # test properties of output


# test slicing and ._params
