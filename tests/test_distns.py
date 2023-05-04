from typing import Tuple

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import (
    Bernoulli,
    Cauchy,
    Distn,
    Exponential,
    Gamma,
    LogNormal,
    MultivariateNormal,
    Normal,
    NormalFixedMean,
    NormalFixedVar,
    T,
    TFixedDf,
    TFixedDfFixedVar,
    k_categorical,
)
from ngboost.scores import CRPScore, LogScore, Score

# test all the dist methods and score implementation methods,#
# i.e. they all return proper shapes and sizes and types
# check metric lines up with defaults for lognormal where applicable


Tuple4Array = Tuple[np.array, np.array, np.array, np.array]
Tuple5Array = Tuple[np.array, np.array, np.array, np.array, np.array]


@pytest.mark.slow
@pytest.mark.parametrize(
    "dist",
    [
        Normal,
        NormalFixedVar,
        NormalFixedMean,
        LogNormal,
        Exponential,
        Gamma,
        T,
        TFixedDf,
        TFixedDfFixedVar,
        Cauchy,
    ],
)
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
    ],
)
def test_dists_runs_on_examples_logscore(dist: Distn, learner, california_housing_data):
    X_train, X_test, y_train, y_test = california_housing_data
    # TODO: test early stopping features
    ngb = NGBRegressor(Dist=dist, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # TODO: test properties of output


@pytest.mark.slow
@pytest.mark.parametrize("dist", [Normal, LogNormal, Exponential])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
    ],
)
def test_dists_runs_on_examples_crpscore(dist: Distn, learner, california_housing_data):
    X_train, X_test, y_train, y_test = california_housing_data
    # TODO: test early stopping features
    ngb = NGBRegressor(Dist=dist, Score=CRPScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)
    # TODO: test properties of output


@pytest.mark.parametrize("dist", [LogNormal, Exponential])
@pytest.mark.parametrize("score", [LogScore, CRPScore])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
    ],
)
def test_survival_runs_on_examples(
    dist: Distn, score: Score, learner, california_housing_survival_data
):
    (
        X_train,
        X_test,
        T_surv_train,
        E_surv_train,
        Y_surv_test,
    ) = california_housing_survival_data
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
@pytest.mark.parametrize("k", [2, 4, 7])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
    ],
)
def test_categorical(k: int, learner, breast_cancer_data: Tuple4Array):
    X_train, X_test, y_train, _ = breast_cancer_data
    dist = k_categorical(k)
    y_train = np.random.randint(0, k, (len(y_train)))
    # test early stopping features
    ngb = NGBClassifier(Dist=dist, Score=LogScore, Base=learner, verbose=False)
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_prob = ngb.predict_proba(X_test)
    y_dist = ngb.pred_dist(X_test)
    # test properties of output


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize(
    "learner",
    [
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
    ],
)
# Ignore the k=1 warning
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multivariatenormal(k: 2, learner):
    dist = MultivariateNormal(k)

    # Generate some sample data
    N = 500
    X_train = np.random.randn(N, k)
    y_fns = [np.sin, np.cos, np.exp]
    y_cols = [
        fn(X_train[:, num_col]).reshape(-1, 1) + np.random.randn(N, 1)
        for num_col, fn in enumerate(y_fns[:k])
    ]
    y_train = np.hstack(y_cols)
    X_test = np.random.randn(N, k)

    ngb = NGBRegressor(
        Dist=dist, Score=LogScore, Base=learner, verbose=False, n_estimators=50
    )
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)
    y_dist = ngb.pred_dist(X_test)

    mean = y_dist.mean
    sample = y_dist.rv()
    scipy_list = y_dist.scipy_distribution()
