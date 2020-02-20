import pytest

from ngboost.distns import Normal, LogNormal, Exponential, Bernoulli, k_categorical
from ngboost.scores import LogScore, CRPScore
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBRegressor, NGBClassifier, NGBSurvival

from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np

# test all the dist methods and score implementation methods, i.e. they all return proper shapes and sizes and types
# check metric lines up with defaults for lognormal where applicable


@pytest.fixture(scope="module")
def learners():
    # add some learners that aren't trees
    return [
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=5),
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=3),
    ]


class TestRegDistns:
    @pytest.fixture(scope="class")
    def reg_dists(self):
        return {
            Normal: [LogScore, CRPScore],
            LogNormal: [LogScore, CRPScore],
            Exponential: [LogScore, CRPScore],
        }

    @pytest.fixture(scope="class")
    def reg_data(self):
        X, Y = load_boston(True)
        return train_test_split(X, Y, test_size=0.2)

    def test_dists(self, learners, reg_dists, reg_data):
        X_reg_train, X_reg_test, Y_reg_train, Y_reg_test = reg_data
        for Dist, Scores in reg_dists.items():
            for Score in Scores:
                for Learner in learners:
                    # test early stopping features
                    ngb = NGBRegressor(
                        Dist=Dist, Score=Score, Base=Learner, verbose=False
                    )
                    ngb.fit(X_reg_train, Y_reg_train)
                    y_pred = ngb.predict(X_reg_test)
                    y_dist = ngb.pred_dist(X_reg_test)
                    # test properties of output

    # test what happens when a dist that's not regression is passed in


class TestSurvDistns:
    @pytest.fixture(scope="class")
    def surv_dists(self):
        return {LogNormal: [LogScore, CRPScore], Exponential: [LogScore, CRPScore]}

    @pytest.fixture(scope="class")
    def surv_data(self):
        X, Y = load_boston(True)
        X_surv_train, X_surv_test, Y_surv_train, Y_surv_test = train_test_split(
            X, Y, test_size=0.2
        )

        # introduce administrative censoring to simulate survival data
        T_surv_train = np.minimum(Y_surv_train, 30)  # time of an event or censoring
        E_surv_train = (
            Y_surv_train > 30
        )  # 1 if T[i] is the time of an event, 0 if it's a time of censoring
        return X_surv_train, X_surv_test, T_surv_train, E_surv_train, Y_surv_test

    def test_dists(self, learners, surv_dists, surv_data):
        X_surv_train, X_surv_test, T_surv_train, E_surv_train, Y_surv_test = surv_data
        for Dist, Scores in surv_dists.items():
            for Score in Scores:
                for Learner in learners:
                    # test early stopping features
                    ngb = NGBSurvival(
                        Dist=Dist, Score=Score, Base=Learner, verbose=False
                    )
                    ngb.fit(X_surv_train, T_surv_train, E_surv_train)
                    y_pred = ngb.predict(X_surv_test)
                    y_dist = ngb.pred_dist(X_surv_test)
                    # test properties of output


class TestClsDistns:
    @pytest.fixture(scope="class")
    def cls_data(self):
        X, Y = load_breast_cancer(True)
        return train_test_split(X, Y, test_size=0.2)

    def test_bernoulli(self, learners, cls_data):
        X_cls_train, X_cls_test, Y_cls_train, Y_cls_test = cls_data
        for Learner in learners:
            # test early stopping features
            # test other args, n_trees, LR, minibatching- args as fixture
            ngb = NGBClassifier(
                Dist=Bernoulli, Score=LogScore, Base=Learner, verbose=False
            )
            ngb.fit(X_cls_train, Y_cls_train)
            y_pred = ngb.predict(X_cls_test)
            y_prob = ngb.predict_proba(X_cls_test)
            y_dist = ngb.pred_dist(X_cls_test)
            # test properties of output

    def test_categorical(self, learners, cls_data):
        X_cls_train, X_cls_test, Y_cls_train, Y_cls_test = cls_data
        for K in [2, 4, 7]:
            Dist = k_categorical(K)
            Y_cls_train = np.random.randint(0, K, (len(Y_cls_train)))

            for Learner in learners:
                # test early stopping features
                ngb = NGBClassifier(
                    Dist=Dist, Score=LogScore, Base=Learner, verbose=False
                )
                ngb.fit(X_cls_train, Y_cls_train)
                y_pred = ngb.predict(X_cls_test)
                y_prob = ngb.predict_proba(X_cls_test)
                y_dist = ngb.pred_dist(X_cls_test)
                # test properties of output


# test slicing and ._params
