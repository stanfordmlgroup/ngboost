import pytest
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer
from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
import numpy as np
import os.path
import pickle


class TestModelSave:
    @pytest.fixture(scope="class")
    def class_data(self):
        X, Y = load_breast_cancer(True)
        return train_test_split(X, Y, test_size=0.2)

    @pytest.fixture(scope="class")
    def reg_data(self):
        X, Y = load_boston(True)
        return train_test_split(X, Y, test_size=0.2)

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

    @pytest.fixture(scope="class")
    def learners(self, class_data, reg_data, surv_data):
        X_reg_train, X_reg_test, Y_reg_train, Y_reg_test = reg_data
        X_class_train, X_class_test, Y_class_train, Y_class_test = class_data
        X_surv_train, X_surv_test, T_surv_train, E_surv_train, Y_surv_test = surv_data
        ngb_class = NGBClassifier(verbose=False)
        ngb_class.fit(X_class_train, Y_class_train)
        ngb_reg = NGBRegressor(verbose=False)
        ngb_reg.fit(X_reg_train, Y_reg_train)
        ngb_surv = NGBSurvival(verbose=False)
        ngb_surv.fit(X_surv_train, T_surv_train, E_surv_train)
        return ngb_class, ngb_reg, ngb_surv

    def test_model_save(self, learners):
        # save out the pickle
        for learner in learners:
            pickle.dump(learner, open("ngbtest.p", "wb"))
            assert os.path.isfile("ngbtest.p")
