import pytest
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer
from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import MultivariateNormal
import numpy as np
import pickle


class TestModelSave:
    @pytest.fixture(scope="class")
    def class_data(self):
        X, Y = load_breast_cancer(return_X_y=True)
        return train_test_split(X, Y, test_size=0.2)

    @pytest.fixture(scope="class")
    def reg_data(self):
        X, Y = load_boston(return_X_y=True)
        return train_test_split(X, Y, test_size=0.2)

    @pytest.fixture(scope="class")
    def surv_data(self):
        X, Y = load_boston(return_X_y=True)
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
    def learners_data(self, class_data, reg_data, surv_data):
        """
        Returns:
            A list of iterables,
            each iterable containing a fitted model and
            X data and the predictions for the X_data
        """
        X_reg_train, X_reg_test, Y_reg_train, Y_reg_test = reg_data
        X_class_train, X_class_test, Y_class_train, Y_class_test = class_data
        X_surv_train, X_surv_test, T_surv_train, E_surv_train, Y_surv_test = surv_data
        models_data = []
        ngb_class = NGBClassifier(verbose=False, n_estimators=10)
        ngb_class.fit(X_class_train, Y_class_train)
        models_data.append((ngb_class, X_class_train, ngb_class.predict(X_class_train)))
        ngb_reg = NGBRegressor(verbose=False, n_estimators=10)
        ngb_reg.fit(X_reg_train, Y_reg_train)
        models_data.append((ngb_reg, X_reg_train, ngb_reg.predict(X_reg_train)))
        ngb_surv = NGBSurvival(verbose=False, n_estimators=10)
        ngb_surv.fit(X_surv_train, T_surv_train, E_surv_train)
        models_data.append((ngb_reg, X_surv_train, ngb_reg.predict(X_surv_train)))
        ngb_mvn = NGBRegressor(Dist=MultivariateNormal(2), n_estimators=10)
        ngb_mvn.fit(X_surv_train, np.vstack([T_surv_train,E_surv_train]).T)
        models_data.append((ngb_mvn, X_surv_train, ngb_mvn.predict(X_surv_train)))
        return models_data

    def test_model_save(self, learners_data):
        """
            Tests that the model can be loaded and predict still works
            It checks that the new predictions are the same as pre-pickling
        """
        for learner, data, preds in learners_data:
            serial = pickle.dumps(learner)
            model = pickle.loads(serial)
            new_preds = model.predict(data)
            assert (new_preds==preds).all()
