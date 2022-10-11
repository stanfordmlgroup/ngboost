import pickle

import numpy as np
import pytest

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import MultivariateNormal


# name = learners_data to avoid pylint redefined-outer-name
@pytest.fixture(name="learners_data")
def fixture_learners_data(
    breast_cancer_data, california_housing_data, california_housing_survival_data
):
    """
    Returns:
        A list of iterables,
        each iterable containing a fitted model and
        X data and the predictions for the X_data
    """

    models_data = []
    X_class_train, _, Y_class_train, _ = breast_cancer_data
    ngb = NGBClassifier(verbose=False, n_estimators=10)
    ngb.fit(X_class_train, Y_class_train)
    models_data.append((ngb, X_class_train, ngb.predict(X_class_train)))

    X_reg_train, _, Y_reg_train, _ = california_housing_data
    ngb = NGBRegressor(verbose=False, n_estimators=10)
    ngb.fit(X_reg_train, Y_reg_train)
    models_data.append((ngb, X_reg_train, ngb.predict(X_reg_train)))

    X_surv_train, _, T_surv_train, E_surv_train, _ = california_housing_survival_data
    ngb = NGBSurvival(verbose=False, n_estimators=10)
    ngb.fit(X_surv_train, T_surv_train, E_surv_train)
    models_data.append((ngb, X_surv_train, ngb.predict(X_surv_train)))

    ngb = NGBRegressor(Dist=MultivariateNormal(2), n_estimators=10)
    ngb.fit(X_surv_train, np.vstack((T_surv_train, E_surv_train)).T)
    models_data.append((ngb, X_surv_train, ngb.predict(X_surv_train)))
    return models_data


def test_model_save(learners_data):
    """
    Tests that the model can be loaded and predict still works
    It checks that the new predictions are the same as pre-pickling
    """
    for learner, data, preds in learners_data:
        serial = pickle.dumps(learner)
        model = pickle.loads(serial)
        new_preds = model.predict(data)
        assert (new_preds == preds).all()
