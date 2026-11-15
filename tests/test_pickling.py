import pickle

import numpy as np
import pytest

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import LogNormal, MultivariateNormal


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

    # LogNormal has only censored scores, so NGBRegressor uncensors them
    ngb = NGBRegressor(Dist=LogNormal, verbose=False, n_estimators=10)
    ngb.fit(X_reg_train, Y_reg_train)
    models_data.append((ngb, X_reg_train, ngb.predict(X_reg_train)))
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


def test_mvn_params_after_pickle():
    """
    A pickled MultivariateNormal model gives the same params, with the cov key,
    and the params dict itself keeps the deprecated scale alias through pickle
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    Y = X[:, :2] + rng.normal(size=(100, 2))
    ngb = NGBRegressor(Dist=MultivariateNormal(2), n_estimators=10, verbose=False)
    ngb.fit(X, Y)
    params = ngb.pred_dist(X).params

    model = pickle.loads(pickle.dumps(ngb))
    new_params = model.pred_dist(X).params
    assert list(new_params) == ["loc", "cov"]
    np.testing.assert_array_equal(new_params["loc"], params["loc"])
    np.testing.assert_array_equal(new_params["cov"], params["cov"])

    params_copy = pickle.loads(pickle.dumps(params))
    assert list(params_copy) == ["loc", "cov"]
    with pytest.warns(FutureWarning):
        np.testing.assert_array_equal(params_copy["scale"], params["cov"])


# The types that the models in learners_data ask skops to trust. A user loads a
# file with a fixed list like this one, and refuses a file that asks for more.
SKOPS_TRUSTED_TYPES = [
    "ngboost.api.NGBClassifier",
    "ngboost.api.NGBRegressor",
    "ngboost.api.NGBSurvival",
    "ngboost.distns.lognormal.LogNormal",
    "ngboost.distns.normal.Normal",
    "ngboost.scores.LogScore",
    "sklearn.tree._tree.Tree",
]


def test_model_save_skops(learners_data):
    """
    Tests that the model can be saved and loaded with skops, and that loading
    needs only a fixed list of trusted types.
    It checks that the new predictions are the same as before saving
    """
    sio = pytest.importorskip("skops.io")
    for learner, data, preds in learners_data:
        serial = sio.dumps(learner)
        assert set(sio.get_untrusted_types(data=serial)) <= set(SKOPS_TRUSTED_TYPES)
        model = sio.loads(serial, trusted=SKOPS_TRUSTED_TYPES)
        new_preds = model.predict(data)
        assert (new_preds == preds).all()
