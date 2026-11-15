# pylint: disable=redefined-outer-name,wrong-import-position
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

shap = pytest.importorskip("shap")

from ngboost import NGBClassifier, NGBRegressor  # noqa: E402


@pytest.fixture(scope="module")
def diabetes_data():
    X, y = load_diabetes(return_X_y=True)
    return X[:300], y[:300], X[300:310]


@pytest.fixture(scope="module")
def cancer_data():
    X, y = load_breast_cancer(return_X_y=True)
    return X[:400], y[:400], X[400:410]


def test_shap_regressor_predict(diabetes_data):
    X_train, y_train, X_test = diabetes_data
    ngb = NGBRegressor(n_estimators=20, verbose=False).fit(X_train, y_train)
    explainer = shap.Explainer(ngb.predict, X_train[:50])
    shap_values = explainer(X_test)
    assert shap_values.values.shape == (len(X_test), X_train.shape[1])
    assert np.isfinite(shap_values.values).all()


def test_shap_classifier_predict_proba(cancer_data):
    X_train, y_train, X_test = cancer_data
    ngb = NGBClassifier(n_estimators=20, verbose=False).fit(X_train, y_train)
    explainer = shap.Explainer(ngb.predict_proba, X_train[:50])
    shap_values = explainer(X_test)
    assert shap_values.values.shape[0] == len(X_test)
    assert np.isfinite(shap_values.values).all()


def test_shap_tree_explainer_on_base_learners(diabetes_data):
    X_train, y_train, X_test = diabetes_data
    ngb = NGBRegressor(n_estimators=20, verbose=False).fit(X_train, y_train)
    explainer = shap.TreeExplainer(ngb.base_models[0][0])
    shap_values = explainer.shap_values(X_test)
    assert np.array(shap_values).shape == (len(X_test), X_train.shape[1])
    assert np.isfinite(shap_values).all()
