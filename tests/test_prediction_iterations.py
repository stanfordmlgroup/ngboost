import numpy as np
import pytest

from ngboost import NGBClassifier, NGBRegressor


@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("max_iter", [0, 1, None])
def test_prediction_iteration_limits(classifier, max_iter):
    rng = np.random.RandomState(12)
    X = rng.normal(size=(30, 3))
    y = np.where(X[:, 0] > 0, "yes", "no") if classifier else X[:, 0]
    estimator = NGBClassifier if classifier else NGBRegressor
    model = estimator(n_estimators=3, verbose=False, random_state=7).fit(X, y)
    count = len(model.base_models) if max_iter is None else max_iter
    expected_params = np.tile(model.init_params, (len(X), 1))
    for models, scale, columns in list(
        zip(model.base_models, model.scalings, model.col_idxs)
    )[:count]:
        residuals = np.array([base.predict(X[:, columns]) for base in models]).T
        expected_params -= model.learning_rate * scale * residuals
    expected_dist = model.Dist(expected_params.T)
    expected_prediction = expected_dist.predict()
    if classifier:
        expected_prediction = model.classes_[expected_prediction]

    np.testing.assert_allclose(model.pred_param(X, max_iter), expected_params)
    np.testing.assert_allclose(model.pred_dist(X, max_iter)._params, expected_params.T)
    np.testing.assert_array_equal(model.predict(X, max_iter), expected_prediction)
    staged_dist = model.staged_pred_dist(X, max_iter)
    staged_predictions = model.staged_predict(X, max_iter)
    assert len(staged_dist) == count
    assert len(staged_predictions) == count
    if count:
        np.testing.assert_allclose(staged_dist[-1]._params, expected_params.T)
        np.testing.assert_array_equal(staged_predictions[-1], expected_prediction)
    if classifier:
        np.testing.assert_allclose(
            model.predict_proba(X, max_iter), expected_dist.class_probs()
        )
        staged_probabilities = model.staged_predict_proba(X, max_iter)
        assert len(staged_probabilities) == count
        if count:
            np.testing.assert_allclose(
                staged_probabilities[-1], expected_dist.class_probs()
            )


@pytest.mark.parametrize("classifier", [False, True])
def test_zero_staged_predictions_are_empty(classifier):
    X = np.arange(30).reshape(10, 3)
    y = np.arange(10) % 2
    estimator = NGBClassifier if classifier else NGBRegressor
    model = estimator(n_estimators=2, verbose=False, random_state=7).fit(X, y)
    assert model.staged_pred_dist(X, max_iter=0) == []
    assert model.staged_predict(X, max_iter=0) == []
    if classifier:
        assert model.staged_predict_proba(X, max_iter=0) == []
