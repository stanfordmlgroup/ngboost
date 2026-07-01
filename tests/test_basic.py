import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import Bernoulli, Normal, k_categorical


class RecordingRegressor(BaseEstimator, RegressorMixin):
    # pylint: disable=attribute-defined-outside-init,unused-argument
    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = None if sample_weight is None else sample_weight.copy()
        self.prediction_ = np.average(y, weights=sample_weight)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.prediction_)


def test_classifier_sets_sklearn_classes_and_encodes_labels(breast_cancer_data):
    from sklearn.base import is_classifier  # pylint: disable=import-outside-toplevel
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        RocCurveDisplay,
    )
    from sklearn.model_selection import (  # pylint: disable=import-outside-toplevel
        cross_val_score,
    )

    x_train, x_test, y_train, y_test = breast_cancer_data
    y_labels = ["malignant" if y == 0 else "benign" for y in y_train]
    y_test_labels = ["malignant" if y == 0 else "benign" for y in y_test]

    ngb = NGBClassifier(
        Dist=Bernoulli,
        n_estimators=2,
        verbose=False,
        random_state=0,
    )

    assert is_classifier(ngb)
    assert isinstance(clone(ngb), NGBClassifier)

    ngb.fit(x_train, y_labels)

    assert list(ngb.classes_) == ["benign", "malignant"]
    assert set(ngb.predict(x_test[:10])).issubset(set(ngb.classes_))
    assert ngb.predict_proba(x_test[:10]).shape == (10, 2)
    assert len(ngb.staged_predict(x_test[:10])) == len(ngb.base_models)
    assert cross_val_score(ngb, x_train, y_labels, scoring="roc_auc", cv=3).shape == (
        3,
    )

    display = RocCurveDisplay.from_estimator(
        ngb,
        x_test,
        y_test_labels,
        pos_label="malignant",
    )
    assert display.roc_auc >= 0.5


# TODO: This is non-deterministic in the model fitting
def test_classification(breast_cancer_data):
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        log_loss,
        roc_auc_score,
    )

    x_train, x_test, y_train, y_test = breast_cancer_data
    ngb = NGBClassifier(Dist=Bernoulli, verbose=False)
    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    score = roc_auc_score(y_test, preds)

    # loose score requirement so it isn't failing all the time
    assert score >= 0.85

    preds = ngb.predict_proba(x_test)
    score = log_loss(y_test, preds)
    assert score <= 0.30

    score = ngb.score(x_test, y_test)
    assert score <= 0.30

    dist = ngb.pred_dist(x_test)
    assert isinstance(dist, Bernoulli)

    score = roc_auc_score(y_test, preds[:, 1])

    assert score >= 0.85


# TODO: This is non-deterministic in the model fitting
def test_regression(california_housing_data):
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        mean_squared_error,
    )

    x_train, x_test, y_train, y_test = california_housing_data
    ngb = NGBRegressor(verbose=False)
    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    score = mean_squared_error(y_test, preds)
    assert score <= 15

    score = ngb.score(x_test, y_test)
    assert score <= 15

    dist = ngb.pred_dist(x_test)
    assert isinstance(dist, Normal)

    score = mean_squared_error(y_test, preds)
    assert score <= 15


def test_classifier_validation_fraction_is_supported(breast_cancer_data):
    x_train, x_test, y_train, _ = breast_cancer_data
    ngb = NGBClassifier(
        Dist=Bernoulli,
        n_estimators=25,
        verbose=False,
        random_state=1,
        validation_fraction=0.2,
        early_stopping_rounds=2,
    )

    assert ngb.get_params()["validation_fraction"] == 0.2
    assert ngb.get_params()["early_stopping_rounds"] == 2

    ngb.fit(x_train, y_train)
    preds = ngb.predict(x_test)
    assert preds.shape[0] == x_test.shape[0]


def test_survival_validation_fraction_is_supported(
    california_housing_survival_data,
):
    x_train, x_test, t_train, e_train, _ = california_housing_survival_data
    ngb = NGBSurvival(
        n_estimators=5,
        verbose=False,
        random_state=1,
        validation_fraction=0.2,
        early_stopping_rounds=2,
    )

    assert ngb.get_params()["validation_fraction"] == 0.2
    assert ngb.get_params()["early_stopping_rounds"] == 2

    ngb.fit(x_train, t_train, e_train)
    preds = ngb.predict(x_test)
    assert preds.shape[0] == x_test.shape[0]


def test_regression_accepts_base_learner_per_distribution_parameter():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    Y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=80)

    ngb = NGBRegressor(
        Dist=Normal,
        Base=[DecisionTreeRegressor(max_depth=1), Ridge(alpha=0.0)],
        n_estimators=2,
        natural_gradient=False,
        verbose=False,
    )

    ngb.fit(X, Y)

    assert len(ngb.base_models) == 2
    for models in ngb.base_models:
        assert len(models) == Normal.n_params
        assert isinstance(models[0], DecisionTreeRegressor)
        assert isinstance(models[1], Ridge)

    preds = ngb.predict(X[:5])
    dist = ngb.pred_dist(X[:5])

    assert preds.shape == (5,)
    assert isinstance(dist, Normal)


def test_classification_accepts_base_learner_per_distribution_parameter():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(90, 3))
    Y = np.argmax(np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]]), axis=1)

    ngb = NGBClassifier(
        Dist=k_categorical(3),
        Base=[DecisionTreeRegressor(max_depth=1), DecisionTreeRegressor(max_depth=2)],
        n_estimators=2,
        natural_gradient=False,
        verbose=False,
    )

    ngb.fit(X, Y)

    for models in ngb.base_models:
        assert len(models) == 2
        assert models[0].max_depth == 1
        assert models[1].max_depth == 2

    assert ngb.predict_proba(X[:5]).shape == (5, 3)


def test_base_learner_sequence_must_match_distribution_parameter_count():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    Y = rng.normal(size=20)

    ngb = NGBRegressor(
        Dist=Normal,
        Base=[DecisionTreeRegressor(max_depth=1)],
        n_estimators=1,
        verbose=False,
    )

    with pytest.raises(ValueError, match="one estimator per distribution parameter"):
        ngb.fit(X, Y)

    assert not ngb.base_models
    assert not ngb.scalings
    assert not ngb.col_idxs


def test_base_learner_sequence_accepts_tuple():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 2))
    Y = X[:, 0] + rng.normal(scale=0.1, size=40)

    ngb = NGBRegressor(
        Dist=Normal,
        Base=(DecisionTreeRegressor(max_depth=1), DecisionTreeRegressor(max_depth=2)),
        n_estimators=1,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y)

    assert isinstance(ngb.Base, tuple)
    assert len(ngb.base_models[0]) == Normal.n_params
    assert ngb.base_models[0][0].max_depth == 1
    assert ngb.base_models[0][1].max_depth == 2


def test_parameter_base_learners_support_monotonic_constraints():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(80, 3))
    Y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=80)
    loc_learner = HistGradientBoostingRegressor(
        max_iter=4,
        max_leaf_nodes=3,
        monotonic_cst=[1, 0, 0],
        random_state=0,
    )

    ngb = NGBRegressor(
        Dist=Normal,
        Base=[loc_learner, DecisionTreeRegressor(max_depth=1)],
        n_estimators=1,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y)

    assert isinstance(ngb.base_models[0][0], HistGradientBoostingRegressor)
    assert ngb.base_models[0][0].monotonic_cst == [1, 0, 0]
    assert isinstance(ngb.base_models[0][1], DecisionTreeRegressor)


def test_parameter_base_learners_receive_sample_weight():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 2))
    Y = X[:, 0] + rng.normal(scale=0.1, size=30)
    sample_weight = np.linspace(1.0, 2.0, num=30)

    ngb = NGBRegressor(
        Dist=Normal,
        Base=[RecordingRegressor(), RecordingRegressor()],
        n_estimators=1,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y, sample_weight=sample_weight)

    for model in ngb.base_models[0]:
        np.testing.assert_allclose(model.sample_weight_, sample_weight)


def test_sklearn_clone_accepts_base_learner_sequence():
    ngb = NGBRegressor(
        Dist=Normal,
        Base=[DecisionTreeRegressor(max_depth=1), DecisionTreeRegressor(max_depth=2)],
        n_estimators=1,
        verbose=False,
    )

    cloned = clone(ngb)

    assert cloned is not ngb
    assert isinstance(cloned.Base, list)
    assert cloned.Base is not ngb.Base
    assert cloned.Base[0] is not ngb.Base[0]
    assert cloned.Base[1] is not ngb.Base[1]
    assert cloned.Base[0].max_depth == 1
    assert cloned.Base[1].max_depth == 2


def test_single_base_nested_set_params_updates_base():
    ngb = NGBRegressor(
        Dist=Normal,
        Base=DecisionTreeRegressor(max_depth=1),
        n_estimators=1,
        verbose=False,
    )

    ngb.set_params(Base__max_depth=4)

    assert ngb.Base.max_depth == 4
    assert not hasattr(ngb, "Base__max_depth")


def test_sequence_base_nested_set_params_updates_element():
    ngb = NGBRegressor(
        Dist=Normal,
        Base=[DecisionTreeRegressor(max_depth=1), DecisionTreeRegressor(max_depth=2)],
        n_estimators=1,
        verbose=False,
    )

    ngb.set_params(Base__0__max_depth=5)

    assert ngb.Base[0].max_depth == 5
    assert ngb.Base[1].max_depth == 2
    assert not hasattr(ngb, "Base__0__max_depth")


def test_single_base_learner_matches_repeated_base_learner_sequence():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    Y = X[:, 0] + rng.normal(scale=0.1, size=60)
    base = DecisionTreeRegressor(max_depth=2, random_state=0)

    single_base = NGBRegressor(
        Dist=Normal,
        Base=base,
        n_estimators=3,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y)
    repeated_base = NGBRegressor(
        Dist=Normal,
        Base=[base, base],
        n_estimators=3,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y)

    np.testing.assert_allclose(single_base.pred_param(X), repeated_base.pred_param(X))


def test_feature_importances_have_parameter_rows_for_tree_sequence():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 3))
    Y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=60)

    ngb = NGBRegressor(
        Dist=Normal,
        Base=[DecisionTreeRegressor(max_depth=1), DecisionTreeRegressor(max_depth=2)],
        n_estimators=2,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y)

    importances = ngb.feature_importances_

    assert importances.shape == (Normal.n_params, X.shape[1])
    assert np.isfinite(importances).all()
    np.testing.assert_allclose(importances.sum(axis=1), np.ones(Normal.n_params))


def test_feature_importances_are_none_for_mixed_base_learners():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    Y = X[:, 0] + rng.normal(scale=0.1, size=40)

    ngb = NGBRegressor(
        Dist=Normal,
        Base=[DecisionTreeRegressor(max_depth=1), Ridge(alpha=0.0)],
        n_estimators=1,
        natural_gradient=False,
        verbose=False,
    ).fit(X, Y)

    assert ngb.feature_importances_ is None
