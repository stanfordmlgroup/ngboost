import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn import config_context
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import (
    Bernoulli,
    Exponential,
    LogNormal,
    MultivariateNormal,
    Normal,
    k_categorical,
)
from ngboost.distns.normal import NormalFixedVar
from ngboost.helpers import Y_from_censored
from ngboost.scores import CRPScore, LogScore


class RecordingRegressor(BaseEstimator, RegressorMixin):
    # pylint: disable=attribute-defined-outside-init,unused-argument
    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = None if sample_weight is None else sample_weight.copy()
        self.prediction_ = np.average(y, weights=sample_weight)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.prediction_)


@pytest.mark.parametrize("n_estimators", [0, 2])
def test_refit_resets_best_validation_iteration(n_estimators):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 3))
    Y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=40)
    options = {
        "Base": Ridge(alpha=0.0),
        "Dist": NormalFixedVar,
        "natural_gradient": False,
        "verbose": False,
        "random_state": 7,
    }
    model = NGBRegressor(n_estimators=4, **options)
    model.fit(X[:30], Y[:30], X_val=X[30:], Y_val=Y[30:])
    assert model.best_val_loss_itr is not None

    model.set_params(n_estimators=n_estimators).fit(X, Y)
    fresh = NGBRegressor(n_estimators=n_estimators, **options).fit(X, Y)

    assert "val" not in model.evals_result
    np.testing.assert_allclose(model.predict(X), fresh.predict(X))
    assert fresh.best_val_loss_itr is None
    assert model.best_val_loss_itr is None


@pytest.mark.parametrize(
    "weight_kind", ["array", "list", "series", "offset_series", "reversed_series"]
)
@pytest.mark.parametrize("minibatch_frac", [1.0, 0.7])
@pytest.mark.parametrize("auto_validation", [False, True])
def test_array_like_sample_weights(weight_kind, minibatch_frac, auto_validation):
    rng = np.random.default_rng(19)
    X = rng.normal(size=(40, 3))
    Y = 3 * X[:, 0] - X[:, 1] + rng.normal(scale=0.3, size=40)
    weights = np.linspace(0.2, 3.0, len(Y))
    sample_weight = {
        "array": weights.copy(),
        "list": weights.tolist(),
        "series": pd.Series(weights),
        "offset_series": pd.Series(weights, index=np.arange(len(Y)) + 100),
        "reversed_series": pd.Series(weights, index=np.arange(len(Y))[::-1]),
    }[weight_kind]
    original_weights = sample_weight.copy()
    options = {
        "n_estimators": 3,
        "verbose": False,
        "random_state": 7,
        "natural_gradient": False,
        "minibatch_frac": minibatch_frac,
        "validation_fraction": 0.25,
        "early_stopping_rounds": 2 if auto_validation else None,
    }
    reference = NGBRegressor(**options).fit(X, Y, sample_weight=weights)
    model = NGBRegressor(**options).fit(X, Y, sample_weight=sample_weight)

    for parameter, expected in reference.pred_dist(X).params.items():
        np.testing.assert_array_equal(model.pred_dist(X).params[parameter], expected)
    if isinstance(sample_weight, pd.Series):
        pd.testing.assert_series_equal(sample_weight, original_weights)
    else:
        np.testing.assert_array_equal(sample_weight, original_weights)


@pytest.mark.parametrize("learner", ["regressor", "classifier", "survival"])
def test_pandas_sample_weights_match_arrays(learner):
    rng = np.random.default_rng(23)
    X = rng.normal(size=(60, 3))
    Y = X[:, 0] + rng.normal(scale=0.3, size=60)
    weights = np.linspace(0.2, 3.0, 60)
    if learner == "regressor":
        model_class, targets = NGBRegressor, [Y]
    elif learner == "classifier":
        model_class, targets = NGBClassifier, [(Y > 0).astype(int)]
    else:
        model_class, targets = NGBSurvival, [np.exp(Y), rng.integers(0, 2, size=60)]
    train_targets = [target[:40] for target in targets]
    val_targets = [target[40:] for target in targets]

    def fit(X_train, w_train, X_val, w_val):
        model = model_class(n_estimators=3, verbose=False, random_state=7)
        return model.fit(
            X_train,
            *train_targets,
            X_val,
            *val_targets,
            sample_weight=w_train,
            val_sample_weight=w_val,
        )

    reference = fit(X[:40], weights[:40], X[40:], weights[40:])
    # Shuffled, offset indexes shared by X and the weights, as when both are
    # columns of one DataFrame.
    train_index = rng.permutation(40) + 100
    val_index = rng.permutation(20) + 500
    model = fit(
        pd.DataFrame(X[:40], index=train_index),
        pd.Series(weights[:40], index=train_index),
        pd.DataFrame(X[40:], index=val_index),
        pd.Series(weights[40:], index=val_index),
    )

    np.testing.assert_array_equal(
        model.pred_param(X[40:]), reference.pred_param(X[40:])
    )
    assert model.evals_result == reference.evals_result


@pytest.mark.parametrize("argument", ["sample_weight", "val_sample_weight"])
@pytest.mark.parametrize(
    "bad_weight, message",
    [
        (np.ones(21), "must have one entry per row"),
        (np.ones(19), "must have one entry per row"),
        (np.ones((20, 1)), "must be 1-D"),
        (np.r_[np.nan, np.ones(19)], "contains NaN"),
    ],
)
def test_invalid_sample_weights_raise(argument, bad_weight, message):
    rng = np.random.default_rng(29)
    X = rng.normal(size=(20, 2))
    Y = X[:, 0]
    with pytest.raises(ValueError, match=rf"\b{argument} {message}"):
        NGBRegressor(n_estimators=1, verbose=False).fit(
            X, Y, X_val=X, Y_val=Y, **{argument: bad_weight}
        )


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


def test_classifier_rejects_label_count_mismatch(breast_cancer_data):
    x_train, _, y_train, _ = breast_cancer_data
    ngb = NGBClassifier(Dist=k_categorical(3), n_estimators=2, verbose=False)

    with pytest.raises(ValueError, match="expects 3 classes, got 2"):
        ngb.fit(x_train, y_train)


def test_classifier_refit_matches_fresh_fit():
    # NGBClassifier.fit encodes the labels and must still go through
    # NGBoost.fit, which resets the random stream and the validation metadata.
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 4))
    Y = np.where(X[:, 0] + rng.normal(size=200) > 0, "yes", "no")
    kw = {
        "n_estimators": 5,
        "minibatch_frac": 0.5,
        "col_sample": 0.5,
        "random_state": 3,
        "verbose": False,
    }
    fresh = NGBClassifier(**kw).fit(X, Y)

    model = NGBClassifier(**kw).fit(X[:150], Y[:150], X_val=X[150:], Y_val=Y[150:])
    assert model.best_val_loss_itr is not None
    model.fit(X, Y)

    assert model.best_val_loss_itr is None
    np.testing.assert_allclose(model.predict_proba(X), fresh.predict_proba(X))

    model.fit(X, np.where(Y == "yes", 1, 0))
    assert list(model.classes_) == [0, 1]


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
    assert score >= -0.30

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
    assert score >= -15

    dist = ngb.pred_dist(x_test)
    assert isinstance(dist, Normal)

    score = mean_squared_error(y_test, preds)
    assert score <= 15


def test_score_is_mean_log_likelihood():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    Y = X @ rng.normal(size=4) + rng.normal(size=200)
    ngb = NGBRegressor(n_estimators=20, verbose=False, random_state=0).fit(X, Y)
    assert ngb.score(X, Y) == pytest.approx(ngb.pred_dist(X).logpdf(Y).mean())


def test_grid_search_prefers_better_model():
    # scikit-learn model selection picks the highest score, so with the default
    # scoring it should prefer more boosting iterations here (issue #418)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    Y = X @ rng.normal(size=4) + rng.normal(size=300)
    search = GridSearchCV(
        NGBRegressor(verbose=False, random_state=0),
        {"n_estimators": [2, 100]},
        cv=3,
    ).fit(X, Y)
    assert search.best_params_ == {"n_estimators": 100}


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


def _survival_data(n=150):
    # Log-normal event times with independent censoring, about 60% events
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    event_time = np.exp(X[:, 0] + 0.3 * rng.normal(size=n))
    censoring_time = np.exp(rng.normal(loc=0.5, size=n))
    T = np.minimum(event_time, censoring_time)
    E = (event_time <= censoring_time).astype(int)
    return X, T, E


@pytest.mark.parametrize("Dist", [LogNormal, Exponential])
def test_survival_clone_keeps_dist(Dist):
    # NGBSurvival used to store a wrapped copy of Dist, so clone failed (issue #270)
    ngb = NGBSurvival(Dist=Dist, n_estimators=10)
    assert ngb.get_params()["Dist"] is Dist

    cloned = clone(ngb)
    assert cloned.get_params()["Dist"] is Dist
    assert cloned.get_params()["n_estimators"] == 10


def test_survival_set_params_dist_takes_effect():
    X, T, E = _survival_data()
    options = {"n_estimators": 10, "verbose": False, "random_state": 0}
    model = NGBSurvival(**options).set_params(Dist=Exponential).fit(X, T, E)
    fresh = NGBSurvival(Dist=Exponential, **options).fit(X, T, E)

    assert model.get_params()["Dist"] is Exponential
    assert isinstance(model.pred_dist(X), Exponential)
    np.testing.assert_allclose(model.predict(X), fresh.predict(X))

    with pytest.raises(ValueError, match="does not have any censored scores"):
        NGBSurvival(**options).set_params(Dist=Normal).fit(X, T, E)


def test_survival_score_is_mean_censored_log_likelihood():
    X, T, E = _survival_data()
    ngb = NGBSurvival(n_estimators=20, verbose=False, random_state=0).fit(X, T, E)
    dist = ngb.pred_dist(X)
    loglik = np.where(E == 1, dist.logpdf(T), dist.logsf(T))

    assert ngb.score(X, T, E) == pytest.approx(loglik.mean(), abs=1e-3)
    assert ngb.score(X, Y_from_censored(T, E)) == ngb.score(X, T, E)
    with pytest.raises(ValueError, match="needs the event indicators E"):
        ngb.score(X, T)


def test_survival_grid_search_with_metadata_routing():
    # GridSearchCV clones the estimator and calls set_params on each clone
    # (issue #270). With metadata routing on, E reaches both fit and score.
    X, T, E = _survival_data()
    with config_context(enable_metadata_routing=True):
        ngb = (
            NGBSurvival(verbose=False, random_state=0)
            .set_fit_request(E=True)
            .set_score_request(E=True)
        )
        search = GridSearchCV(
            ngb,
            {"Dist": [LogNormal, Exponential], "n_estimators": [2, 100]},
            cv=3,
        ).fit(X, T, E=E)

    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    assert search.best_params_ == {"Dist": LogNormal, "n_estimators": 100}


def _positive_regression_data(n=150):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    Y = np.exp(X[:, 0] + 0.3 * rng.normal(size=n))
    return X, Y


@pytest.mark.parametrize("Dist", [LogNormal, Exponential])
def test_regressor_keeps_dist_with_only_censored_scores(Dist):
    # NGBRegressor used to store Dist.uncensor(Score) in place of Dist
    ngb = NGBRegressor(Dist=Dist, n_estimators=10)
    assert ngb.get_params()["Dist"] is Dist
    assert clone(ngb).get_params()["Dist"] is Dist


@pytest.mark.parametrize("Dist", [LogNormal, Exponential])
def test_regressor_set_params_score_with_only_censored_scores(Dist):
    X, Y = _positive_regression_data()
    options = {"Dist": Dist, "n_estimators": 10, "verbose": False, "random_state": 0}
    model = NGBRegressor(**options).set_params(Score=CRPScore).fit(X, Y)
    fresh = NGBRegressor(Score=CRPScore, **options).fit(X, Y)

    assert issubclass(model.Manifold, CRPScore)
    assert not issubclass(model.Manifold, LogScore)
    np.testing.assert_allclose(model.predict(X), fresh.predict(X))


def test_regressor_grid_search_over_score_with_only_censored_scores():
    from sklearn.model_selection import (  # pylint: disable=import-outside-toplevel
        cross_val_score,
    )

    # Each candidate's default score is its own scoring rule, so compare the
    # candidates with one metric
    X, Y = _positive_regression_data()
    options = {"Dist": LogNormal, "n_estimators": 20, "verbose": False}
    search = GridSearchCV(
        NGBRegressor(random_state=0, **options),
        {"Score": [LogScore, CRPScore]},
        scoring="neg_mean_squared_error",
        cv=3,
    ).fit(X, Y)
    direct = cross_val_score(
        NGBRegressor(Score=CRPScore, random_state=0, **options),
        X,
        Y,
        scoring="neg_mean_squared_error",
        cv=3,
    )

    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    assert search.cv_results_["mean_test_score"][1] == pytest.approx(direct.mean())


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


def test_n_jobs_is_a_supported_param():
    ngb = NGBRegressor(n_jobs=4, verbose=False)
    assert ngb.get_params()["n_jobs"] == 4
    assert clone(ngb).get_params()["n_jobs"] == 4


def test_parallel_tree_fits_match_serial():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    W = rng.normal(size=(5, 3))
    Y = X @ W + 0.1 * rng.normal(size=(200, 3))

    kw = {
        "Dist": MultivariateNormal(3),
        "Base": DecisionTreeRegressor(max_depth=3, random_state=0),
        "n_estimators": 10,
        "verbose": False,
    }
    serial = NGBRegressor(n_jobs=1, **kw).fit(X, Y)
    parallel = NGBRegressor(n_jobs=-1, **kw).fit(X, Y)

    np.testing.assert_allclose(serial.pred_param(X), parallel.pred_param(X))


def test_parallel_matches_serial_default_random_state():
    # With the model random_state set, per learner seeding makes the default
    # random_state=None base learner reproducible and independent of n_jobs.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 5))
    W = rng.normal(size=(5, 3))
    Y = X @ W + 0.1 * rng.normal(size=(200, 3))

    kw = {"Dist": MultivariateNormal(3), "n_estimators": 10, "verbose": False}
    serial = NGBRegressor(random_state=0, n_jobs=1, **kw).fit(X, Y)
    parallel = NGBRegressor(random_state=0, n_jobs=-1, **kw).fit(X, Y)

    np.testing.assert_allclose(serial.pred_param(X), parallel.pred_param(X))


def test_refit_and_clone_are_reproducible():
    # random_state is kept as given, so refitting the same model or a clone of
    # it starts from the same random stream as a fresh model.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    Y = X @ rng.normal(size=5) + 0.1 * rng.normal(size=200)

    kw = {
        "n_estimators": 20,
        "minibatch_frac": 0.5,
        "col_sample": 0.6,
        "random_state": 0,
        "verbose": False,
    }
    fresh = NGBRegressor(**kw).fit(X, Y).predict(X)

    model = NGBRegressor(**kw)
    assert model.get_params()["random_state"] == 0
    model.fit(X, Y)
    np.testing.assert_allclose(model.fit(X, Y).predict(X), fresh)
    np.testing.assert_allclose(clone(model).fit(X, Y).predict(X), fresh)


def test_parallel_matches_serial_sparse_float32_csc():
    # A float32 CSC matrix with unsorted indices is the exact input where a
    # threaded in place sort_indices could race across the per parameter fits.
    # fit_base canonicalizes the shared matrix once first, so parallel must
    # still match serial. Repeat a few times to make a regression obvious.
    rng = np.random.default_rng(0)
    Xd = rng.normal(size=(300, 8)).astype(np.float32)
    Xd[Xd < 0.6] = 0.0
    Y = rng.integers(0, 6, size=300)

    def make_X():
        X = sparse.csc_matrix(Xd)
        X.sort_indices()
        for j in range(X.shape[1]):
            s, e = X.indptr[j], X.indptr[j + 1]
            if e - s > 1:
                perm = rng.permutation(e - s)
                X.indices[s:e] = X.indices[s:e][perm]
                X.data[s:e] = X.data[s:e][perm]
        X.has_sorted_indices = False
        return X

    kw = {
        "Dist": k_categorical(6),
        "Base": DecisionTreeRegressor(max_depth=3, random_state=0),
        "n_estimators": 8,
        "verbose": False,
    }
    ref = NGBClassifier(n_jobs=1, **kw).fit(make_X(), Y).pred_param(Xd)
    for _ in range(5):
        got = NGBClassifier(n_jobs=-1, **kw).fit(make_X(), Y).pred_param(Xd)
        np.testing.assert_allclose(got, ref)


def test_parallel_matches_serial_multiclass():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 4))
    Y = np.argmax(np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]]), axis=1)

    kw = {
        "Dist": k_categorical(3),
        "Base": DecisionTreeRegressor(max_depth=2, random_state=0),
        "n_estimators": 10,
        "verbose": False,
    }
    serial = NGBClassifier(n_jobs=1, **kw).fit(X, Y)
    parallel = NGBClassifier(n_jobs=-1, **kw).fit(X, Y)

    np.testing.assert_allclose(serial.predict_proba(X), parallel.predict_proba(X))


def test_parallel_matches_serial_with_missing_values():
    # With a fixed random_state on the base learner, parallel fitting stays
    # deterministic and matches serial even on the missing value path, which
    # otherwise races on numpy's global random state under threads.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 5))
    X[rng.random(X.shape) < 0.1] = np.nan
    Y = rng.normal(size=200)

    kw = {
        "Base": DecisionTreeRegressor(max_depth=3, random_state=0),
        "n_estimators": 10,
        "verbose": False,
    }
    serial = NGBRegressor(n_jobs=1, **kw).fit(X, Y)
    parallel = NGBRegressor(n_jobs=-1, **kw).fit(X, Y)

    np.testing.assert_allclose(serial.pred_param(X), parallel.pred_param(X))


@pytest.mark.parametrize("learner", [NGBRegressor, NGBClassifier, NGBSurvival])
def test_check_is_fitted(learner):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3))
    targets = {
        NGBRegressor: (X[:, 0] + rng.normal(scale=0.1, size=60),),
        NGBClassifier: ((X[:, 0] > 0).astype(int),),
        NGBSurvival: (np.exp(X[:, 0]), rng.random(60) < 0.7),
    }[learner]
    ngb = learner(n_estimators=3, verbose=False)

    with pytest.raises(NotFittedError):
        check_is_fitted(ngb)
    with pytest.raises(NotFittedError):
        check_is_fitted(pickle.loads(pickle.dumps(ngb)))

    ngb.fit(X, *targets)
    check_is_fitted(ngb)
    # A pickle saved before __sklearn_is_fitted__ existed has the same state,
    # so it must load as fitted too
    check_is_fitted(pickle.loads(pickle.dumps(ngb)))


def test_pipeline_predict_sees_fitted_regressor():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 3))
    Y = X[:, 0] + rng.normal(scale=0.1, size=60)
    ngb = NGBRegressor(n_estimators=3, verbose=False, random_state=0)

    with warnings.catch_warnings():
        # scikit-learn < 1.8 warns that the pipeline is not fitted; later
        # versions raise NotFittedError instead
        warnings.filterwarnings("error", message=".*not fitted")
        preds = make_pipeline(StandardScaler(), clone(ngb)).fit(X, Y).predict(X)

    X_scaled = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(preds, ngb.fit(X_scaled, Y).predict(X_scaled))
