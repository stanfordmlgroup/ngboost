import numpy as np
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import Bernoulli, MultivariateNormal, Normal, k_categorical


class RecordingRegressor(BaseEstimator, RegressorMixin):
    # pylint: disable=attribute-defined-outside-init,unused-argument
    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = None if sample_weight is None else sample_weight.copy()
        self.prediction_ = np.average(y, weights=sample_weight)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.prediction_)


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
