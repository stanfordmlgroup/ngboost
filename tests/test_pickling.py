import io
import os
import pickle
import tempfile

import joblib
import numpy as np
import pytest
import scipy.stats
import sklearn.tree._tree as _sklearn_tree  # pylint: disable=c-extension-no-member
import sympy as sp
import sympy.stats as symstats
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival, load_ngboost_model
from ngboost.distns import (
    Beta,
    BetaBernoulli,
    BetaBinomial,
    BetaBinomialEstN,
    LogitNormal,
    MultivariateNormal,
    NegativeBinomial,
)
from ngboost.distns.sympy_utils import make_distribution

# A distribution built with the SymPy factory outside ngboost, as a user would
_alpha, _beta, _y = sp.symbols("alpha beta y", positive=True)
UserBeta = make_distribution(
    params=[(_alpha, True), (_beta, True)],
    y=_y,
    sympy_dist=symstats.Beta("Y", _alpha, _beta),
    scipy_dist_cls=scipy.stats.beta,
    scipy_kwarg_map={"a": _alpha, "b": _beta},
    name="UserBeta",
)

FACTORY_DISTS = [
    Beta,
    BetaBernoulli,
    BetaBinomial,
    BetaBinomialEstN,
    LogitNormal,
    NegativeBinomial,
]


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


def test_classifier_setstate_restores_missing_label_metadata(breast_cancer_data):
    """Older classifier pickles do not include the sklearn label metadata."""

    X_train, _, Y_train, _ = breast_cancer_data
    ngb = NGBClassifier(verbose=False, n_estimators=2)
    ngb.fit(X_train, Y_train)

    state = ngb.__getstate__()
    state.pop("classes_", None)
    state.pop("_le", None)

    model = NGBClassifier()
    model.__setstate__(state)

    assert np.array_equal(model.classes_, np.array([0, 1]))
    label_classes = model._le.classes_  # pylint: disable=protected-access
    assert np.array_equal(label_classes, model.classes_)
    assert np.array_equal(model.predict(X_train[:5]), ngb.predict(X_train[:5]))
    assert np.allclose(model.predict_proba(X_train[:5]), ngb.predict_proba(X_train[:5]))


def test_model_save_skops(learners_data):
    """
    Tests that the model can be saved and loaded with skops
    It checks that the new predictions are the same as before saving
    """
    sio = pytest.importorskip("skops.io")
    for learner, data, preds in learners_data:
        serial = sio.dumps(learner)
        model = sio.loads(serial, trusted=sio.get_untrusted_types(data=serial))
        new_preds = model.predict(data)
        assert (new_preds == preds).all()


@pytest.mark.parametrize(
    "dist", FACTORY_DISTS + [UserBeta], ids=lambda dist: dist.__name__
)
def test_factory_distn_pickle(dist):
    """Distributions made with make_distribution pickle by reference"""
    assert pickle.loads(pickle.dumps(dist)) is dist

    pred_dist = dist(np.zeros((dist.n_params, 3)))
    new_pred_dist = pickle.loads(pickle.dumps(pred_dist))
    assert isinstance(new_pred_dist, dist)
    for name, value in pred_dist.params.items():
        assert np.allclose(new_pred_dist.params[name], value)


def _factory_model(dist):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    Y = scipy.stats.beta(np.exp(X[:, 0]), 2).rvs(random_state=rng)
    return NGBRegressor(Dist=dist, n_estimators=5, verbose=False).fit(X, Y), X


@pytest.mark.parametrize(
    "dist", [Beta, LogitNormal, UserBeta], ids=lambda dist: dist.__name__
)
def test_factory_distn_model_save(dist):
    # compare parameters, since LogitNormal predicts a Monte Carlo mean
    ngb, X = _factory_model(dist)
    model = pickle.loads(pickle.dumps(ngb))
    new_params = model.pred_dist(X).params
    for name, value in ngb.pred_dist(X).params.items():
        assert np.allclose(new_params[name], value)


def test_factory_distn_model_save_skops():
    sio = pytest.importorskip("skops.io")
    ngb, X = _factory_model(Beta)
    serial = sio.dumps(ngb)
    model = sio.loads(serial, trusted=sio.get_untrusted_types(data=serial))
    assert np.array_equal(model.predict(X), ngb.predict(X))


# ---------------------------------------------------------------------------
# Helpers for backward-compatibility test (issue #389)
# ---------------------------------------------------------------------------


def _sklearn_has_missing_go_to_left():
    """Return True if the current sklearn stores missing_go_to_left in tree nodes."""
    dt = DecisionTreeRegressor(max_depth=1)
    dt.fit([[0], [1]], [0, 1])
    return "missing_go_to_left" in (dt.tree_.__getstate__()["nodes"].dtype.names or ())


def _make_old_style_pickle_bytes(model):
    """Return pickle bytes of *model* with ``missing_go_to_left`` stripped from
    every tree's node array, simulating a file produced by scikit-learn < 1.3.

    Uses a pickle dispatch table to intercept Tree serialisation without
    modifying the (immutable) extension type in place.
    """

    def _old_tree_reducer(tree):
        """Reducer that omits ``missing_go_to_left`` from the node dtype."""
        # tree.__reduce__() → (Tree, (n_features, n_classes, n_outputs), state)
        cls, args, state = tree.__reduce__()
        nodes = state.get("nodes")
        if nodes is not None and "missing_go_to_left" in (nodes.dtype.names or ()):
            keep = [n for n in nodes.dtype.names if n != "missing_go_to_left"]
            new_dtype = np.dtype([(n, nodes.dtype[n]) for n in keep])
            new_nodes = np.zeros(nodes.shape, dtype=new_dtype)
            for name in keep:
                new_nodes[name] = nodes[name]
            state = {**state, "nodes": new_nodes}
        return (cls, args, state)

    buf = io.BytesIO()
    p = pickle.Pickler(buf)
    p.dispatch_table = {  # pylint: disable=c-extension-no-member
        _sklearn_tree.Tree: _old_tree_reducer
    }
    p.dump(model)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Backward-compatibility tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _sklearn_has_missing_go_to_left(),
    reason="sklearn < 1.3: missing_go_to_left not present; nothing to back-compat",
)
def test_old_pickle_fails_without_fix(learners_data):
    """Baseline: plain pickle.loads raises ValueError on old-format tree nodes.

    This test documents and verifies the root cause of issue #389 so that we
    have a failing baseline before applying the fix.
    """
    learner, _, _ = learners_data[0]  # NGBClassifier is sufficient
    old_bytes = _make_old_style_pickle_bytes(learner)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        f.write(old_bytes)
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="missing_go_to_left"):
            joblib.load(tmp_path)
    finally:
        os.unlink(tmp_path)


@pytest.mark.skipif(
    not _sklearn_has_missing_go_to_left(),
    reason="sklearn < 1.3: no backward-compat needed",
)
def test_backward_compat_load(learners_data):
    """load_ngboost_model transparently handles models saved with sklearn < 1.3.

    After applying the fix the loaded model must produce predictions identical
    to those of the original model.
    """
    for learner, data, preds in learners_data:
        old_bytes = _make_old_style_pickle_bytes(learner)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(old_bytes)
            tmp_path = f.name
        try:
            model = load_ngboost_model(tmp_path)
            new_preds = model.predict(data)
            assert (new_preds == preds).all()
            for iter_models in model.base_models:
                for estimator in iter_models:
                    assert isinstance(  # pylint: disable=c-extension-no-member
                        estimator.tree_, _sklearn_tree.Tree
                    )
        finally:
            os.unlink(tmp_path)
