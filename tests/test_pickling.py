import io
import os
import pickle
import tempfile

import joblib
import numpy as np
import pytest
import sklearn.tree._tree as _sklearn_tree  # pylint: disable=c-extension-no-member
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival, load_ngboost_model
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
