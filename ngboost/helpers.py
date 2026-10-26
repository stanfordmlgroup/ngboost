import sys
import threading
import types as _types

import joblib
import numpy as np
import sklearn.tree._tree as _sklearn_tree  # pylint: disable=c-extension-no-member
from sklearn.utils import check_array

# ---------------------------------------------------------------------------
# Backward compatibility helpers (issue #389)
# ---------------------------------------------------------------------------

_TREE_MODULE_SWAP_LOCK = threading.RLock()


class _CompatTree(_sklearn_tree.Tree):  # pylint: disable=too-few-public-methods
    """Transient subclass of sklearn's Tree used only during loading.

    Overrides ``__setstate__`` to transparently inject the ``missing_go_to_left``
    field that was added in scikit-learn 1.3, so that models pickled with older
    sklearn can be loaded without raising ``ValueError``.
    """

    def __setstate__(self, state):
        nodes = state.get("nodes")
        if (
            nodes is not None
            and nodes.dtype.names is not None
            and "missing_go_to_left" not in nodes.dtype.names
        ):
            new_dtype = np.dtype(nodes.dtype.descr + [("missing_go_to_left", "u1")])
            new_nodes = np.zeros(nodes.shape, dtype=new_dtype)
            for name in nodes.dtype.names:
                new_nodes[name] = nodes[name]
            new_nodes["missing_go_to_left"] = 1  # default: send missing values left
            state = {**state, "nodes": new_nodes}
        super().__setstate__(state)


def _make_compat_tree_module():
    """Return a drop-in replacement for sklearn.tree._tree with Tree → _CompatTree."""
    mod = _types.ModuleType("sklearn.tree._tree")
    for _attr in dir(_sklearn_tree):
        setattr(mod, _attr, getattr(_sklearn_tree, _attr))
    mod.Tree = _CompatTree
    return mod


def _to_proper_tree(compat_tree):
    """Re-wrap a _CompatTree as a standard sklearn Tree (same state, proper type)."""
    state = compat_tree.__getstate__()
    n_classes = compat_tree.n_classes
    if isinstance(n_classes, int):
        n_classes = np.array([n_classes], dtype=np.intp)
    proper = _sklearn_tree.Tree(  # pylint: disable=c-extension-no-member
        compat_tree.n_features, n_classes.copy(), compat_tree.n_outputs
    )
    proper.__setstate__(state)
    return proper


def _fix_compat_trees(model):
    """Replace every _CompatTree in model.base_models with a proper sklearn Tree."""
    if not hasattr(model, "base_models"):
        return
    for iter_models in model.base_models:
        for estimator in iter_models:
            if hasattr(estimator, "tree_") and isinstance(estimator.tree_, _CompatTree):
                estimator.tree_ = _to_proper_tree(estimator.tree_)


def load_ngboost_model(filepath):
    """Load an NGBoost model with backward compatibility for older scikit-learn versions.

    Scikit-learn 1.3 added a ``missing_go_to_left`` field to the internal node
    structure of decision trees.  Models trained and saved with scikit-learn < 1.3
    do not contain this field, so loading them under scikit-learn >= 1.3 raises::

        ValueError: node array from the pickle has an incompatible dtype

    This function handles the incompatibility transparently by temporarily replacing
    ``sklearn.tree._tree.Tree`` in ``sys.modules`` with a compatible subclass that
    injects the missing field during ``__setstate__``.  After loading, all trees are
    converted back to standard sklearn ``Tree`` objects so that subsequent use and
    re-serialisation behave normally.

    Args:
        filepath: Path to the saved model file (joblib or pickle format).

    Returns:
        The loaded NGBoost model.

    Example::

        from ngboost import load_ngboost_model
        model = load_ngboost_model("my_old_model.pkl")
        preds = model.predict(X)
    """
    with _TREE_MODULE_SWAP_LOCK:
        compat_module = _make_compat_tree_module()
        original_module = sys.modules.get("sklearn.tree._tree")
        sys.modules["sklearn.tree._tree"] = compat_module
        try:
            model = joblib.load(filepath)
        finally:
            if original_module is None:
                sys.modules.pop("sklearn.tree._tree", None)
            else:
                sys.modules["sklearn.tree._tree"] = original_module

    _fix_compat_trees(model)
    return model


# ---------------------------------------------------------------------------


def Y_from_censored(T, E=None):
    if T is None:
        return None
    if T.dtype == [
        ("Event", "?"),
        ("Time", "<f8"),
    ]:  # already processed. Necessary for when d_score() calls score() as in LogNormalCRPScore
        return T
    T = check_array(T, ensure_2d=False)
    T = T.reshape(T.shape[0])
    if E is None:
        E = np.ones_like(T)
    else:
        E = check_array(E, ensure_2d=False)
        E = E.reshape(E.shape[0])
    Y = np.empty(dtype=[("Event", np.bool_), ("Time", np.float64)], shape=T.shape[0])
    Y["Event"] = E.astype(np.bool_)
    Y["Time"] = T.astype(np.float64)
    return Y
