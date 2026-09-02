import importlib
import json
import sys
import threading
import types as _types

import joblib
import numpy as np
import sklearn.tree._tree as _sklearn_tree  # pylint: disable=c-extension-no-member
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_array

from ngboost.manifold import manifold

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
# JSON inference serialization helpers (issue #392)
# ---------------------------------------------------------------------------

_JSON_FORMAT = "ngboost-json-inference"
_JSON_VERSION = 1


def _qualified_name(obj):
    return f"{obj.__module__}.{obj.__qualname__}"


def _ensure_importable_class(obj, description):
    name = _qualified_name(obj)
    if "<locals>" in name:
        raise TypeError(
            "JSON inference serialization does not currently support "
            f"dynamic {description} classes: {name}."
        )
    try:
        if _import_qualified_name(name) is not obj:
            raise TypeError
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise TypeError(
            "JSON inference serialization requires importable "
            f"{description} classes, got {name}."
        ) from exc
    return name


def _import_qualified_name(name):
    module_name, _, attr_name = name.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid qualified name: {name!r}")
    module = importlib.import_module(module_name)
    obj = module
    for attr in attr_name.split("."):
        obj = getattr(obj, attr)
    return obj


def _encode_distribution(dist):
    if getattr(dist, "__name__", None) == "Categorical":
        return {"kind": "categorical", "K": dist.n_params + 1}
    return {"kind": "qualified", "name": _ensure_importable_class(dist, "distribution")}


def _decode_distribution(payload):
    if payload["kind"] == "categorical":
        from ngboost.distns import (  # pylint: disable=import-outside-toplevel
            k_categorical,
        )

        return k_categorical(payload["K"])
    if payload["kind"] == "qualified":
        return _import_qualified_name(payload["name"])
    raise ValueError(f"Unknown distribution payload: {payload['kind']!r}")


def _encode_json_value(value):
    if isinstance(value, np.ndarray):
        payload = {
            "__ndarray__": True,
            "shape": value.shape,
        }
        if value.dtype.names:
            payload["dtype_struct"] = {
                "names": list(value.dtype.names),
                "formats": [
                    value.dtype.fields[name][0].str for name in value.dtype.names
                ],
                "offsets": [value.dtype.fields[name][1] for name in value.dtype.names],
                "itemsize": value.dtype.itemsize,
            }
            payload["fields"] = {
                name: value[name].tolist() for name in value.dtype.names
            }
        else:
            payload["dtype"] = str(value.dtype)
            payload["data"] = value.tolist()
        return payload
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": True, "items": [_encode_json_value(v) for v in value]}
    if isinstance(value, list):
        return [_encode_json_value(v) for v in value]
    if isinstance(value, dict):
        return {key: _encode_json_value(val) for key, val in value.items()}
    return value


def _decode_json_value(value):
    if isinstance(value, list):
        return [_decode_json_value(v) for v in value]
    if not isinstance(value, dict):
        return value
    if value.get("__ndarray__"):
        dtype = np.dtype(
            {
                "names": value["dtype_struct"]["names"],
                "formats": value["dtype_struct"]["formats"],
                "offsets": value["dtype_struct"]["offsets"],
                "itemsize": value["dtype_struct"]["itemsize"],
            }
            if "dtype_struct" in value
            else value["dtype"]
        )
        if "dtype_struct" in value:
            array = np.zeros(value["shape"], dtype=dtype)
            for name, field_values in value["fields"].items():
                array[name] = field_values
            return array
        array = np.array(value["data"], dtype=dtype)
        return array.reshape(value["shape"])
    if value.get("__tuple__"):
        return tuple(_decode_json_value(v) for v in value["items"])
    return {key: _decode_json_value(val) for key, val in value.items()}


def _serialize_decision_tree(estimator):
    if not isinstance(estimator, DecisionTreeRegressor):
        raise TypeError(
            "JSON inference serialization currently supports fitted "
            "DecisionTreeRegressor base learners only."
        )
    if not hasattr(estimator, "tree_"):
        raise ValueError("Cannot serialize an unfitted DecisionTreeRegressor.")

    attrs = {}
    for name in (
        "n_features_in_",
        "n_outputs_",
        "max_features_",
        "feature_names_in_",
    ):
        if hasattr(estimator, name):
            attrs[name] = _encode_json_value(getattr(estimator, name))

    return {
        "class": _qualified_name(estimator.__class__),
        "params": _encode_json_value(estimator.get_params(deep=False)),
        "attrs": attrs,
        "tree": {
            "n_features": estimator.tree_.n_features,
            "n_classes": _encode_json_value(estimator.tree_.n_classes),
            "n_outputs": estimator.tree_.n_outputs,
            "state": _encode_json_value(estimator.tree_.__getstate__()),
        },
    }


def _deserialize_decision_tree(payload):
    estimator_class = _import_qualified_name(payload["class"])
    if estimator_class is not DecisionTreeRegressor:
        raise TypeError(
            "Only sklearn.tree.DecisionTreeRegressor JSON payloads are supported."
        )

    estimator = estimator_class(**_decode_json_value(payload["params"]))
    for name, value in payload["attrs"].items():
        setattr(estimator, name, _decode_json_value(value))

    tree_payload = payload["tree"]
    n_classes = _decode_json_value(tree_payload["n_classes"])
    if isinstance(n_classes, int):
        n_classes = np.array([n_classes], dtype=np.intp)
    else:
        n_classes = np.asarray(n_classes, dtype=np.intp)
    tree = _sklearn_tree.Tree(  # pylint: disable=c-extension-no-member
        tree_payload["n_features"], n_classes, tree_payload["n_outputs"]
    )
    tree.__setstate__(_decode_json_value(tree_payload["state"]))
    estimator.tree_ = tree
    return estimator


def _serialize_base_config(base):
    if isinstance(base, (list, tuple)):
        return {
            "kind": "sequence",
            "sequence_type": type(base).__name__,
            "items": [_serialize_base_config(item) for item in base],
        }
    if isinstance(base, DecisionTreeRegressor):
        return {
            "kind": "decision_tree_regressor",
            "params": _encode_json_value(base.get_params(deep=False)),
        }
    raise TypeError(
        "JSON inference serialization currently supports DecisionTreeRegressor "
        "base learner configuration only."
    )


def _deserialize_base_config(payload):
    if payload["kind"] == "sequence":
        items = [_deserialize_base_config(item) for item in payload["items"]]
        return tuple(items) if payload.get("sequence_type") == "tuple" else items
    if payload["kind"] == "decision_tree_regressor":
        return DecisionTreeRegressor(**_decode_json_value(payload["params"]))
    raise ValueError(f"Unknown base learner payload: {payload['kind']!r}")


def save_ngboost_model_json(model, filepath):
    """Save a fitted NGBoost model to a JSON file for inference.

    The JSON payload stores only the fitted state needed by ``predict``,
    ``pred_dist``, and classifier ``predict_proba``. It avoids pickle while
    preserving exact sklearn decision-tree base learner state.
    """
    if not getattr(model, "base_models", None):
        raise ValueError("Cannot JSON serialize an unfitted NGBoost model.")

    payload = {
        "format": _JSON_FORMAT,
        "version": _JSON_VERSION,
        "model_class": _qualified_name(model.__class__),
        "dist": _encode_distribution(model.Dist),
        "score": _qualified_name(model.Score),
        "base": _serialize_base_config(model.Base),
        "params": {
            "natural_gradient": model.natural_gradient,
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "minibatch_frac": model.minibatch_frac,
            "col_sample": model.col_sample,
            "verbose": model.verbose,
            "verbose_eval": model.verbose_eval,
            "tol": model.tol,
            "validation_fraction": model.validation_fraction,
            "early_stopping_rounds": model.early_stopping_rounds,
        },
        "state": {
            "init_params": _encode_json_value(model.init_params),
            "n_features": model.n_features,
            "best_val_loss_itr": model.best_val_loss_itr,
            "multi_output": model.multi_output,
            "estimator_type": getattr(model, "_estimator_type", None),
            "scalings": _encode_json_value(model.scalings),
            "col_idxs": _encode_json_value(model.col_idxs),
            "evals_result": _encode_json_value(getattr(model, "evals_result", {})),
            "classes": _encode_json_value(getattr(model, "classes_", None)),
            "base_models": [
                [_serialize_decision_tree(estimator) for estimator in iter_models]
                for iter_models in model.base_models
            ],
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, separators=(",", ":"))


def load_ngboost_model_json(filepath):
    """Load a fitted NGBoost model saved by ``save_ngboost_model_json``."""
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("format") != _JSON_FORMAT or payload.get("version") != _JSON_VERSION:
        raise ValueError("Unsupported NGBoost JSON model format.")

    model_class = _import_qualified_name(payload["model_class"])
    model = model_class.__new__(model_class)
    model.Dist = _decode_distribution(payload["dist"])
    model.Score = _import_qualified_name(payload["score"])
    model.Base = _deserialize_base_config(payload["base"])
    model.Manifold = manifold(model.Score, model.Dist)
    model.random_state = None

    for name, value in payload["params"].items():
        setattr(model, name, value)

    state = payload["state"]
    model.init_params = _decode_json_value(state["init_params"])
    model.n_features = state["n_features"]
    model.best_val_loss_itr = state["best_val_loss_itr"]
    model.multi_output = state["multi_output"]
    model._estimator_type = state["estimator_type"]  # pylint: disable=protected-access
    model.scalings = _decode_json_value(state["scalings"])
    model.col_idxs = _decode_json_value(state["col_idxs"])
    model.evals_result = _decode_json_value(state["evals_result"])
    classes = _decode_json_value(state.get("classes"))
    if classes is not None:
        model.classes_ = classes
        model._le = LabelEncoder()  # pylint: disable=protected-access
        model._le.classes_ = classes  # pylint: disable=protected-access
    model.base_models = [
        [_deserialize_decision_tree(estimator) for estimator in iter_models]
        for iter_models in state["base_models"]
    ]
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
