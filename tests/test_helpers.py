import io
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import sklearn.tree._tree as _sklearn_tree  # pylint: disable=c-extension-no-member
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBSurvival
from ngboost.distns import Exponential
from ngboost.helpers import Y_from_censored, load_ngboost_model


def _sklearn_has_missing_go_to_left():
    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit([[0], [1]], [0, 1])
    return "missing_go_to_left" in (
        tree.tree_.__getstate__()["nodes"].dtype.names or ()
    )


def _make_old_style_pickle_bytes(model):
    def _old_tree_reducer(tree):
        cls, args, state = tree.__reduce__()
        nodes = state.get("nodes")
        if nodes is not None and "missing_go_to_left" in (nodes.dtype.names or ()):
            keep = [name for name in nodes.dtype.names if name != "missing_go_to_left"]
            new_dtype = np.dtype([(name, nodes.dtype[name]) for name in keep])
            new_nodes = np.zeros(nodes.shape, dtype=new_dtype)
            for name in keep:
                new_nodes[name] = nodes[name]
            state = {**state, "nodes": new_nodes}
        return (cls, args, state)

    buf = io.BytesIO()
    pickler = pickle.Pickler(buf)
    pickler.dispatch_table = {_sklearn_tree.Tree: _old_tree_reducer}
    pickler.dump(model)
    return buf.getvalue()


def test_Y_from_censored_with_default_event_vector():
    y = Y_from_censored(np.array([1.0, 2.0, 3.0]))
    assert y.dtype.names == ("Event", "Time")
    assert np.array_equal(y["Event"], np.array([True, True, True]))
    assert np.array_equal(y["Time"], np.array([1.0, 2.0, 3.0]))


def test_Y_from_censored_with_given_event_vector():
    y = Y_from_censored(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 1]))
    assert y.dtype.names == ("Event", "Time")
    assert np.array_equal(y["Event"], np.array([True, False, True]))
    assert np.array_equal(y["Time"], np.array([1.0, 2.0, 3.0]))


def test_Y_from_censored_preserves_preformatted_dtype():
    preformatted = np.array(
        [(True, 4.0), (False, 5.0)],
        dtype=[("Event", "?"), ("Time", "<f8")],
    )
    assert Y_from_censored(preformatted) is preformatted


def test_Y_from_censored_none_is_none():
    assert Y_from_censored(None) is None


@pytest.mark.skipif(
    not _sklearn_has_missing_go_to_left(),
    reason="sklearn < 1.3: no backward-compat needed",
)
def test_load_ngboost_model_handles_plain_sklearn_tree_pickle():
    tree = DecisionTreeRegressor(max_depth=2, random_state=0)
    tree.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    old_bytes = _make_old_style_pickle_bytes(tree)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        f.write(old_bytes)
        temp_path = Path(f.name)
    try:
        loaded = load_ngboost_model(temp_path)
        assert isinstance(loaded, DecisionTreeRegressor)
        assert "missing_go_to_left" in (
            loaded.tree_.__getstate__()["nodes"].dtype.names or ()
        )
    finally:
        temp_path.unlink()


@pytest.mark.parametrize("container", [list, np.asarray])
def test_Y_from_censored_accepts_array_like_times(container):
    y = Y_from_censored(container([1.0, 2.0, 3.0]), [1, 0, 1])
    np.testing.assert_array_equal(y["Time"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y["Event"], [True, False, True])


@pytest.mark.parametrize("events", [[0], [1, 0], [1, 0, 1, 0]])
def test_Y_from_censored_rejects_inconsistent_lengths(events):
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        Y_from_censored(np.array([1.0, 2.0, 3.0]), np.asarray(events))


def test_survival_fit_accepts_lists():
    x = np.arange(12).reshape(6, 2)
    times, events = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 0, 1, 0, 1, 1]
    kwargs = {
        "Dist": Exponential,
        "n_estimators": 2,
        "verbose": False,
        "random_state": 0,
    }
    array_model = NGBSurvival(**kwargs).fit(x, np.asarray(times), np.asarray(events))
    list_model = NGBSurvival(**kwargs).fit(x.tolist(), times, events)
    np.testing.assert_allclose(list_model.predict(x), array_model.predict(x))


@pytest.mark.parametrize("validation", [False, True])
def test_survival_fit_rejects_inconsistent_event_lengths(validation):
    x = np.arange(12).reshape(6, 2)
    times = np.arange(1.0, 7.0)
    events = np.ones(6)
    kwargs = {"X_val": x, "T_val": times, "E_val": np.array([1])} if validation else {}
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        NGBSurvival(n_estimators=1, verbose=False).fit(
            x, times, events if validation else np.array([1]), **kwargs
        )
