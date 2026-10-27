import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from ngboost.helpers import Y_from_censored, load_ngboost_model
from tests.test_pickling import (
    _make_old_style_pickle_bytes,
    _sklearn_has_missing_go_to_left,
)


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
