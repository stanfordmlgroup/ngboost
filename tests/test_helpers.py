import numpy as np

from ngboost.helpers import Y_from_censored


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
