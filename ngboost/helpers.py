import numpy as np


def Y_from_censored(T, E=None):
    if T is None:
        return None
    if T.dtype == [
        ("Event", "?"),
        ("Time", "<f8"),
    ]:  # already processed. Necessary for when d_score() calls score() as in LogNormalCRPScore
        return T
    if E is None:
        E = np.ones_like(T)
    Y = np.empty(dtype=[("Event", np.bool), ("Time", np.float64)], shape=T.shape[0])
    Y["Event"] = E.astype(np.bool)
    Y["Time"] = T.astype(np.float64)
    return Y
