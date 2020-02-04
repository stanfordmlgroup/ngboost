import numpy as np

def Y_from_censored(T,E=None):
    if T is None:
        return None
    if E is None:
    	E = np.ones_like(T)
    Y = np.empty(dtype=[('Event', np.bool), ('Time', np.float64)],
                 shape=T.shape[0])
    Y['Event'] = E.astype(bool)
    Y['Time'] = T.astype(float)
    return Y