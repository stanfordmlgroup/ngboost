import numpy as np


class Base_Linear(object):
    def __init__(self, l2=0.01):
        self.l2 = l2
        return

    def fit(self, X, Y):
        m, n = X.shape
        hmask = np.zeros([1, n]).astype(np.float32)
        for _ in range(int(n / 2)):
            hmask[0, np.random.choice(n)] = 1
        vmask = np.zeros([m, 1]).astype(np.float32)
        for _ in range(int(m / 2)):
            vmask[np.random.choice(m), 0] = 1
        self.mask = vmask * hmask
        self.theta = np.linalg.pinv(X * self.mask).dot(Y)
        self.Y = Y
        return self

    def predict(self, X):
        #return self.Y
        return (X * self.mask).dot(self.theta)

