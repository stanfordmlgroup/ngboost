import numpy as np

class Base_Linear(object):
    def __init__(self, l2=0.01):
        self.l2 = l2
        return

    def fit(self, X, Y):
        m, n = X.shape
        self.theta = np.linalg.inv(X.T.dot(X) + self.l2 * np.eye(n).astype(np.float32)).dot(X.T.dot(Y))
        return self

    def predict(self, X):
        return X.dot(self.theta)

