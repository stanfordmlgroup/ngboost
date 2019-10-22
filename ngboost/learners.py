from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge


class ConstantLearner(object):

    def fit(self, x, y):
        self.const = 1

    def predict(self, x):
        return self.const


def default_tree_learner(depth=3):
    return DecisionTreeRegressor(
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=depth,
        splitter='best')

def default_linear_learner(alpha=1):
    return Ridge(alpha = 1)

default_constant_learner = ConstantLearner
