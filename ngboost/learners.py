from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from scipy.interpolate import UnivariateSpline


class ConstantLearner(object):

    def fit(self, x, y):
        self.const = 1

    def predict(self, x):
        return self.const


tree_learner_with_depth = lambda d: DecisionTreeRegressor(
    criterion='friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=d,
    splitter='best')

default_tree_learner = lambda: tree_learner_with_depth(3)
# default_linear_learner = LinearRegression
default_linear_learner = lambda: Ridge(alpha = 1)
default_constant_learner = ConstantLearner
