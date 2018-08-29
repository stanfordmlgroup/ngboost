from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


default_tree_learner = lambda: DecisionTreeRegressor(
    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_depth=3)


default_linear_learner = lambda: LinearRegression()
