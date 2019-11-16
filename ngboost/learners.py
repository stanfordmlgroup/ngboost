from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge


def default_tree_learner(depth=3, random_state=None):
    return DecisionTreeRegressor(
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=depth,
        splitter='best',
        random_state=random_state)


def default_linear_learner(alpha=1):
    return Ridge(alpha=alpha, random_state=None)
