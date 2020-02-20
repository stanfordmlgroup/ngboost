from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

default_tree_learner = DecisionTreeRegressor(
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    splitter="best",
    random_state=None,
)

default_linear_learner = Ridge(alpha=0.0, random_state=None)
