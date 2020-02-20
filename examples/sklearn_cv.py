from ngboost.distns import k_categorical
from ngboost import NGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge


if __name__ == "__main__":
    # An example where the base learner is also searched over (this is how you would vary tree depth):

    X, Y = load_breast_cancer(True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    b1 = DecisionTreeRegressor(criterion="friedman_mse", max_depth=2)
    b2 = DecisionTreeRegressor(criterion="friedman_mse", max_depth=4)
    b3 = Ridge(alpha=0.0)

    param_grid = {
        "n_estimators": [20, 50],
        "minibatch_frac": [1.0, 0.5],
        "Base": [b1, b2],
    }

    ngb = NGBClassifier(natural_gradient=True, verbose=False, Dist=k_categorical(2))

    grid_search = GridSearchCV(ngb, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    print(grid_search.best_params_)
