from ngboost import NGBRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split


if __name__ == "__main__":
    X, y = load_boston(True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    param_grid = {
        'n_estimators': [200, 500],
        'minibatch_frac': [1.0, 0.5],
    }

    ngb = NGBRegressor(
        natural_gradient=True,
        verbose=False,
    )
    grid_search = GridSearchCV(ngb, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
