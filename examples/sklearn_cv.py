from ngboost.distns import Bernoulli
from ngboost import NGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split


if __name__ == "__main__":

    X, y = load_breast_cancer(True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    param_grid = {
        'n_estimators': [20, 50],
        'minibatch_frac': [1.0, 0.5],
    }

    ngb = NGBClassifier(
        natural_gradient=True,
        verbose=False,
        Dist=Bernoulli
    )

    grid_search = GridSearchCV(ngb, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
