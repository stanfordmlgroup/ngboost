### NGBoost: Natural Gradient Boosting for Probabilistic Prediction

ngboost is a Python library that implements Natural Gradient Boosting, as described in ["NGBoost: Natural Gradient Boosting for Probabilistic Prediction"](https://stanfordmlgroup.github.io/projects/ngboost/). It is built on top of [Scikit-Learn](https://scikit-learn.org/stable/), and is designed to be scalable and modular with respect to choice of proper scoring rule, distribution, and base learners.



Installation:

```
pip3 install git+https://github.com/stanfordmlgroup/ngboost
```

Probabilistic regression example on the Boston housing dataset:

```python
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE
from ngboost.distns import Normal

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X, Y = load_boston(True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb = NGBoost(Base=default_tree_learner, Dist=Normal, Score=MLE(), natural_gradient=True,
              verbose=False)
ngb.fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

#test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
print('Test NLL', test_NLL)
```

Also, you can use NGBoost as Scikit-Learn API. This is an examle to use GridSearchCV.

```python
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

```