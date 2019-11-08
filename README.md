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

Also, you can use NGBoost as Scikit-Learn API.

```python
from ngboost import NGBRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    X, y = load_boston(True)
    X_train, X_test, y_train, y_train = train_test_split(X, y, test_size=0.2)

    ngb = NGBRegressor()
    ngb.fit(X_train, y_train)
    y_pred = ngb.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE on test data: {mse}')

```
