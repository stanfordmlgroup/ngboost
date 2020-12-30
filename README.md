# NGBoost: Natural Gradient Boosting for Probabilistic Prediction

![Python package](https://github.com/stanfordmlgroup/ngboost/workflows/Python%20package/badge.svg)
[![Github License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ngboost is a Python library that implements Natural Gradient Boosting, as described in ["NGBoost: Natural Gradient Boosting for Probabilistic Prediction"](https://stanfordmlgroup.github.io/projects/ngboost/). It is built on top of [Scikit-Learn](https://scikit-learn.org/stable/), and is designed to be scalable and modular with respect to choice of proper scoring rule, distribution, and base learner. A didactic introduction to the methodology underlying NGBoost is available in this [slide deck](https://drive.google.com/file/d/183BWFAdFms81MKy6hSku8qI97OwS_JH_/view?usp=sharing).

## Installation

```sh
via pip

pip install --upgrade ngboost

via conda-forge

conda install -c conda-forge ngboost
```

## Usage

Probabilistic regression example on the Boston housing dataset:

```python
from ngboost import NGBRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, Y = load_boston(True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test).mean()
print('Test NLL', test_NLL)
```

Details on available distributions, scoring rules, learners, tuning, and model interpretation are available in our [user guide](https://stanfordmlgroup.github.io/ngboost/intro.html), which also includes numerous usage examples and information on how to add new distributions or scores to NGBoost.

## License

[Apache License 2.0](https://github.com/stanfordmlgroup/ngboost/blob/master/LICENSE).

## Reference

Tony Duan, Anand Avati, Daisy Yi Ding, Khanh K. Thai, Sanjay Basu, Andrew Y. Ng, Alejandro Schuler. 2019.
NGBoost: Natural Gradient Boosting for Probabilistic Prediction.
[arXiv](https://arxiv.org/abs/1910.03225)
