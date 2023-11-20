# NGBoost: Natural Gradient Boosting for Probabilistic Prediction

<h4 align="center">

![Python package](https://github.com/stanfordmlgroup/ngboost/workflows/Python%20package/badge.svg)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/stanfordmlgroup/ngboost?label=Repo+Size)](https://github.com/stanfordmlgroup/ngboost/graphs/contributors)
[![Github License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/ngboost?logo=pypi&logoColor=white)](https://pypi.org/project/ngboost)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ngboost?logo=icloud&logoColor=white)](https://pypistats.org/packages/ngboost)

</h4>

ngboost is a Python library that implements Natural Gradient Boosting, as described in ["NGBoost: Natural Gradient Boosting for Probabilistic Prediction"](https://stanfordmlgroup.github.io/projects/ngboost/). It is built on top of [Scikit-Learn](https://scikit-learn.org/stable/), and is designed to be scalable and modular with respect to choice of proper scoring rule, distribution, and base learner. A didactic introduction to the methodology underlying NGBoost is available in this [slide deck](https://docs.google.com/presentation/d/1Tn23Su0ygR6z11jy3xVNiLGv0ggiUQue/edit?usp=share_link&ouid=102290675300480810195&rtpof=true&sd=true).

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Load Boston housing dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
Y = raw_df.values[1::2, 2]

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
