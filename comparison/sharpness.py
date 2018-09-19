from __future__ import print_function
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from ngboost.distns import HomoskedasticNormal
from torch.distributions import Normal

from ngboost.ngboost import NGBoost
from experiments.evaluation import r2_score
from sklearn.metrics import mean_squared_error
from ngboost.scores import *
from ngboost.learners import *
from scipy.stats import norm


DISTRIBUTION = Normal

def load_data():
    m, n = 500, 25
    X = np.random.random((m, n)) * 2 - 1
    theta_mu = np.random.random(n) * 10 - 5
    theta_logs = np.random.random(n)
    mu = X.dot(theta_mu)
    logs = X.dot(theta_logs)
    s = np.exp(logs)
    print('Mean MU/STDV: %.3f, %.3f' % (np.mean(mu), np.mean(s)))
    y = norm.rvs(size=m) * s + mu + 100 + norm.rvs(size=m) * 2 # tune unexplained variance!
    return {'data': X, 'target': y}

def run(data):
    X, y = data["data"], data["target"]
    C = np.zeros(len(y))

    X_train, X_test, y_train, y_test, C_train, C_test = train_test_split(X, y, C)

    ngb = NGBoost(Base=default_linear_learner,
                  Dist=DISTRIBUTION,
                  Score=CRPS,
                  n_estimators=100,
                  learning_rate=0.1,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=False,
                  minibatch_frac=1.0,
                  normalize_inputs=True,
                  normalize_outputs=True,
                  tol=1e-4,
                  verbose=False)

    ngb.fit(X_train, y_train)
    y_pred = ngb.pred_dist(X_test).mean.detach().numpy()
    print('===== CRPS =====')
    print('CRPS Train RMSE: %.3f' % mean_squared_error(y_train, ngb.pred_dist(X_train).mean.detach().numpy()))
    print('CRPS Test RMSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('CRPS Train NLL: %.3f' % ngb.pred_dist(X_train).log_prob(torch.FloatTensor(y_train)).mean().detach().numpy())
    print('CRPS Test NLL: %.3f' % ngb.pred_dist(X_test).log_prob(torch.FloatTensor(y_test)).mean().detach().numpy())
    print('CRPS Train STDV: %.3f' % np.mean(ngb.pred_param(X_train)[1].exp().data.numpy()))
    print('CRPS Test STDV: %.3f' % np.mean(ngb.pred_param(X_test)[1].exp().data.numpy()))

    ngb = NGBoost(Base=default_linear_learner,
                  Dist=DISTRIBUTION,
                  Score=MLE,
                  n_estimators=100,
                  learning_rate=0.1,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=False,
                  minibatch_frac=1.0,
                  normalize_inputs=True,
                  normalize_outputs=True,
                  tol=1e-4,
                  verbose=False)

    ngb.fit(X_train, y_train)
    y_pred = ngb.pred_dist(X_test).mean.detach().numpy()

    print('===== MLE =====')
    print('MLE Train RMSE: %.3f' % mean_squared_error(y_train, ngb.pred_dist(X_train).mean.detach().numpy()))
    print('MLE Test RMSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MLE Train NLL: %.3f' % ngb.pred_dist(X_train).log_prob(torch.FloatTensor(y_train)).mean().detach().numpy())
    print('MLE Test NLL: %.3f' % ngb.pred_dist(X_test).log_prob(torch.FloatTensor(y_test)).mean().detach().numpy())
    print('MLE Train STDV: %.3f' % np.mean(ngb.pred_param(X_train)[1].exp().data.numpy()))
    print('MLE Test STDV: %.3f' % np.mean(ngb.pred_param(X_test)[1].exp().data.numpy()))

    return

if __name__ == "__main__":

    data = load_data()
    for i in range(3):
        print('Iter %d' % i)
        run(data)
