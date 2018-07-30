from __future__ import print_function
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from distns import HomoskedasticNormal
from torch.distributions import Normal

from distns import HomoskedasticNormal
from ngboost.ngboost import NGBoost, SurvNGBoost
from experiments.evaluation import r2_score
from sklearn.metrics import mean_squared_error
from ngboost.scores import MLE_surv, CRPS_surv


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
    y = norm.rvs(size=m) * s + mu + 100
    return { 'data': X, 'target': y}

def run(data):
    X, y = data["data"], data["target"]
    C = np.zeros(len(y))

    X_train, X_test, y_train, y_test, C_train, C_test = train_test_split(X, y, C)
    # base_learner = lambda: DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3)
    base_learner = lambda: LinearRegression()

    ngb = SurvNGBoost(Base=base_learner,
                  Dist=DISTRIBUTION,
                  Score=CRPS_surv,
                  n_estimators=200,
                  learning_rate=0.1,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=False,
                  minibatch_frac=1.0,
                  nu_penalty=1e-5,
                  verbose=False)

    ngb.fit(X_train, y_train, C_train, X_test, y_test, C_test)
    y_pred = ngb.pred_mean(X_test)
    #print(r2_score(y_test, y_pred))
    print('===== CRPS =====')
    print('CRPS Train RMSE: %.3f' % mean_squared_error(y_train, ngb.pred_mean(X_train)))
    print(' CRPS Test RMSE: %.3f' % mean_squared_error(y_test, y_pred))
    # print('CRPS Train STDV: %.3f' % np.mean(ngb.pred_param(X_train)[1].exp().data.numpy()))
    # print(' CRPS Test STDV: %.3f' % np.mean(ngb.pred_param(X_test)[1].exp().data.numpy()))
    # print(np.mean(y_pred))



    ngb = SurvNGBoost(Base=base_learner,
                  Dist=DISTRIBUTION,
                  Score=MLE_surv,
                  n_estimators=200,
                  learning_rate=0.1,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=False,
                  minibatch_frac=1.0,
                  nu_penalty=1e-5,
                  verbose=False)

    ngb.fit(X_train, y_train, C_train, X_test, y_test, C_test)
    y_pred = ngb.pred_mean(X_test)

    print('===== MLE =====')
    print(' MLE Train RMSE: %.3f' % mean_squared_error(y_train, ngb.pred_mean(X_train)))
    print('  MLE Test RMSE: %.3f' % mean_squared_error(y_test, y_pred))
    # print(' MLE Train STDV: %.3f' % np.mean(ngb.pred_param(X_train)[1].exp().data.numpy()))
    # print('  MLE Test STDV: %.3f' % np.mean(ngb.pred_param(X_test)[1].exp().data.numpy()))

    return

if __name__ == "__main__":
    data = load_data()
    for i in range(3):
        print('Iter %d' % i)
        run(data)
