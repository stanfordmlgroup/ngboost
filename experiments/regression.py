from __future__ import print_function
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from distns import HomoskedasticNormal
from torch.distributions import Normal

from distns import HomoskedasticNormal
from ngboost.ngboost import NGBoost, SurvNGBoost
from experiments.evaluation import r2_score, calibration_regression, plot_calibration_curve
from sklearn.metrics import mean_squared_error
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE, CRPS

np.random.seed(123)


if __name__ == "__main__":

    data = load_boston()
    X, y = data["data"], data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Heteroskedastic")
    ngb = NGBoost(Base=default_tree_learner,
                  Dist=Normal,
                  Score=CRPS,
                  n_estimators=200,
                  learning_rate=0.1,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=True,
                  minibatch_frac=1.0,
                  nu_penalty=1e-5,
                  verbose=True)

    ngb.fit(X_train, y_train, X_test, y_test)
    y_pred = ngb.pred_mean(X_test)
    forecast = ngb.pred_dist(X_test)
    print("R2: %.4f" % r2_score(y_test, y_pred))
    print("MSE: %.4f" % mean_squared_error(y_test, y_pred))

    pred, obs, slope, intercept = calibration_regression(forecast, y_test)
    # plot_calibration_curve(pred, obs)
    ngb.write_to_disk("./ngb_ex.pkl")

    # print("Homoskedastic")
    # ngb = NGBoost(Base=base_learner,
    #               Dist=HomoskedasticNormal,
    #               Score=CRPS,
    #               n_estimators=400,
    #               learning_rate=0.1,
    #               natural_gradient=True,
    #               second_order=True,
    #               quadrant_search=False,
    #               minibatch_frac=1.0,
    #               nu_penalty=1e-5,
    #               verbose=False)
    #
    # ngb.fit(X_train, y_train, X_test, y_test)
    # y_pred = ngb.pred_mean(X_test)
    # print("R2: %.4f" % r2_score(y_test, y_pred))
    # print("MSE: %.4f" % mean_squared_error(y_test, y_pred))
    #
    # print("Scikit-Learn GBM")
    # gbr = GradientBoostingRegressor()
    # gbr.fit(X_train, y_train)
    # print("R2: %.4f" % r2_score(y_test, gbr.predict(X_test)))
    # print("MSE: %.4f" % mean_squared_error(y_test, gbr.predict(X_test)))
