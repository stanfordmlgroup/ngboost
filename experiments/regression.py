from __future__ import print_function
import csv
import numpy as np
import itertools

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from distns import HomoskedasticNormal
from torch.distributions import Normal, LogNormal

from distns import HomoskedasticNormal
from ngboost.ngboost import NGBoost, SurvNGBoost
from experiments.evaluation import *
from sklearn.metrics import mean_squared_error
from ngboost.learners import *
from ngboost.scores import *

np.random.seed(123)


if __name__ == "__main__":

    data = load_boston()
    X, y = data["data"], data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("Heteroskedastic")

    results = []

    for (n_estimators, lr, score, base) in itertools.product(
            (50,, 100, 150, 200, 250, 300),
            (0.005,, 0.01, 0.025, 0.05, 0.1),
            (MLE, CRPS),
            (default_tree_learner, default_linear_learner)):

        ngb = NGBoost(Base=default_tree_learner,
                      Dist=Normal,
                      Score=score,
                      n_estimators=n_estimators,
                      learning_rate=lr,
                      natural_gradient=True,
                      second_order=True,
                      quadrant_search=True,
                      minibatch_frac=0.5,
                      nu_penalty=1e-5,
                      verbose=True)

        ngb.fit(X_train, y_train)

        y_pred = ngb.pred_mean(X_test)
        forecast = ngb.pred_dist(X_test)
        print("R2: %.4f" % r2_score(y_test, y_pred))
        print("MSE: %.4f" % mean_squared_error(y_test, y_pred))

        pred, obs, slope, intercept = calibration_regression(forecast, y_test)
        print("Val slope: %.4f | intercept: %.4f" % (slope, intercept))

        forecast = ngb.pred_dist(X_train)
        _, _, tslope, tintercept = calibration_regression(forecast, y_train)
        print("Train slope: %.4f | intercept: %.4f" % (tslope, tintercept))

        params = (n_estimators, lr,
                  "MLE" if score == MLE else "CRPS",
                  "linear" if base == default_linear_learner else "tree")
        print(params)

        row = params + (r2_score(y_test, y_pred),
                        mean_squared_error(y_test, y_pred),
                        slope, intercept, tslope, tintercept)
        results.append(row)

    with open("./results/regression_experiment.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_learners", "lr", "score", "base", "r2", "mse",
                         "val_slope", "val_int", "tr_slope", "tr_int"])
        for row in results:
            writer.writerow(row)

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
