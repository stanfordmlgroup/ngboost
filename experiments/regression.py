from __future__ import print_function
import csv
import numpy as np
import pandas as pd
import itertools
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from distns import HomoskedasticNormal
from torch.distributions import Normal, LogNormal

from distns import HomoskedasticNormal
from ngboost.ngboost import NGBoost, SurvNGBoost
from experiments.evaluation import *
from sklearn.metrics import mean_squared_error
from ngboost import *

np.random.seed(123)

dataset_name_to_loader = {
    "housing": lambda: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, delim_whitespace=True),
    "wine": lambda: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=";"),
    "concrete": lambda: pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"),
    "kin8nm": lambda: pd.read_csv("data/uci/kin8nm.csv"),
}

base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
}

score_name_to_score = {
    "MLE": MLE,
    "CRPS": CRPS,
}


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="housing")
    argparser.add_argument("--n_est", type=int, default=250)
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--score", type=str, default="CRPS")
    argparser.add_argument("--base", type=str, default="tree")
    args = argparser.parse_args()

    data = dataset_name_to_loader[args.dataset]()
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    results = []

    ngb = NGBoost(Base=base_name_to_learner[args.base],
                  Dist=Normal,
                  Score=score_name_to_score[args.score],
                  n_estimators=args.n_est,
                  learning_rate=args.lr,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=True,
                  minibatch_frac=0.5,
                  nu_penalty=1e-5,
                  verbose=True)

    ngb.fit(X_train, y_train)

    y_pred = ngb.pred_mean(X_test)
    y_test_tens = torch.tensor(y_test, dtype=torch.float32)
    forecast = ngb.pred_dist(X_test)

    print("R2: %.4f" % r2_score(y_test, y_pred))
    print("MSE: %.4f" % mean_squared_error(y_test, y_pred))
    print("NLL: %.4f" % -forecast.log_prob(y_test_tens).mean().detach().numpy())

    pred, obs, slope, intercept = calibration_regression(forecast, y_test)
    print("Val slope: %.4f | intercept: %.4f" % (slope, intercept))

    forecast = ngb.pred_dist(X_train)
    _, _, tslope, tintercept = calibration_regression(forecast, y_train)
    print("Train slope: %.4f | intercept: %.4f" % (tslope, tintercept))

    # with open("./results/regression_experiment.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["n_learners", "lr", "score", "base", "r2", "mse",
    #                      "val_slope", "val_int", "tr_slope", "tr_int"])
    #     for row in results:
    #         writer.writerow(row)

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
