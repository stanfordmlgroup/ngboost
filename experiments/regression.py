from __future__ import print_function
import csv
import numpy as np
import pandas as pd
import pickle
import datetime
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from distns import HomoskedasticNormal
from torch.distributions import Normal, LogNormal
from distns import HomoskedasticNormal
from ngboost.ngboost import NGBoost, SurvNGBoost
from experiments.evaluation import *
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

class RegressionLogger(object):

    def __init__(self, args):
        self.name = args.dataset
        self.verbose = args.verbose
        self.r2s = []
        self.mses = []
        self.nlls = []
        self.calib_slopes = []

    def tick(self, forecast, y_test):
        y_pred = forecast.mean.detach().numpy()
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        y_test_tens = torch.tensor(y_test, dtype=torch.float32)
        nll = -forecast.log_prob(y_test_tens).mean().detach().numpy()
        pred, obs, slope, intercept = calibration_regression(forecast, y_test)
        self.r2s.append(r2)
        self.mses.append(mse)
        self.nlls.append(nll)
        self.calib_slopes.append(slope)
        if self.verbose:
            print("R2: %.4f\tMSE:%.4f\tNLL:%.4f\tSlope:%.4f" %
                  (r2, mse, nll, slope))

    def print_results(self):
        print("R2: %.4f +/- %.4f" % (np.mean(self.r2s), np.std(self.r2s)))
        print("MSE: %.4f +/- %.4f" % (np.mean(self.mses), np.std(self.mses)))
        print("NLL: %.4f +/- %.4f" % (np.mean(self.nlls), np.std(self.nlls)))
        print("Slope: %.4f +/- %.4f" % (np.mean(self.calib_slopes),
                                        np.std(self.calib_slopes)))

    def save(self):
        if self.verbose:
            self.print_results()
        time = datetime.datetime.now()
        outfile = open("results/regression/logs_%s_%s.pkl" %
            (self.name, time.strftime("%Y-%m-%d-%H:%M")), "wb")
        pickle.dump(self, outfile)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="housing")
    argparser.add_argument("--n_est", type=int, default=250)
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--score", type=str, default="CRPS")
    argparser.add_argument("--base", type=str, default="tree")
    argparser.add_argument("--n_reps", type=int, default=5)
    argparser.add_argument("--minibatch_frac", type=float, default=0.5)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    logger = RegressionLogger(args)
    data = dataset_name_to_loader[args.dataset]()
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values

    for rep in range(args.n_reps):

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
                    minibatch_frac=args.minibatch_frac,
                    nu_penalty=1e-5,
                    verbose=args.verbose)

        ngb.fit(X_train, y_train)
        forecast = ngb.pred_dist(X_test)
        logger.tick(forecast, y_test)

    logger.save()

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
