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
from torch.distributions import Normal, LogNormal
from ngboost.ngboost import NGBoost, SurvNGBoost
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from experiments.evaluation import *

np.random.seed(123)

dataset_name_to_loader = {
    "housing": lambda: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, delim_whitespace=True),
    "concrete": lambda: pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"),
    "wine": lambda: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=";"),
    "kin8nm": lambda: pd.read_csv("data/uci/kin8nm.csv"),
    "naval": lambda: pd.read_csv("data/uci/naval-propulsion.txt", delim_whitespace=True, header=None).iloc[:,:-1],
    "power": lambda: pd.read_excel("data/uci/power-plant.xlsx"),
    "energy": lambda: pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx").iloc[:,:-1],
    "protein": lambda: pd.read_csv("data/uci/protein.csv")[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'RMSD']],
    "yacht": lambda: pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data", header=None, delim_whitespace=True),
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
        self.args = args
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

    def to_row(self):
        return pd.DataFrame({
            "dataset": [self.args.dataset],
            "score": [self.args.score],
            "rmse_mean": [np.mean(np.sqrt(self.mses))],
            "rmse_sd": [np.std(np.sqrt(self.mses))],
            "nll_mean": [np.mean(self.nlls)],
            "nll_sd": [np.std(self.nlls)],
            "r2s_mean": [np.mean(self.r2s)],
            "r2s_sd": [np.std(self.r2s)],
            "slope_mean": [np.mean(self.calib_slopes)],
            "slope_sd": [np.std(self.calib_slopes)],
        })

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
            (self.args.dataset, self.args.score), "wb")
        pickle.dump(self, outfile)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="housing")
    argparser.add_argument("--n_est", type=int, default=350)
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--score", type=str, default="CRPS")
    argparser.add_argument("--base", type=str, default="tree")
    argparser.add_argument("--n_reps", type=int, default=5)
    argparser.add_argument("--minibatch_frac", type=float, default=None)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    # load dataset -- use last column as label
    data = dataset_name_to_loader[args.dataset]()
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values

    # set default minibatch fraction based on dataset size
    if not args.minibatch_frac:
        args.minibatch_frac = min(0.8, 5000 / len(X))

    logger = RegressionLogger(args)

    for rep in range(args.n_reps):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

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
                      normalize_inputs=True,
                      normalize_outputs=True,
                      verbose=args.verbose)

        ngb.fit(X_train, y_train)
        forecast = ngb.pred_dist(X_test)
        logger.tick(forecast, y_test)

    logger.save()
