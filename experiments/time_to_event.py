from __future__ import division, print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from torch.distributions import LogNormal, Exponential, Weibull

from ngboost import *
from experiments.evaluation import *
from ngboost.distns import HomoskedasticLogNormal


def load_data(dataset):
    """
    Loads training, validation and testing data from sprint or accord.
    """
    if dataset == "sprint":
        df = pd.read_csv("data/sprint/sprint_cut.csv")
        df["diabetes"] = np.zeros(len(df))
    elif dataset == "accord":
        df = pd.read_csv("data/accord/accord_cut.csv")
        df["diabetes"] = np.ones(len(df))
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    y = np.array(df["cvd"]).astype(np.int32)
    t = np.array(df["t_cvds"]).astype(np.int32)
    w = np.array(df["INTENSIVE"]).astype(np.int32)

    del df["Unnamed: 0"]
    del df["cvd"]
    del df["t_cvds"]
    del df["INTENSIVE"]
    cols = df.columns
    X = df.astype(np.float32).as_matrix()

    return {
        "X": X,
        "w": w,
        "y": y,
        "t": t,
        "cols": cols
    }

if __name__ == "__main__":

    sprint = load_data("sprint")
    sb = SurvNGBoost(Base = default_tree_learner,
                     Dist = LogNormal,
                     Score = CRPS_surv,
                     n_estimators = 50,
                     learning_rate = 0.05,
                     natural_gradient = True,
                     second_order = True,
                     quadrant_search = False,
                     normalize_inputs=True,
                     normalize_outputs=False,
                     minibatch_frac = 0.2,
                     nu_penalty=1e-5)
    sprint["X"] = np.c_[sprint["X"], sprint["w"]]
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        sprint["X"], sprint["t"] / 365.25, sprint["y"], test_size = 0.2)
    sb.fit(X_train, T_train, 1 - Y_train)

    # truth = sprint["t"] / 365.25
    preds = sb.pred_dist(X_test)

    c_stat = calculate_concordance_naive(preds.mean, T_test, 1 - Y_test)
    pred, obs, slope, intercept = calibration_time_to_event(preds, T_test, 1 - Y_test)
    print("Censorship rate:", 1-np.mean(sprint["y"]))
    print("True mean:", T_test.mean())
    print("Pred mean:", preds.mean.mean().detach().numpy())
    print("Calibration slope: %.4f, intercept: %.4f" % (slope, intercept))
    print("Predicted: %s\nObserved:%s" % (pred, obs))

