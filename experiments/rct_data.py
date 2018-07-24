from __future__ import division, print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from torch.distributions import LogNormal, Exponential

from ngboost import SurvNGBoost, CRPS_surv, MLE_surv
from experiments.evaluation import calculate_concordance_naive
from distns import HomoskedasticLogNormal


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
    sb = SurvNGBoost(Base = lambda : DecisionTreeRegressor(criterion="friedman_mse"),
                     Dist = HomoskedasticLogNormal,
                     Score = MLE_surv,
                     n_estimators = 20,
                     learning_rate = 0.1,
                     natural_gradient = True,
                     second_order = True,
                     quadrant_search = False,
                     minibatch_frac = 0.5,
                     nu_penalty=1e-5)
    sprint["X"] = np.c_[sprint["X"], sprint["w"]]
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        sprint["X"], sprint["t"] / 365.25, sprint["y"],
        test_size = 0.2)
    sb.fit(X_train, T_train, 1 - Y_train, X_test, T_test, 1 - Y_test)

    # truth = sprint["t"] / 365.25
    # preds = sb.pred_dist(sprint["X"])

    # pred_means = preds.mean.detach().numpy()
    print(calculate_concordance_naive(sb.pred_mean(X_train), T_train, 1 - Y_train))
    print(calculate_concordance_naive(sb.pred_mean(X_test), T_test, 1 - Y_test))

