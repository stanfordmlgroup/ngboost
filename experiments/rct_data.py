from __future__ import division, print_function

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from torch.distributions import LogNormal

from ngboost import SurvNGBoost, CRPS_surv
from experiments.evaluation import calculate_concordance_naive


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
    sb = SurvNGBoost(Base = lambda : DecisionTreeRegressor(criterion='mse'),
                     Dist = LogNormal,
                     Score = CRPS_surv,
                     n_estimators = 50,
                     learning_rate = 0.1,
                     natural_gradient = True,
                     second_order = True,
                     quadrant_search = False,
                     minibatch_frac = 1.0,
                     nu_penalty=1e-5)
    sb.fit(sprint["X"], sprint["t"] / 365.25, 1-sprint["y"])

    truth = sprint["t"] / 365.25
    preds = sb.pred_dist(sprint["X"])

    pred_means = preds.mean.detach().numpy()
    print(calculate_concordance_naive(pred_means, sprint["t"] / 365.25, 1 - sprint["y"]))

