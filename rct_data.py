from __future__ import division, print_function

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.distributions import Exponential, LogNormal
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from survboost import SurvBoost
from tqdm import tqdm
from evaluation import calculate_concordance_naive


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
    sb = SurvBoost(learning_rate = 1, n_estimators = 500, 
                   Base = DecisionTreeRegressor,
                   Dist = LogNormal, 
                   minibatch_frac = 0.25)
    sb.fit(sprint["X"], sprint["t"] / 365.25, 1-sprint["y"])

    truth = sprint["t"] / 365.25
    preds = sb.pred_dist(sprint["X"])

    pred_means = preds.mean.detach().numpy()
    print(calculate_concordance_naive(pred_means, sprint["t"] / 365.25, 
                                      1 - sprint["y"]))

