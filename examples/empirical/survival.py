import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sksurv.datasets import load_flchain
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from dfply import *
from ngboost.distns import Normal, Laplace, LogNormal
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE_SURV, CRPS_SURV
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import *


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

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="flchain")
    args = argparser.parse_args()

    # processing strategy from [chapfuwa 2019]
    # impute missing values with the most frequent
    # then one-hot encode categorical variables

    if args.dataset == "flchain":
        X, Y = load_flchain()
        C = np.array([t[0] for t in Y])
        T = np.array([t[1] for t in Y]) / 365.25
        X = X[T > 0]
        C = C[T > 0]
        T = T[T > 0]
        Y = np.c_[np.log(T) - np.mean(np.log(T)), C]
        imputer = SimpleImputer(strategy="most_frequent")
        X = imputer.fit_transform(X.values)
        X_num = np.c_[X[:,0], X[:,2], X[:,3], X[:,4], X[:,5],].astype(float)
        X_cat = np.c_[X[:,1], X[:,6], X[:,7], X[:,8]]
        encoder = OneHotEncoder(sparse=False)
        X_cat = encoder.fit_transform(X_cat)
        X = np.c_[X_num, X_cat]

    elif args.dataset == "support":
        csv = pd.read_csv("./data/vanderbilt/support2.csv")
        csv = csv.rename(columns={"d.time": "dtime"})
        T = csv["dtime"]
        C = 1 - csv["death"]
        Y = np.c_[np.log(T) - np.mean(np.log(T)), C]
        csv >>= drop(X.dtime, X.deathr)
        X_num = csv.select_dtypes(include=["float", "int"])
        X_cat = csv.select_dtypes(exclude=["float", "int"])
        imputer = SimpleImputer(strategy="median")
        X_num = imputer.fit_transform(X_num.values)
        imputer = SimpleImputer(strategy="most_frequent")
        X_cat = imputer.fit_transform(X_cat.values)
        breakpoint()
        encoder = OneHotEncoder(sparse=False)
        X_cat = encoder.fit_transform(X_cat)
        X = np.c_[X_num, X_cat]
        breakpoint()

    elif args.dataset == "sprint":
        sprint = load_data("sprint")
        sprint["X"] = np.c_[sprint["X"], sprint["w"]]
        X = sprint["X"]
        T = np.log(sprint["t"])
        C = 1 - sprint["y"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    print("Shape:", X.shape)
    print("Censorship rate:", np.mean(C))

    ngb = NGBoost(Dist=Normal,
                  n_estimators=50,
                  learning_rate=1.0,
                  natural_gradient=True,
                  verbose=True,
                  minibatch_frac=1.0,
                  Base=default_linear_learner,
                  Score=MLE_SURV())
    train_losses = ngb.fit(X_train, Y_train)

    preds = ngb.pred_dist(X_test)
    print("Scale: %.4f" % preds.scale.mean())
    c_stat = calculate_concordance_naive(preds.ppf(0.5), Y_test[:,0], Y_test[:,1])
    print("C stat: %.4f" % c_stat)
    c_stat = calculate_concordance_dead_only(preds.ppf(0.5), Y_test[:,0], Y_test[:,1])
    print("C stat: %.4f" % c_stat)

    pred, obs, _, _ = calibration_time_to_event(preds, Y_test[:,0], Y_test[:,1])
    plot_calibration_curve(pred, obs)
    plt.show()
    plot_pit_histogram(pred, obs)
    plt.show()
    print("True median [uncens]:", np.median(Y_test[:,0][Y_test[:,1] == 0]))
    print("True median [cens]:", np.median(Y_test[:,0][Y_test[:,1] == 1]))
    print("Pred median:", preds.ppf(0.5).mean())
    # print("Calibration slope: %.4f, intercept: %.4f" % (slope, intercept))

    # print("Mean CoV:", np.mean(np.sqrt(preds.var) / preds.loc))
