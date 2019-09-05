import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from dfply import *
from ngboost.distns import Normal, Laplace, LogNormal
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE_SURV, CRPS_SURV
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="flchain")
    argparser.add_argument("--distn", type=str, default="Normal")
    argparser.add_argument("--n-est", type=int, default=100)
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--score", type=str, default="CRPS_SURV")
    argparser.add_argument("--natural", action="store_true")
    args = argparser.parse_args()

    # processing strategy from [chapfuwa et al 2019]
    # impute missing values with the most frequent
    # then one-hot encode categorical variables

    if args.dataset == "flchain":
        df = pd.read_csv("./data/surv/flchain.csv")
        C = 1 - df["death"]
        T = df["futime"]
        X = df >> drop(X.death, X.futime, X.chapter) \
                >> mutate(mgus=X.mgus.astype(float), age=X.age.astype(float))
        X = X[T > 0]
        C = C[T > 0]
        T = T[T > 0]
        Y = np.c_[np.log(T) - np.mean(np.log(T)), C]
        X_num = X.select_dtypes(include=["float"])
        X_cat = X.select_dtypes(exclude=["float"])
        imputer = SimpleImputer(strategy="median")
        X_num = imputer.fit_transform(X_num.values)
        imputer = SimpleImputer(strategy="most_frequent")
        X_cat = imputer.fit_transform(X_cat.values)
        encoder = OneHotEncoder(sparse=False)
        X_cat = encoder.fit_transform(X_cat)
        X = np.c_[X_num, X_cat]

    elif args.dataset == "support":
        df = pd.read_csv("./data/surv/support2.csv")
        df = df.rename(columns={"d.time": "dtime"})
        T = df["dtime"]
        C = 1 - df["death"]
        Y = np.c_[np.log(T) - np.mean(np.log(T)), C]
        df >>= drop(X.dtime, X.death, X.hospdead, X.prg2m, X.prg6m, X.dnr,
                     X.dnrday, X.aps, X.sps, X.surv2m, X.surv6m, X.totmcst)
        X_num = df.select_dtypes(include=["float", "int"])
        X_cat = df.select_dtypes(exclude=["float", "int"])
        imputer = SimpleImputer(strategy="median")
        X_num = imputer.fit_transform(X_num.values)
        imputer = SimpleImputer(strategy="most_frequent")
        X_cat = imputer.fit_transform(X_cat.values)
        encoder = OneHotEncoder(sparse=False)
        X_cat = encoder.fit_transform(X_cat)
        X = np.c_[X_num, X_cat]

    elif args.dataset == "sprint":
        df = pd.read_csv("data/surv/sprint-cut.csv")
        C = 1 - df["cvd"]
        T = df["t_cvds"] / 365.25
        Y = np.c_[np.log(T) - np.mean(np.log(T)), C]
        X = (df >> drop("cvd", "t_cvds", "INTENSIVE")).values

    #poly_transform = PolynomialFeatures(2)
    #X = poly_transform.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    print("Shape:", X.shape)
    print("Censorship rate:", np.mean(C))

    ngb = NGBoost(Dist=eval(args.distn),
                  n_estimators=args.n_est,
                  learning_rate=args.lr,
                  natural_gradient=args.natural,
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
    print("Mean CoV:", np.mean(np.sqrt(preds.var) / preds.loc))

    pred, obs, _, _ = calibration_time_to_event(preds, Y_test[:,0], Y_test[:,1])
    plot_calibration_curve(pred, obs)
    plt.show()
    plot_pit_histogram(pred, obs)
    plt.show()
    print("True median [uncens]:", np.median(Y_test[:,0][Y_test[:,1] == 0]))
    print("True median [cens]:", np.median(Y_test[:,0][Y_test[:,1] == 1]))
    print("Pred median:", preds.ppf(0.5).mean())
    print("Calibration slope: %.4f, intercept: %.4f" % (slope, intercept))
