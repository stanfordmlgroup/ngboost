import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from dfply import *
from ngboost.distns import LogNormal, Exponential, MultivariateNormal
from ngboost.api import NGBSurvival
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import *
from sksurv.ensemble import GradientBoostingSurvivalAnalysis as GBSA
from sksurv.metrics import concordance_index_censored

np.random.seed(1)

base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
}


def Y_join(T, E):
    col_event = "Event"
    col_time = "Time"
    y = np.empty(dtype=[(col_event, np.bool), (col_time, np.float64)], shape=T.shape[0])
    y[col_event] = E.values
    y[col_time] = T.values
    return y


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="flchain")
    argparser.add_argument("--distn", type=str, default="LogNormal")
    argparser.add_argument("--n-est", type=int, default=200)
    argparser.add_argument("--reps", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--score", type=str, default="MLE")
    argparser.add_argument("--natural", action="store_true")
    argparser.add_argument("--base", type=str, default="tree")
    argparser.add_argument("--minibatch-frac", type=float, default=1.0)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    # processing strategy from [chapfuwa et al 2019]
    # impute missing values with the most frequent
    # then one-hot encode categorical variables

    if args.dataset == "flchain":
        df = pd.read_csv("./data/surv/flchain.csv")
        E = df["death"]
        Y = df["futime"]
        X = (
            df
            >> drop(X.death, X.futime, X.chapter)
            >> mutate(mgus=X.mgus.astype(float), age=X.age.astype(float))
        )
        X = X[Y > 0]
        E = E[Y > 0]
        Y = Y[Y > 0]
        # Y = np.c_[np.log(T) - np.mean(np.log(T)), C]
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
        Y = df["dtime"]
        E = df["death"]
        # Y = np.c_[np.log(Y) - np.mean(np.log(Y)), C]
        df >>= drop(
            X.dtime,
            X.death,
            X.hospdead,
            X.prg2m,
            X.prg6m,
            X.dnr,
            X.dnrday,
            X.aps,
            X.sps,
            X.surv2m,
            X.surv6m,
            X.totmcst,
        )
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
        E = df["cvd"]
        Y = df["t_cvds"] / 365.25
        # Y = np.c_[np.log(Y) - np.mean(np.log(Y)), C]
        X = (df >> drop("cvd", "t_cvds", "INTENSIVE")).values

    print(
        "== Dataset=%s X.shape=%s Censorship=%.4f"
        % (args.dataset, str(X.shape), np.mean(1 - E))
    )

    for itr in range(args.reps):

        X_train, X_test, Y_train, Y_test, E_train, E_test = train_test_split(
            X, Y, E, test_size=0.2
        )
        X_train, X_val, Y_train, Y_val, E_train, E_val = train_test_split(
            X_train, Y_train, E_train, test_size=0.2
        )

        ngb = NGBSurvival(
            Dist=eval(args.distn),
            n_estimators=args.n_est,
            learning_rate=args.lr,
            natural_gradient=args.natural,
            verbose=args.verbose,
            minibatch_frac=1.0,
            Base=base_name_to_learner[args.base],
            Score=eval(args.score),
        )

        train_losses = ngb.fit(X_train, Y_train, E_train)
        forecast = ngb.pred_dist(X_test)
        train_forecast = ngb.pred_dist(X_train)
        print(
            "NGB score: %.4f (val), %.4f (train)"
            % (
                concordance_index_censored(
                    E_test.astype(bool), Y_test, -forecast.mean()
                )[0],
                concordance_index_censored(
                    E_train.astype(bool), Y_train, -train_forecast.mean()
                )[0],
            )
        )

        ##
        ## sksurv
        ##
        gbsa = GBSA(
            n_estimators=args.n_est,
            learning_rate=args.lr,
            subsample=args.minibatch_frac,
            verbose=args.verbose,
        )
        gbsa.fit(X_train, Y_join(Y_train, E_train))
        print(
            "GBSA score: %.4f (val), %.4f (train)"
            % (
                gbsa.score(X_test, Y_join(Y_test, E_test)),
                gbsa.score(X_train, Y_join(Y_train, E_train)),
            )
        )
