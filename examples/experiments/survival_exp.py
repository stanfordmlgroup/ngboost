import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from dfply import *
from ngboost.distns import LogNormal, Exponential, MultivariateNormal, BivariateLogNormal
from ngboost.api import NGBSurvival
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import *
from ngboost.helpers import Y_from_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis as GBSA
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

np.random.seed(1)

base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
}

def Y_join(T, E):
    col_event = 'Event'
    col_time = 'Time'
    y = np.empty(dtype=[(col_event, np.bool), (col_time, np.float64)],
                 shape=T.shape[0])
    y[col_event] = E
    y[col_time] = T
    return y

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="flchain")
    argparser.add_argument("--distn", type=str, default="LogNormal")
    argparser.add_argument("--n-est", type=int, default=500)
    argparser.add_argument("--lr", type=float, default=.01)
    argparser.add_argument("--score", type=str, default="MLE")
    argparser.add_argument("--natural", action="store_true")
    argparser.add_argument("--base", type=str, default="tree")
    argparser.add_argument("--minibatch-frac", type=float, default=1.0)
    argparser.add_argument("--n-splits", type=int, default=20)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    # processing strategy from [chapfuwa et al 2019]
    # impute missing values with the most frequent
    # then one-hot encode categorical variables

    if args.dataset == "flchain":
        df = pd.read_csv("./data/surv/flchain.csv")
        E = df["death"]
        Y = df["futime"]
        X = df >> drop(X.death, X.futime, X.chapter) \
                >> mutate(mgus=X.mgus.astype(float), age=X.age.astype(float))
        X = X[Y > 0]
        E = E[Y > 0]
        Y = Y[Y > 0]
        #Y = np.c_[np.log(T) - np.mean(np.log(T)), C]
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
        #Y = np.c_[np.log(Y) - np.mean(np.log(Y)), C]
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
        E = df["cvd"]
        Y = df["t_cvds"] / 365.25
        #Y = np.c_[np.log(Y) - np.mean(np.log(Y)), C]
        X = (df >> drop("cvd", "t_cvds", "INTENSIVE")).values

    print('== Dataset=%s X.shape=%s Censorship=%.4f' % (args.dataset, str(X.shape), np.mean(1-E)))

    # normalize Y
    logY = np.log(Y)
    Y = np.exp((logY - np.mean(logY)) / np.std(logY))
    E = E.to_numpy()
    Y = Y.to_numpy()

    # split the dataset
    n = X.shape[0]
    np.random.seed(1)
    folds = []

    ngb_cstat = []
    ngb_score = []
    ngb_slope = []
    ngb_intcpt = []
    gbsa_cstat = []

    for i in range(args.n_splits):
        permutation = np.random.choice(range(n), n, replace = False)
        end_train = round(n * 9.0 / 10)
        end_test = n

        train_index = permutation[ 0 : end_train ]
        test_index = permutation[ end_train : n ]
        folds.append( (train_index, test_index) )

    for itr, (train_index, test_index) in enumerate(folds):

        X_trainall, X_test = X[train_index], X[test_index]
        Y_trainall, Y_test = Y[train_index], Y[test_index]
        E_trainall, E_test = E[train_index], E[test_index]

        X_train, X_val, Y_train, Y_val, E_train, E_val = train_test_split(X_trainall, Y_trainall, E_trainall, test_size=0.2)

        ngb = NGBSurvival(Dist=eval(args.distn),
                          n_estimators=args.n_est,
                          learning_rate=args.lr,
                          natural_gradient=args.natural,
                          verbose=args.verbose,
                          minibatch_frac=1.0,
                          Base=base_name_to_learner[args.base],
                          verbose_eval=1,
                          Score=eval(args.score))

        train_losses = ngb.fit(X_train, Y_train, E_train)

        # pick the best iteration on the validation set
        Y_preds = ngb.staged_predict(X_val)
        Y_forecasts = ngb.staged_pred_dist(X_val)

#        val_cstat = [concordance_index_censored(E_val.astype(bool), Y_val, -Y_pred) for Y_pred in Y_preds]
        val_nll = [ngb.Manifold(Y_forecast._params).score(Y_from_censored(Y_val, E_val)).mean() for Y_forecast in Y_forecasts]
#        best_itr = np.argmin(val_cstat) + 1
        best_itr = np.argmin(val_nll) + 1

        # re-train using all the data after tuning number of iterations
        ngb = NGBSurvival(Dist=eval(args.distn),
                          n_estimators=args.n_est,
                          learning_rate=args.lr,
                          natural_gradient=args.natural,
                          verbose=args.verbose,
                          minibatch_frac=1.0,
                          Base=base_name_to_learner[args.base],
                          verbose_eval=1,
                          Score=eval(args.score))
        ngb.fit(X_trainall, Y_trainall, E_trainall)

        # the final prediction for this fold
        forecast = ngb.pred_dist(X_test, max_iter=best_itr)

        ngb_slope += [calibration_time_to_event(forecast, Y_test, E_test, bins=5)[2]]
        ngb_intcpt += [calibration_time_to_event(forecast, Y_test, E_test, bins=5)[3]]
        ngb_score += [ngb.Manifold(forecast._params).score(Y_from_censored(Y_test, E_test)).mean()]
#        ngb_cstat += [concordance_index_censored(E_test.astype(bool), Y_test, -forecast.mean())[0]]
        ngb_cstat += [concordance_index_ipcw(Y_from_censored(Y_trainall, E_trainall), Y_from_censored(Y_test, E_test), -forecast.mean())[0]]
        print('Itr: %d, NGB cstat: %.4f, NGB NLL: %.4f, NGB Slope: %.4f, NGB Intcpt: %.4f, Estimators: %d' % (itr, ngb_cstat[-1], ngb_score[-1], ngb_slope[-1], ngb_intcpt[-1], best_itr))
#
#        gbsa = GBSA(n_estimators=args.n_est,
#                    learning_rate=args.lr,
#                    subsample=args.minibatch_frac,
#                    verbose=args.verbose)
#        gbsa.fit(X_trainall, Y_join(Y_trainall, E_trainall))
#        preds = gbsa.predict(X_test)
#        gbsa_cstat += [concordance_index_censored(E_test.astype(bool), Y_test, preds)[0]]
#        print('Itr: %d, GBSA score: %.4f' % (itr, gbsa_cstat[-1]))
#

    print('==  NGB=%.4f +/- %.4f, NLL NGB=%.4f +/- %.4f, Slope: %.4f +/- %.4f, Intercept: %.4f +/- %.4f' % ( np.mean(ngb_cstat), np.std(ngb_cstat), np.mean(ngb_score), np.std(ngb_score),
                                                                                  np.mean(ngb_slope), np.std(ngb_slope), np.mean(ngb_intcpt), np.std(ngb_intcpt)))
