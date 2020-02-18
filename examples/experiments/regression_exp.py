import numpy as np
import pandas as pd
from scipy.stats import norm as norm_dist
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ngboost.distns import Normal, NormalFixedVar
from ngboost import NGBoost, NGBRegressor
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

np.random.seed(1)

dataset_name_to_loader = {
    "housing": lambda: pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        header=None,
        delim_whitespace=True,
    ),
    "concrete": lambda: pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    ),
    "wine": lambda: pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        delimiter=";",
    ),
    "kin8nm": lambda: pd.read_csv("data/uci/kin8nm.csv"),
    "naval": lambda: pd.read_csv(
        "data/uci/naval-propulsion.txt", delim_whitespace=True, header=None
    ).iloc[:, :-1],
    "power": lambda: pd.read_excel("data/uci/power-plant.xlsx"),
    "energy": lambda: pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    ).iloc[:, :-1],
    "protein": lambda: pd.read_csv("data/uci/protein.csv")[
        ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"]
    ],
    "yacht": lambda: pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        header=None,
        delim_whitespace=True,
    ),
    "msd": lambda: pd.read_csv("data/uci/YearPredictionMSD.txt").iloc[:, ::-1],
}

base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
}


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="concrete")
    argparser.add_argument("--reps", type=int, default=5)
    argparser.add_argument("--n-est", type=int, default=2000)
    argparser.add_argument("--n-splits", type=int, default=20)
    argparser.add_argument("--distn", type=str, default="Normal")
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--natural", action="store_true")
    argparser.add_argument("--score", type=str, default="MLE")
    argparser.add_argument("--base", type=str, default="tree")
    argparser.add_argument("--minibatch-frac", type=float, default=None)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    # load dataset -- use last column as label
    data = dataset_name_to_loader[args.dataset]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    if not args.minibatch_frac:
        args.minibatch_frac = 1.0

    print(
        "== Dataset=%s X.shape=%s %s/%s"
        % (args.dataset, str(X.shape), args.score, args.distn)
    )

    y_true, ngb_rmse, ngb_nll = [], [], []

    if args.dataset == "msd":
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
        args.minibatch_frac = 0.1
    else:
        kf = KFold(n_splits=args.n_splits)
        folds = kf.split(X)

        # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
        n = X.shape[0]
        np.random.seed(1)
        folds = []
        for i in range(args.n_splits):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n

            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))

    for itr, (train_index, test_index) in enumerate(folds):

        X_trainall, X_test = X[train_index], X[test_index]
        y_trainall, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainall, y_trainall, test_size=0.2
        )

        y_true += list(y_test.flatten())

        ngb = NGBRegressor(
            Base=base_name_to_learner[args.base],
            Dist=eval(args.distn),
            Score=eval(args.score),
            n_estimators=args.n_est,
            learning_rate=args.lr,
            natural_gradient=args.natural,
            minibatch_frac=args.minibatch_frac,
            verbose=args.verbose,
        )

        ngb.fit(X_train, y_train)

        # pick the best iteration on the validation set
        y_preds = ngb.staged_predict(X_val)
        y_forecasts = ngb.staged_pred_dist(X_val)

        val_rmse = [mean_squared_error(y_pred, y_val) for y_pred in y_preds]
        val_nll = [
            -y_forecast.logpdf(y_val.flatten()).mean() for y_forecast in y_forecasts
        ]
        best_itr = np.argmin(val_rmse) + 1

        # re-train using all the data after tuning number of iterations
        ngb = NGBRegressor(
            Base=base_name_to_learner[args.base],
            Dist=eval(args.distn),
            Score=eval(args.score),
            n_estimators=args.n_est,
            learning_rate=args.lr,
            natural_gradient=args.natural,
            minibatch_frac=args.minibatch_frac,
            verbose=args.verbose,
        )
        ngb.fit(X_trainall, y_trainall)

        # the final prediction for this fold
        forecast = ngb.pred_dist(X_test, max_iter=best_itr)
        forecast_val = ngb.pred_dist(X_val, max_iter=best_itr)

        # set the appropriate scale if using a homoskedastic Normal
        if args.distn == "NormalFixedVar":
            scale = (
                forecast.var * ((forecast_val.loc - y_val.flatten()) ** 2).mean() ** 0.5
            )
            forecast = norm_dist(loc=forecast.loc, scale=scale)

        ngb_rmse += [np.sqrt(mean_squared_error(forecast.mean(), y_test))]
        ngb_nll += [-forecast.logpdf(y_test.flatten()).mean()]

        print(
            "[%d/%d] BestIter=%d RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f"
            % (
                itr + 1,
                args.n_splits,
                best_itr,
                np.sqrt(val_rmse[best_itr - 1]),
                np.sqrt(mean_squared_error(forecast.mean(), y_test)),
                ngb_nll[-1],
            )
        )

    print(
        "== RMSE GBM=%.4f +/- %.4f, NGB=%.4f +/- %.4f, NLL NGB=%.4f +/ %.4f"
        % (
            0.0,
            0.0,
            np.mean(ngb_rmse),
            np.std(ngb_rmse),
            np.mean(ngb_nll),
            np.std(ngb_nll),
        )
    )
