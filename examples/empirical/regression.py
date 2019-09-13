import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from ngboost.distns import Normal, Laplace, HomoskedasticNormal
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from examples.loggers.loggers import RegressionLogger

from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

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


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="concrete")
    argparser.add_argument("--reps", type=int, default=5)
    argparser.add_argument("--n-est", type=int, default=300)
    argparser.add_argument("--n-splits", type=int, default=20)
    argparser.add_argument("--distn", type=str, default="Normal")
    argparser.add_argument("--lr", type=float, default=0.1)
    argparser.add_argument("--natural", action="store_true")
    argparser.add_argument("--score", type=str, default="CRPS")
    argparser.add_argument("--base", type=str, default="tree")
    argparser.add_argument("--minibatch-frac", type=float, default=None)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    # load dataset -- use last column as label
    data = dataset_name_to_loader[args.dataset]()
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values[:,np.newaxis]

    logger = RegressionLogger(args)
    gbrlog = RegressionLogger(args)
    gbrlog.distn = 'GBR'

    if not args.minibatch_frac:
        args.minibatch_frac = 1.0

    print('== Dataset=%s X.shape=%s %s/%s' % (args.dataset, str(X.shape), args.score, args.distn))

    ngb_rmse, gbm_rmse = [], []
    
    kf = KFold(n_splits=args.n_splits)
    
    for itr, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ngb = NGBoost(Base=base_name_to_learner[args.base],
                      Dist=eval(args.distn),
                      Score=score_name_to_score[args.score](64),
                      n_estimators=args.n_est,
                      learning_rate=args.lr,
                      natural_gradient=args.natural,
                      minibatch_frac=args.minibatch_frac,
                      verbose=args.verbose)

        train_loss, val_loss = ngb.fit(X_train, y_train) #, X_val, y_val)
        forecast = ngb.pred_dist(X_test)

        ngb_rmse += [np.sqrt(mean_squared_error(forecast.loc, y_test))]

        if args.verbose or True:
            print("[%d/%d] %s/%s RMSE=%.4f" % (itr+1, args.n_splits, args.score, args.distn,
                                               np.sqrt(mean_squared_error(forecast.loc, y_test))))
        
        logger.tick(forecast, y_test)

        gbr = GBR(n_estimators=args.n_est,
                  learning_rate=args.lr,
                  subsample=args.minibatch_frac,
                  verbose=args.verbose)
        gbr.fit(X_train, y_train.flatten())
        y_pred = gbr.predict(X_test)
        forecast = HomoskedasticNormal(y_pred.reshape((1, -1)))

        gbm_rmse += [np.sqrt(mean_squared_error(y_pred.flatten(), y_test.flatten()))]
        
        if args.verbose or True:
            print("[%d/%d] GBR RMSE=%.4f" % (itr+1, args.n_splits,
                                             np.sqrt(mean_squared_error(y_pred.flatten(), y_test.flatten()))))
        gbrlog.tick(forecast, y_test)

    print('== RMSE GBM=%.4f, NGB=%.4f' % (np.mean(gbm_rmse), np.mean(ngb_rmse)))

    logger.save()
    gbrlog.save()


