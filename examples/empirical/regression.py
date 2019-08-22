import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from ngboost.distns import Normal, Laplace
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from examples.loggers.loggers import RegressionLogger


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
    argparser.add_argument("--n-est", type=int, default=200)
    argparser.add_argument("--distn", type=str, default="Normal")
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--natural", action="store_true")
    argparser.add_argument("--score", type=str, default="CRPS")
    argparser.add_argument("--base", type=str, default="tree")
    argparser.add_argument("--minibatch-frac", type=float, default=None)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    # load dataset -- use last column as label
    data = dataset_name_to_loader[args.dataset]()
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values[:,np.newaxis]

    print("Dataset size:", X.shape)
    logger = RegressionLogger(args)

    # set default minibatch fraction based on dataset size
    if not args.minibatch_frac:
        args.minibatch_frac = min(1.0, 5000 / len(X))

    for _ in range(args.reps):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

        ngb = NGBoost(Base=base_name_to_learner[args.base],
                      Dist=eval(args.distn),
                      Score=score_name_to_score[args.score](64),
                      n_estimators=args.n_est,
                      learning_rate=args.lr,
                      natural_gradient=args.natural,
                      minibatch_frac=args.minibatch_frac,
                      verbose=args.verbose)

        train_loss, val_loss = ngb.fit(X_train, y_train, X_val, y_val)
        forecast = ngb.pred_dist(X_test)
        logger.tick(forecast, y_test)

    logger.save()

