import numpy as np
import scipy as sp
import scipy.stats
from ngboost.distns import Normal
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import *
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from argparse import ArgumentParser


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--lr", type=float, default=0.1)
    argparser.add_argument("--dataset", type=str, default="simulated")
    argparser.add_argument("--noise-lvl", type=float, default=0.25)
    argparser.add_argument("--distn", type=str, default="Normal")
    argparser.add_argument("--natural", action="store_true")
    argparser.add_argument("--score", type=str, default="CRPS")
    args = argparser.parse_args()

    np.random.seed(123)

    m, n = 1200, 50
    noise = np.random.randn(*(m, 1))
    beta1 = np.random.randn(n, 1)
    X = np.random.randn(m, n) / np.sqrt(n)
    Y = (X @ beta1 + args.noise_lvl * noise).squeeze()
    print(X.shape, Y.shape)

    X_train, X_test = X[:1000, :], X[1000:,]
    Y_train, Y_test = Y[:1000], Y[1000:]

    ngb = NGBoost(
        n_estimators=400,
        learning_rate=args.lr,
        Dist=Normal,
        Base=default_linear_learner,
        natural_gradient=args.natural,
        minibatch_frac=1.0,
        Score=eval(args.score)(),
        verbose=True,
        verbose_eval=100,
    )

    losses = ngb.fit(X_train, Y_train)
    forecast = ngb.pred_dist(X_test)
    print("R2:", r2_score(Y_test, forecast.loc))
