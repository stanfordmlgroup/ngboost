import numpy as np
import scipy as sp
import scipy.stats
from ngboost.distns import Normal, Laplace
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE, CRPS, MLE_SURV, CRPS_SURV
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import *
from examples.loggers.loggers import RegressionLogger
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from argparse import ArgumentParser


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--lr", type=float, default=0.1)
    argparser.add_argument("--dataset", type=str, default="simulated")
    argparser.add_argument("--noise-lvl", type=float, default=0.25)
    argparser.add_argument("--distn", type=str, default="Normal")
    argparser.add_argument("--natural", action="store_true")
    argparser.add_argument("--score", type=str, default="CRPS")
    args = argparser.parse_args()

    m, n = 1200, 50
    noise = np.random.randn(*(m, 1))
    beta1 = np.random.randn(n, 1)
    beta2 = np.random.randn(n, 1)
    X = np.random.randn(m, n) / np.sqrt(n)
    Y = X @ beta1 + args.noise_lvl * np.sqrt(np.exp(X @ beta2)) * noise
    print(X.shape, Y.shape)

    X_train, X_test = X[:1000,:], X[1000:,]
    Y_train, Y_test = Y[:1000], Y[1000:]

    ngb = NGBoost(n_estimators=150, learning_rate=args.lr,
                  Dist=eval(args.distn),
                  Base=default_linear_learner,
                  natural_gradient=args.natural,
                  minibatch_frac=1.0,
                  Score=eval(args.score)())

    losses = ngb.fit(X_train, Y_train)

    preds = ngb.pred_dist(X_test)
    logger = RegressionLogger(args)
    for itr in tqdm(range(len(ngb.scalings))):
        forecast = ngb.pred_dist(X_test, itr)
        logger.tick(forecast, Y_test)
    logger.save()
