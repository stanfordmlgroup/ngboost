import numpy as np
import scipy as sp
import scipy.stats
from ngboost.distns import Normal, Laplace, LogNormal, LogLaplace
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE, CRPS, MLE_SURV, CRPS_SURV
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import *
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from argparse import ArgumentParser


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--dist", type=str, default="Laplace")
    argparser.add_argument("--noise-dist", type=str, default="Normal")
    args = argparser.parse_args()

    m, n = 1000, 50
    if args.noise_dist == "Normal":
        noise = np.random.randn(*(m, 1))
    elif args.noise_dist == "Laplace":
        noise = sp.stats.laplace.rvs(size=(m, 1))
    beta = np.random.randn(n, 1)
    X = np.random.randn(m, n) / np.sqrt(n)
    Y = np.exp(X @ beta + 0.5 * noise)
    print(X.shape, Y.shape)

    dist = eval("Log" + args.dist)

    ngb = NGBoost(n_estimators=50, learning_rate=0.5,
                  Dist=dist,
                  Base=default_linear_learner,
                  natural_gradient=False,
                  minibatch_frac=1.0,
                  Score=CRPS())
    losses = ngb.fit(X, Y)

    preds = ngb.pred_dist(X)

    print(f"R2: {r2_score(Y, np.exp(preds.loc)):.4f}")
    pctles, observed, slope, intercept = calibration_regression(preds, Y)

    plt.figure(figsize = (8, 3))
    plt.subplot(1, 2, 1)
    plot_pit_histogram(pctles, observed)
    plt.title("Original scale")

    Y = np.log(Y)
    dist = eval(args.dist)

    ngb = NGBoost(n_estimators=50, learning_rate=0.5,
                  Dist=dist,
                  Base=default_linear_learner,
                  natural_gradient=False,
                  minibatch_frac=1.0,
                  Score=CRPS())
    losses = ngb.fit(X, Y)

    preds = ngb.pred_dist(X)

    print(f"R2: {r2_score(Y, np.exp(preds.loc)):.4f}")
    pctles, observed, slope, intercept = calibration_regression(preds, Y)

    plt.subplot(1, 2, 2)
    plot_pit_histogram(pctles, observed)
    plt.title("Log-scale")
    plt.tight_layout()
    plt.savefig("./figures/pit_logscale.pdf")
    plt.show()
