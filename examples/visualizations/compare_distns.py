import numpy as np
import scipy as sp
import scipy.stats
from ngboost.distns import Normal, Laplace
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE, CRPS, MLE_SURV, CRPS_SURV
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import *
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from argparse import ArgumentParser


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--dist", type=str, default="Normal")
    argparser.add_argument("--noise-dist", type=str, default="Normal")
    args = argparser.parse_args()

    m, n = 1000, 50
    if args.noise_dist == "Normal":
        noise = np.random.randn(*(m, 1))
    elif args.noise_dist == "Laplace":
        noise = sp.stats.laplace.rvs(size=(m, 1))
    beta = np.random.randn(n, 1)
    X = np.random.randn(m, n) / np.sqrt(n)
    Y = X @ beta + 0.5 * noise + 20
    print(X.shape, Y.shape)

    ngb = NGBoost(n_estimators=100, learning_rate=1.,
                  Dist=eval(args.dist),
                  Base=default_linear_learner,
                  natural_gradient=True,
                  minibatch_frac=1.0,
                  Score=MLE())
    ngb.fit(X, Y)

    preds = ngb.pred_dist(X)
    print(f"R2: {r2_score(Y, preds.loc):.4f}")

    pctles, observed, slope, intercept = calibration_regression(preds, Y)
    print(observed)
    plt.figure(figsize = (8, 3))
    plt.subplot(1, 2, 1)
    plot_calibration_curve(pctles, observed)
    plt.subplot(1, 2, 2)
    plot_pit_histogram(pctles, observed)
    plt.tight_layout()
    plt.savefig("./figures/pit.pdf")
    plt.show()
