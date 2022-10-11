from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.preprocessing import PolynomialFeatures

from ngboost.distns import Normal
from ngboost.evaluation import *
from ngboost.learners import default_tree_learner
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE

np.random.seed(1)

default_knr_learner = lambda: KNR()


def gen_data(n=50, bound=1, deg=3, beta=1, noise=0.9, intcpt=-1):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    h = np.linspace(-bound, bound, n)[:, np.newaxis]
    e = np.random.randn(*x.shape) * (0.1 + 10 * np.abs(x))
    y = 50 * (x**deg) + h * beta + noise * e + intcpt
    return x, y.squeeze(), np.c_[h, np.ones_like(h)]


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n-estimators", type=int, default=301)
    argparser.add_argument("--lr", type=float, default=0.03)
    argparser.add_argument("--minibatch-frac", type=float, default=0.1)
    argparser.add_argument("--natural", action="store_true")
    args = argparser.parse_args()

    x_tr, y_tr, _ = gen_data(n=50)

    poly_transform = PolynomialFeatures(1)
    x_tr = poly_transform.fit_transform(x_tr)

    ngb = NGBoost(
        Base=default_tree_learner,
        Dist=Normal,
        Score=MLE,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        natural_gradient=args.natural,
        minibatch_frac=args.minibatch_frac,
        verbose=True,
    )

    ngb.fit(x_tr, y_tr)

    x_te, y_te, _ = gen_data(n=1000, bound=1.3)
    x_te = poly_transform.transform(x_te)
    preds = ngb.pred_dist(x_te)

    pctles, obs, _, _ = calibration_regression(preds, y_te)

    all_preds = ngb.staged_pred_dist(x_te)
    preds = all_preds[-1]
    plt.figure(figsize=(6, 3))
    plt.scatter(x_tr[:, 1], y_tr, color="black", marker=".", alpha=0.5)
    plt.plot(
        x_te[:, 1],
        preds.loc,
        color="black",
        linestyle="-",
        linewidth=1,
        label="Predicted mean",
    )
    plt.plot(
        x_te[:, 1],
        preds.loc - 1.96 * preds.scale,
        color="black",
        linestyle="--",
        linewidth=0.3,
        label="95\% prediction interval",
    )
    plt.plot(
        x_te[:, 1],
        preds.loc + 1.96 * preds.scale,
        color="black",
        linestyle="--",
        linewidth=0.3,
    )
    plt.ylim([-75, 75])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("./figures/anim/toy_single.pdf")
    plt.show()
