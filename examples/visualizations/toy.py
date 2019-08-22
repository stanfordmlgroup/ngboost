import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from ngboost.evaluation import *
from ngboost.ngboost import NGBoost
from ngboost.learners import default_linear_learner
from ngboost.distns import Normal
from ngboost.scores import MLE, CRPS


def gen_data(n=50, bound=1, deg=2, beta=1, noise=0.1, intcpt=-1):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    h = np.linspace(-bound, bound, n)[:, np.newaxis]
    e = np.random.randn(*x.shape) * (1 + 3 * np.abs(x))
    y = x ** deg + h * beta + noise * e + intcpt
    return x, y, np.c_[h, np.ones_like(h)]


if __name__ == "__main__":

    x_tr, y_tr, _ = gen_data(n=100)

    poly_transform = PolynomialFeatures(2)
    x_tr = poly_transform.fit_transform(x_tr)

    ngb = NGBoost(Base=default_linear_learner,
                  Dist=Normal,
                  Score=MLE(),
                  n_estimators=50,
                  learning_rate=1.0,
                  natural_gradient=False,
                  minibatch_frac=1.0,
                  verbose=True)

    train_loss, val_loss = ngb.fit(x_tr, y_tr)

    x_te, y_te, _ = gen_data(n=1000, bound=1.5)
    x_te = poly_transform.transform(x_te)
    preds = ngb.pred_dist(x_te)

    pctles, obs, _, _ = calibration_regression(preds, y_te)
    plot_calibration_curve(pctles, obs)
    plt.show()

    plt.figure(figsize = (8, 3))
    plt.scatter(x_tr[:,1], y_tr, color = "black", marker = "o", alpha=0.5)
    plt.plot(x_te[:,1], preds.loc, color = "black", linestyle = "-")
    plt.plot(x_te[:,1], preds.loc - 1.96 * preds.scale, color = "black", linestyle = "--")
    plt.plot(x_te[:,1], preds.loc + 1.96 * preds.scale, color = "black", linestyle = "--")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    plt.savefig("./figures/toy.pdf")
    plt.show()
