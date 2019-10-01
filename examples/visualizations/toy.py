import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from ngboost.evaluation import *
from ngboost.ngboost import NGBoost
from ngboost.learners import default_linear_learner, default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE, CRPS
from sklearn.neighbors import KNeighborsRegressor as KNR

np.random.seed(1)

default_knr_learner=lambda : KNR()

def gen_data(n=50, bound=1, deg=3, beta=1, noise=0.9, intcpt=-1):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    h = np.linspace(-bound, bound, n)[:, np.newaxis]
    e = np.random.randn(*x.shape) * (0.1 + 10 * np.abs(x))
    y = 50 * (x ** deg) + h * beta + noise * e + intcpt
    return x, y, np.c_[h, np.ones_like(h)]


if __name__ == "__main__":

    x_tr, y_tr, _ = gen_data(n=100)

    poly_transform = PolynomialFeatures(1)
    x_tr = poly_transform.fit_transform(x_tr)

    ngb = NGBoost(Base=default_tree_learner,
                  Dist=Normal,
                  Score=MLE(),
                  n_estimators=3200,
                  learning_rate=.01,
                  natural_gradient=False,
                  minibatch_frac=.1,
                  verbose=True)

    train_loss, val_loss = ngb.fit(x_tr, y_tr)

    x_te, y_te, _ = gen_data(n=1000, bound=1.3)
    x_te = poly_transform.transform(x_te)
    preds = ngb.pred_dist(x_te)

    pctles, obs, _, _ = calibration_regression(preds, y_te)
    #plot_calibration_curve(pctles, obs)
    #plt.show()

    filenames = []
    all_preds = ngb.staged_pred_dist(x_te)
    for i, preds in enumerate(all_preds):
        if i % 20 != 0:
            continue
        plt.figure(figsize = (5, 5))
        plt.scatter(x_tr[:,1], y_tr, color = "black", marker = ".", alpha=0.5)
        plt.plot(x_te[:,1], preds.loc, color = "black", linestyle = "-", linewidth=1)
        plt.plot(x_te[:,1], preds.loc - 1.96 * preds.scale, color = "black", linestyle = "--", linewidth=0.3)
        plt.plot(x_te[:,1], preds.loc + 1.96 * preds.scale, color = "black", linestyle = "--", linewidth=0.3)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.ylim([-60, 60])
        plt.tight_layout()

        filenames.append("./figures/anim/toy%d.png" % i)
        print("Saving ./figures/anim/toy%d.png" % i)
        plt.savefig("./figures/anim/toy%d.png" % i)

    import imageio
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('./figures/toy.gif', images)
