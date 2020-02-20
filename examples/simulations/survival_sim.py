import numpy as np
from ngboost.distns import LogNormal, Exponential, Normal
from ngboost.api import NGBSurvival
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import (
    calibration_time_to_event,
    plot_calibration_curve,
    calibration_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from argparse import ArgumentParser


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n-estimators", type=int, default=100)
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--eps", type=float, default=1.0)
    args = argparser.parse_args()

    m, n = 1000, 5
    X = np.random.randn(m, n) / np.sqrt(n)
    Y = X @ np.ones((n,)) + 0.5 * np.random.randn(*(m,))
    T = X @ np.ones((n,)) + 0.5 * np.random.randn(*(m,)) + args.eps
    E = (T > Y).astype(int)

    print(X.shape, Y.shape, E.shape)
    print(f"Event rate: {np.mean(E):.2f}")

    X_tr, X_te, Y_tr, Y_te, T_tr, T_te, E_tr, E_te = train_test_split(
        X, Y, T, E, test_size=0.2
    )

    ngb = NGBSurvival(
        Dist=Exponential,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        natural_gradient=True,
        Base=default_linear_learner,
        Score=MLE,
        verbose=True,
        verbose_eval=True,
    )
    train_losses = ngb.fit(X_tr, np.exp(np.minimum(Y_tr, T_tr)), E_tr)

    preds = ngb.pred_dist(X_te)
    print(f"R2: {r2_score(Y_te, np.log(preds.mean()))}")

    plt.hist(preds.mean(), range=(0, 10), bins=30, alpha=0.5, label="Pred")
    plt.hist(np.exp(Y_te), range=(0, 10), bins=30, alpha=0.5, label="True")
    plt.legend()
    plt.show()

    # since we simulated the data we fully observe all outcomes
    # calibration assuming complete observations
    pctles, observed, slope, intercept = calibration_regression(preds, Y_te)
    plot_calibration_curve(pctles, observed)
    plt.show()

    # calibration for partial observations
    pctles, observed, slope, intercept = calibration_time_to_event(
        preds, np.minimum(Y_te, T_te).squeeze(), E_te.squeeze()
    )
    plot_calibration_curve(pctles, observed)
    plt.show()
