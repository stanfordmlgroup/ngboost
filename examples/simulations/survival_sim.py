import numpy as np
from ngboost.distns import LogNormal, Exponential, Normal, BivariateNormal
from ngboost.api import NGBSurvival
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import calibration_time_to_event, plot_calibration_curve, calibration_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sksurv.metrics import concordance_index_censored
from matplotlib import pyplot as plt
from argparse import ArgumentParser


if __name__ == '__main__':

    np.random.seed(123)

    argparser = ArgumentParser()
    argparser.add_argument("--n-estimators", type=int, default=100)
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--eps", type=float, default=1.0)
    args = argparser.parse_args()

    m, n = 1000, 10
    X = np.random.randn(m, n) / np.sqrt(n)
    theta = np.random.randn((n,))
    Y = X @ theta + 0.5 * np.random.randn(*(m,))
    T = X @ theta + 0.5 * np.random.randn(*(m,)) + args.eps
    E = (T > Y).astype(int)

    print(X.shape, Y.shape, E.shape)
    print(f"Event rate: {np.mean(E):.2f}")

    X_tr, X_te, Y_tr, Y_te, T_tr, T_te, E_tr, E_te = train_test_split(
        X, Y, T, E, test_size=0.2)

    ngb = NGBSurvival(Dist=BivariateNormal,
                      n_estimators=args.n_estimators,
                      learning_rate=args.lr,
                      natural_gradient=False,
                      Base=default_linear_learner,
                      Score=MLE,
                      verbose=True,
                      verbose_eval=1)
    train_losses = ngb.fit(X_tr, np.minimum(Y_tr, T_tr), E_tr)

    preds = ngb.pred_dist(X_te)
    print(f"R2: {r2_score(Y_te, preds.mean())}")

    print(f"C-stat: {concordance_index_censored(E_te.astype(bool), Y_te, -preds.mean())[0]}")

    plt.hist(preds.mean(), range=(0, 10), bins=30, alpha=0.5, label="Pred")
    plt.hist(Y_te, range=(0, 10), bins=30, alpha=0.5, label="True")
    plt.legend()
    plt.show()

    # since we simulated the data we fully observe all outcomes
    # calibration assuming complete observations
    pctles, observed, slope, intercept = calibration_regression(preds, Y_te)
    plot_calibration_curve(pctles, observed)
    plt.show()

    # calibration for partial observations
    pctles, observed, slope, intercept = calibration_time_to_event(preds, np.minimum(Y_te, T_te).squeeze(), E_te.squeeze())
    plot_calibration_curve(pctles, observed)
    plt.show()

