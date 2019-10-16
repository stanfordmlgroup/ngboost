import numpy as np
from ngboost.distns.normal import Normal
from ngboost.distns.lognormal import LogNormal
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE, CRPS, MLE_SURV, CRPS_SURV
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.evaluation import calibration_time_to_event, plot_calibration_curve, calibration_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from argparse import ArgumentParser


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--n-estimators", type=int, default=100)
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--eps", type=float, default=-1.0)
    args = argparser.parse_args()

    m, n = 1000, 5
    X = np.random.randn(m, n) / np.sqrt(n)
    Y = X @ np.ones((n, 1)) + 0.5 * np.random.randn(*(m, 1))
    T = X @ np.ones((n, 1)) + 0.5 * np.random.randn(*(m, 1)) + args.eps
    C = (T < Y).astype(int)

    print(X.shape, Y.shape, C.shape)
    print(f"Censorship: {np.mean(C):.2f}")

    X_tr, X_te, Y_tr, Y_te, T_tr, T_te, C_tr, C_te = train_test_split(
        X, Y, T, C, test_size=0.2)

    ngb = NGBoost(Dist=Laplace,
                  n_estimators=args.n_estimators,
                  learning_rate=args.lr,
                  natural_gradient=False,
                  Base=default_linear_learner,
                  Score=MLE_SURV())
    train_losses = ngb.fit(X_tr, np.c_[np.minimum(Y_tr, T_tr), C_tr])

    preds = ngb.pred_dist(X_te)
    print(f"R2: {r2_score(Y_te, preds.loc)}")

    plt.hist(preds.loc, range=(-5, 5), bins=30, alpha=0.5, label="Pred")
    plt.hist(Y_te, range=(-5, 5), bins=30, alpha=0.5, label="True")
    plt.legend()
    plt.show()

    # since we simulated the data we fully observe all outcomes
    pctles, observed, slope, intercept = calibration_regression(preds, Y_te)
    plot_calibration_curve(pctles, observed)
    print(f"== Mean SD: {preds.scale.mean()}")
    plt.show()

    pctles, observed, slope, intercept = calibration_time_to_event(preds, np.minimum(Y_te, T_te).squeeze(), C_te.squeeze())
    plot_calibration_curve(pctles, observed)
    plt.show()
