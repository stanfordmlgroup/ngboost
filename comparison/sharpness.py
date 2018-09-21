from __future__ import print_function
import numpy as np

from sklearn.model_selection import train_test_split
from torch.distributions import Normal
from ngboost.ngboost import NGBoost
from experiments.regression import RegressionLogger, base_name_to_learner, \
                                   score_name_to_score
from argparse import ArgumentParser


def load_data(m=500, n=25, alpha=1.0):
    """
    Simulate data generating process.
    """
    X = np.random.random((m, n)) * 2 - 1
    theta_mu = np.random.random(n) * 10 - 5
    theta_logs = np.random.random(n)
    mu = X.dot(theta_mu)
    logs = X.dot(theta_logs)
    s = np.exp(logs)
    y = norm.rvs(size=m) * s + mu
    y += norm.rvs(size=m) * np.mean(s) * np.sqrt((1 / alpha - 1))
    print('Mean MU/STDV: %.3f, %.3f' % (np.mean(mu), np.mean(s)))
    return X, y


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="sharpness")
    argparser.add_argument("--alpha", type=float, default=1.0)
    argparser.add_argument("--n_est", type=int, default=350)
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--score", type=str, default="CRPS")
    argparser.add_argument("--base", type=str, default="tree")
    argparser.add_argument("--n_reps", type=int, default=10)
    argparser.add_argument("--minibatch_frac", type=float, default=None)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()
    args.dataset = "%s_alpha_%.1f" % (args.dataset, args.alpha)

    for rep in range(args.n_reps):

        print('Iter %d' % rep)
        X, y = load_data(m=500, n=25)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        ngb = NGBoost(Base=base_name_to_learner[args.base],
                      Dist=Normal,
                      Score=score_name_to_score[args.score],
                      n_estimators=args.n_est,
                      learning_rate=args.lr,
                      natural_gradient=True,
                      second_order=True,
                      quadrant_search=True,
                      minibatch_frac=args.minibatch_frac,
                      nu_penalty=1e-5,
                      normalize_inputs=True,
                      normalize_outputs=True,
                      verbose=args.verbose)

        ngb.fit(X_train, y_train)
        forecast = ngb.pred_dist(X_test)
        logger.tick(forecast, y_test)

    logger.save()
