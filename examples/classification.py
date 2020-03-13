import numpy as np
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from ngboost.learners import default_tree_learner

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":

    np.random.seed(12345)

    X, Y = load_breast_cancer(True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    ngb = NGBClassifier(Dist=Bernoulli)
    ngb.fit(X_train, Y_train)

    preds = ngb.pred_dist(X_test)
    print("ROC:", roc_auc_score(Y_test, preds.probs[1]))
