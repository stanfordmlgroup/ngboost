from ngboost import NGBClassifier
from ngboost.distns import k_categorical
from ngboost.learners import default_tree_learner

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    X, y = load_breast_cancer(True)
    y[0:15] = 2  # artificially make this a 3-class problem instead of a 2-class problem
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    ngb = NGBClassifier(
        Dist=k_categorical(3)
    )  # tell ngboost that there are 3 possible outcomes
    ngb.fit(X_train, Y_train)  # Y should have only 3 values: {0,1,2}

    # predicted probabilities of class 0, 1, and 2 (columns) for each observation (row)
    preds = ngb.predict_proba(X_test)
