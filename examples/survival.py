from ngboost import NGBSurvival
from ngboost.distns import LogNormal
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

if __name__ == "__main__":

    X, Y = load_boston(True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # introduce administrative censoring
    T_train = np.minimum(Y_train, 30)
    E_train = Y_train > 30

    ngb = NGBSurvival(Dist=LogNormal).fit(X_train, T_train, E_train)
    Y_preds = ngb.predict(X_test)
    Y_dists = ngb.pred_dist(X_test)

    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    print("Test MSE", test_MSE)

    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
    print("Test NLL", test_NLL)
