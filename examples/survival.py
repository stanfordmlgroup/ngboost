from ngboost import NGBSegressor
from ngboost.distns import LogNormal
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def Y_join(T, E):
    y = np.empty(dtype=[('Event', np.bool), ('Time', np.float64)],
                 shape=T.shape[0])
    y['Event'] = (1-E).astype(bool)
    y['Time'] = T.astype(float)
    return y

if __name__ == "__main__":

    X, Y = load_boston(True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # introduce administrative censoring 
    Y_tr = Y_join(np.minimum(Y_tr, 30), Y_tr>30)

    ngb = NGBSurvival(Dist=LogNormal).fit(X_train, Y_train)
    Y_preds = ngb.predict(X_test)
    Y_dists = ngb.pred_dist(X_test)

    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    print('Test MSE', test_MSE)

    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
    print('Test NLL', test_NLL)
