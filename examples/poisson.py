import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import gc
from ngboost import NGBRegressor
from ngboost.distns import Poisson
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import poisson as dist

if __name__ == "__main__":

    SEED = 12345
    np.random.seed(SEED)

    """
    This NFL dataset comprises season-long rolling averages of passing stats. It's not an expanding window, so early season games
    rely on data from the previous year. NFL passing touchdowns closely mimic a Poisson distribution. There is plenty of room for 
    improvement, this is just an example. 

    For more detailed NFL data, visit https://github.com/mrcaseb/nflfastR
    """

    url = "https://raw.githubusercontent.com/btatkinson/sample-data/master/nfl_tds.csv"
    df = pd.read_csv(url, error_bad_lines=False)

    x = df.pass_touchdown.values

    mean = np.mean(x)
    variance = np.var(x)

    print("Passing touchdowns mean: {:.4f}".format(mean))
    print("Passing touchdowns variance: {:.4f}".format(variance))

    k = np.arange(x.max() + 1)

    plt.plot(k, dist.pmf(k, mean) * len(x), "bo", markersize=9, label="expected tds")
    sns.distplot(x, kde=False, label="actual tds")
    plt.title("Naive Poisson Dist Using the Mean")
    plt.legend(loc="upper right")
    plt.show()

    # drop non-feature cols
    df = df.drop(
        columns=[
            "season",
            "game_id",
            "posteam",
            "passer_player_name",
            "passer_player_id",
            "defteam",
        ]
    )

    target = "pass_touchdown"
    X = df.drop(columns=[target]).values
    Y = df[target].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=SEED
    )

    # baseline not using predictor data
    avg_tds = np.mean(Y_train)
    y_dist = dist(avg_tds)
    naive_NLL = -y_dist.logpmf(Y_test).mean()

    print(
        "Mean squared error using only the mean: {:.4f}".format(
            mean_squared_error(np.repeat(avg_tds, len(Y_test)), Y_test)
        )
    )
    print(
        "Poisson negative log liklihood without using predictor variables: {:.4f}".format(
            naive_NLL
        )
    )

    ngb = NGBRegressor(Dist=Poisson)

    ngb.fit(X_train, Y_train)

    Y_preds = ngb.predict(X_test)
    Y_dists = ngb.pred_dist(X_test)

    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    print("NGBoost MSE: {:.4f}".format(test_MSE))

    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpmf(Y_test.flatten()).mean()
    print("NGBoost NLL: {:.4f}".format(test_NLL))

    # Let's see if we can improve by dropping confounding variables
    ## Feature importance for loc trees
    feature_importance_mu = ngb.feature_importances_[0]
    feature_columns = list(df.drop(columns=[target]))

    df_mu = pd.DataFrame(
        {"feature": feature_columns, "importance": feature_importance_mu}
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    fig.suptitle("Feature importance plot for distribution parameters", fontsize=17)
    sns.barplot(
        x="importance", y="feature", ax=ax, data=df_mu, color="skyblue"
    ).set_title("mu param")
    plt.show()

    shap.initjs()

    ## SHAP plot for loc trees
    explainer = shap.TreeExplainer(ngb, model_output=0)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_columns)
    plt.show()

    print("Fitting NGBoost again after dropping unused vars...")

    confounding_vars = ["roof"]

    print("PREVIOUS MSE: {:.4f}".format(test_MSE))
    print("PREVIOUS NLL: {:.4f}".format(test_NLL))

    df = df.drop(columns=confounding_vars)
    X = df.drop(columns=[target]).values
    Y = df[target].values

    ngb = NGBRegressor(Dist=Poisson)

    ngb.fit(X_train, Y_train)

    Y_preds = ngb.predict(X_test)
    Y_dists = ngb.pred_dist(X_test)

    # test Mean Squared Error
    new_MSE = mean_squared_error(Y_preds, Y_test)
    print("NEW MSE AFTER DROPPING CONFOUNDING VARS: {:.4f}".format(new_MSE))

    # test Negative Log Likelihood
    new_NLL = -Y_dists.logpmf(Y_test.flatten()).mean()
    print("NEW NLL AFTER DROPPING CONFOUNDING VARS: {:.4f}".format(new_NLL))
