from __future__ import print_function
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from distns import HomoskedasticNormal
from torch.distributions import Normal

from distns import HomoskedasticNormal
from ngboost.ngboost import NGBoost, SurvNGBoost
from experiments.evaluation import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from ngboost.scores import MLE_surv, CRPS_surv

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

def cv_n_estimators(X, y, C, cv_list, n_folds=10, distrib = HomoskedasticNormal, quadrant = False, s = CRPS_surv):
    kf = KFold(n_splits=n_folds)
    kf.get_n_splits(X)
    mse_list = []
    for param in cv_list:
        print("Cross validating with parameter %.2f" % (param))
        mse = 0
        for train_index, val_index in kf.split(X):
            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            C_train_cv, C_val_cv = C[train_index], C[val_index]
            
            base_learner = lambda: DecisionTreeRegressor(criterion='friedman_mse', \
                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3)

            ngb = SurvNGBoost(Base=base_learner,
                  Dist=distrib,
                  Score=s,
                  n_estimators=param,
                  learning_rate=0.1,
                  natural_gradient=True,
                  second_order=True,
                  quadrant_search=quadrant,
                  minibatch_frac=1.0,
                  nu_penalty=1e-5,
                  verbose=False)

            ngb.fit(X_train_cv, y_train_cv, C_train_cv, X_val_cv, y_val_cv, C_val_cv)
            y_pred = ngb.pred_mean(X_val_cv)
            mse_cv = mean_squared_error(y_val_cv, y_pred)
            #print(mse_cv)
            mse += mse_cv

        mse /= n_folds
        mse_list.append(mse)
    return mse_list

def cv_n_estimators_gbr(X, y, C, cv_list, n_folds=10):
    kf = KFold(n_splits=n_folds)
    kf.get_n_splits(X)
    mse_list = []
    for param in cv_list:
        mse = 0
        for train_index, val_index in kf.split(X):
            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            C_train_cv, C_val_cv = C[train_index], C[val_index]
            
            gbr = GradientBoostingRegressor(n_estimators=param)
            gbr.fit(X_train_cv, y_train_cv)

            y_pred = gbr.predict(X_val_cv)
            mse_cv = mean_squared_error(y_val_cv, y_pred)
            mse += mse_cv
        mse /= n_folds
        mse_list.append(mse)
    return mse_list

if __name__ == "__main__":
    data = load_boston()
    X, y = data["data"], data["target"]
    C = np.zeros(len(y))
    X_train, X_test, y_train, y_test, C_train, C_test = train_test_split(X, y, C)
    n_estimators_list = [10,20]
    fold_num = 2

    print("*"*6 + "  Heteroskedastic Distributions with MLE [Orthan Search]  " + "*"*6)
    #n_estimators_list = [10,50,80,100,150,200,500,600, 800, 1000, 1200,1500]
    het_q_mle = cv_n_estimators(X_train, y_train, C_train, cv_list = n_estimators_list, \
                                n_folds=fold_num, distrib = Normal, quadrant = True, s=MLE_surv)
    optimal_het_q_mle = n_estimators_list[np.argmin(het_q_mle)]
    print("--- Cross Validation MSE ---")
    print(het_q_mle)
    print("--- Optimal parameter for Heteroskedastic Distributions with MLE [Orthan Search] ---")
    print(optimal_het_q_mle)


    print("*"*6 + "  Homoskedastic Distributions with MLE [Orthan Search]  " + "*"*6)
    hom_q_mle = cv_n_estimators(X_train, y_train, C_train, cv_list = n_estimators_list, \
                                n_folds=fold_num, distrib = HomoskedasticNormal, quadrant = True, s = MLE_surv)
    optimal_hom_q_mle = n_estimators_list[np.argmin(hom_q_mle)]
    print("--- Cross Validation MSE ---")
    print(hom_q_mle)
    print("--- Optimal parameter for Heteroskedastic Distributions with MLE [Orthan Search] ---")
    print(optimal_hom_q_mle)


    print("*"*6 + "  Heteroskedastic Distributions with MLE [Line Search]  " + "*"*6)
    het_l_mle = cv_n_estimators(X_train, y_train, C_train, cv_list = n_estimators_list, \
                                n_folds=fold_num, distrib = Normal, quadrant = False, s=MLE_surv)
    optimal_het_l_mle = n_estimators_list[np.argmin(het_l_mle)]

    print("--- Cross Validation MSE ---")
    print(het_l_mle)
    print("--- Optimal parameter for Heteroskedastic Distributions with MLE [Line Search] ---")
    print(optimal_het_l_mle)

    print("*"*6 + "  Homoskedastic Distributions with MLE [Line Search]  " + "*"*6)
    hom_l_mle = cv_n_estimators(X_train, y_train, C_train, cv_list = n_estimators_list, \
                                n_folds=fold_num, distrib = HomoskedasticNormal, quadrant = False, s = MLE_surv)
    optimal_hom_l_mle = n_estimators_list[np.argmin(hom_l_mle)]

    print("--- Cross Validation MSE ---")
    print(hom_l_mle)
    print("--- Optimal parameter for Heteroskedastic Distributions with MLE [Line Search] ---")
    print(optimal_hom_l_mle)


    print("*"*6 + "  Heteroskedastic Distributions with CRPS [Orthan Search]  " + "*"*6)
    het_q_crps = cv_n_estimators(X_train, y_train, C_train, cv_list = n_estimators_list, \
                                n_folds=fold_num, distrib = Normal, quadrant = True, s=CRPS_surv)
    optimal_het_q_crps = n_estimators_list[np.argmin(het_q_crps)]
    print("--- Cross Validation MSE ---")
    print(het_q_crps)
    print("--- Optimal parameter for Heteroskedastic Distributions with CRPS [Orthan Search] ---")
    print(optimal_het_q_crps)


    print("*"*6 + "  Homoskedastic Distributions with CRPS [Orthan Search]  " + "*"*6)
    hom_q_crps = cv_n_estimators(X_train, y_train, C_train, cv_list = n_estimators_list, \
                                n_folds=fold_num, distrib = HomoskedasticNormal, quadrant = True, s = CRPS_surv)
    optimal_hom_q_crps = n_estimators_list[np.argmin(hom_q_crps)]
    print("--- Cross Validation MSE ---")
    print(hom_q_crps)
    print("--- Optimal parameter for Heteroskedastic Distributions with CRPS [Orthan Search] ---")
    print(optimal_hom_q_crps)


    print("*"*6 + "  Heteroskedastic Distributions with CRPS [Line Search]  " + "*"*6)
    het_l_crps = cv_n_estimators(X_train, y_train, C_train, cv_list = n_estimators_list, \
                                n_folds=fold_num, distrib = Normal, quadrant = False, s=CRPS_surv)
    optimal_het_l_crps = n_estimators_list[np.argmin(het_l_crps)]

    print("--- Cross Validation MSE ---")
    print(het_l_crps)
    print("--- Optimal parameter for Heteroskedastic Distributions with CRPS [Line Search] ---")
    print(optimal_het_l_crps)

    print("*"*6 + "  Homoskedastic Distributions with CRPS [Line Search]  " + "*"*6)
    hom_l_crps = cv_n_estimators(X_train, y_train, C_train, cv_list = n_estimators_list, \
                                n_folds=fold_num, distrib = HomoskedasticNormal, quadrant = False, s = CRPS_surv)
    optimal_hom_l_crps = n_estimators_list[np.argmin(hom_l_crps)]

    print("--- Cross Validation MSE ---")
    print(hom_l_crps)
    print("--- Optimal parameter for Heteroskedastic Distributions with CRPS [Line Search] ---")
    print(optimal_hom_l_crps)

    print("*"*6 + "  Gradient Boosting Regressor  " + "*"*6)
    gbr = cv_n_estimators_gbr(X_train, y_train, C_train, cv_list = n_estimators_list, n_folds=fold_num)
    optimal_gbr = n_estimators_list[np.argmin(gbr)]

    print("--- Cross Validation MSE ---")
    print(gbr)
    print("--- Optimal parameter for Gradient Boosting Regressor ---")
    print(optimal_gbr)


    colors = ["denim blue", "amber", "dark pink", "medium green", 
          "mulberry", "pale red", "military green", 
          "prussian blue", "avocado green"]
    customized_palette = sns.xkcd_palette(colors)
    sns.set_palette(customized_palette)


    figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    #plt.style.use('ggplot')
    plt.plot(n_estimators_list, het_q_mle, linewidth=1.5)
    plt.plot(n_estimators_list, hom_q_mle,linestyle="--", linewidth=1.5)
    plt.plot(n_estimators_list, het_l_mle, linewidth=1.5)
    plt.plot(n_estimators_list, hom_l_mle,linestyle="--", linewidth=1.5)

    plt.plot(n_estimators_list, het_q_crps, linewidth=1.5)
    plt.plot(n_estimators_list, hom_q_crps,linestyle="--", linewidth=1.5)
    plt.plot(n_estimators_list, het_l_crps, linewidth=1.5)
    plt.plot(n_estimators_list, hom_l_crps,linestyle="--", linewidth=1.5)

    plt.plot(n_estimators_list, gbr, linestyle=":", linewidth=1.5)


    plt.legend(["Heteroskedastic-MLE [Orthant]", "Homoskedastic-MLE [Orthant]", 
                "Heteroskedastic-MLE [Line]", "Homoskedastic-MLE [Line]",
               "Heteroskedastic-CRPS [Orthant]", "Homoskedastic-CRPS [Orthant]", 
                "Heteroskedastic-CRPS [Line]", "Homoskedastic-CRPS [Line]", "GB"], 
               loc='best')

    plt.xlabel("Number of Estimators")
    plt.ylabel("MSE")
    plt.title("Cross Validation on the Number of Estimators", fontsize=16)
    plt.show()

    