library(survival)
library(gbm)
library(pracma)
library(ranger)
library(tidyverse)
library(BART)
library(RcppCNPy)


setwd("~/Projects/survival-boosting/")

train_data = read_csv("data/simulated/sim_data_train.csv")
test_data = read_csv("data/simulated/sim_data_test.csv")

r2_score = function(true, pred) {
  1 - (sum((true - pred )^2) / sum((true - mean(true))^2))
}

# ---
# Survival BART -- code is not yet working
# ---
#
# train_data = train_data[1:10,]
# test_data = test_data[1:10,]
# surv_bart = mc.surv.bart(x.train = as.matrix(train_data %>% select(-c(Y, C))),
#                          times = train_data$Y,
#                          delta = 1 - train_data$C)
# preds = predict(surv_bart, newdata = as.matrix(test_data %>% select(-c(Y, C))))

# ---
# AFT regression models
# ---

predict_reg_model = function(reg_model, data) {
  n_coeffs = length(reg_model$coefficients)
  return(as.matrix(data) %*% coef(reg_model)[2:n_coeffs] + coef(reg_model)[1])
}

reg_model = survreg(Surv(Y, 1 - C) ~ ., data = train_data, dist = "exponential")
preds = exp(predict_reg_model(reg_model, test_data[,1:5]))
#logliks = dexp(test_data$Y, rate = 1 / preds, log = TRUE)
#logtails = pexp(test_data$Y, rate = 1 / preds, log = TRUE)
#write_csv(tibble(pred = preds, loglik = logliks, logtails = logtails),
npySave("data/simulated/sim_preds_reg_exp.npy", preds)

reg_model = survreg(Surv(Y, 1 - C) ~ ., data = train_data, dist = "lognormal")
preds = exp(predict_reg_model(reg_model, test_data[,1:5]))
npySave("data/simulated/sim_preds_reg_ln.npy", preds)

# ---
# Cox models
# ---

cox_fit = coxph(Surv(Y, 1 - C) ~ ., data = train_data)

hazard_ratios = predict(cox_fit, newdata = test_data, type = "lp")
npySave("data/simulated/sim_preds_cox.npy", hazard_ratios)

cox_fit = gbm(Surv(Y, 1 - C) ~ ., data = train_data, distribution = "coxph")
hazards = predict(cox_fit, newdata = test_data, n.trees = 100, type = "response")


# ---
# Random survival forest [outputs _restricted_ expected times to event]
# ---

surv_forest = ranger(Surv(Y, 1 - C) ~ ., data = train_data)
preds = predict(surv_forest, data = test_data)
expected = apply(preds$survival, 1, function(x) trapz(preds$unique.death.times, x) )
npySave("data/simulated/sim_preds_survforest.npy", expected)


gb <- mboost(Surv(time, status) ~ ., data = lung, baselearner = "bols",
            control = boost_control(mstop = 200),
            family = Weibull())

