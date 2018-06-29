library(survival)
library(gbm)
library(pracma)
library(ranger)
library(tidyverse)
library(BART)
library(flexsurv)
library(RcppCNPy)


setwd("~/Projects/survival-boosting/")

train_data = read_csv("data/simulated/sim_data_train.csv")
test_data = read_csv("data/simulated/sim_data_test.csv")

# ---
# Survival BART -- code is not yet working
# ---

surv_bart = mc.surv.bart(x.train = as.matrix(train_data %>% select(-c(Y, C))),
                         times = train_data$Y,
                         delta = 1 - train_dataata$C)
preds = predict(surv_bart, newdata = as.matrix(test_data %>% select(-c(Y, C))))

# ---
# Regression models
# ---

reg_model = survreg(Surv(Y, 1 - C) ~ ., data = train_data, dist = "exponential")
preds = predict(reg_model, type = "response", data = test_data)
npySave("data/simulated/sim_preds_reg_exp.npy", preds)

reg_model = survreg(Surv(Y, 1 - C) ~ ., data = train_data, dist = "lognormal")
preds = predict(reg_model, type="response", data = test_data)
npySave("data/simulated/sim_preds_reg_ln.npy", preds)

reg_model = survreg(Surv(Y, 1 - C) ~ ., data = train_data, dist = "weibull")
mus = predict(reg_model, type="response", data = test_data)
npySave("data/simulated/sim_preds_reg_weib.npy", mus)

plot(density(test_data$Y))
lines(density(preds))

cox_fit = coxph(Surv(Y, 1 - C) ~ ., data = train_data)
hazard_ratios = predict(cox_fit, newdata = test_data, type = "expected")
npySave("data/simulated/sim_preds_cox.npy", hazard_ratios)

# ---
# Random survival forest [outputs _restricted_ expected times to event]
# ---

surv_forest = ranger(Surv(Y, 1 - C) ~ ., data = train_data)
preds = predict(surv_forest, data = test_data)
exp = apply(preds$survival, 1, function(x) trapz(preds$unique.death.times, x) )
npySave("data/simulated/sim_preds_survforest.npy", exp)

