
def MLE_surv(Forecast, Y, C):
    return - ((1 - C) * Forecast.log_prob(Y) + (1 - Forecast.cdf(Y) + 1e-5).log() * C)

def CRPS_surv(Forecast, Y, C):
    return 0

