
def MLE_surv(Forecast, Y, C):
    return - C * Forecast.log_prob(Y) + (1 - Forecast.cdf(Y)).log() * (1 - C)

def CRPS_surv(Forecast, Y, C):
    return 0

