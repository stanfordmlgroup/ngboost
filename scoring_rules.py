import torch
import numpy as np

def MLE_surv(Forecast, Y, C):
    return - ((1 - C) * Forecast.log_prob(Y) + (1 - Forecast.cdf(Y) + 1e-5).log() * C)

def CRPS_surv(Forecast, Y, C):
    def I(F, G, U):
        prev_F = 0.
        prev_x = 0.
        I_sum = 0.
        for th in np.linspace(0, 1, 32):
            if th == 0:
                continue
            this_x = U * th
            this_F = F(this_x) * G(this_x)
            Fdiff = 0.5 * (this_F + prev_F)
            xdiff = this_x - prev_x
            I_sum += (Fdiff * xdiff)
            prev_F = this_F
            prev_x = this_x
        return I_sum

    left = I(lambda y: Forecast.cdf(y).pow(2), lambda x: 1, Y)
    right = I(lambda y: (1 - Forecast.cdf(y)).pow(2), lambda x: x.pow(-2), 1/Y)
    return (left + (1-C) * right)
