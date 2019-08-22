#!/bin/sh
python3 -m examples.simulated.regression_convergence --dataset=simulated --score=MLE --natural
python3 -m examples.simulated.regression_convergence --dataset=simulated --score=CRPS --natural
python3 -m examples.simulated.regression_convergence --dataset=simulated --score=MLE
python3 -m examples.simulated.regression_convergence --dataset=simulated --score=CRPS
