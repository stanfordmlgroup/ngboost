#!/bin/sh
python3 -m examples.simulations.regression_sim --dataset=simulated_90 --score=MLE --natural --noise-lvl=0.90
python3 -m examples.simulations.regression_sim --dataset=simulated_80 --score=MLE --natural --noise-lvl=0.80
python3 -m examples.simulations.regression_sim --dataset=simulated_70 --score=MLE --natural --noise-lvl=0.70
python3 -m examples.simulations.regression_sim --dataset=simulated_60 --score=MLE --natural --noise-lvl=0.60
python3 -m examples.simulations.regression_sim --dataset=simulated_50 --score=MLE --natural --noise-lvl=0.50
python3 -m examples.simulations.regression_sim --dataset=simulated_40 --score=MLE --natural --noise-lvl=0.40
python3 -m examples.simulations.regression_sim --dataset=simulated_30 --score=MLE --natural --noise-lvl=0.30
python3 -m examples.simulations.regression_sim --dataset=simulated_20 --score=MLE --natural --noise-lvl=0.20
python3 -m examples.simulations.regression_sim --dataset=simulated_10 --score=MLE --natural --noise-lvl=0.10
python3 -m examples.simulations.regression_sim --dataset=simulated_05 --score=MLE --natural --noise-lvl=0.05
