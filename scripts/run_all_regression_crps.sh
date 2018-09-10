#!/usr/bin/bash
python3 -m experiments.regression --dataset housing --score=CRPS
python3 -m experiments.regression --dataset concrete --score=CRPS
python3 -m experiments.regression --dataset wine --score=CRPS
python3 -m experiments.regression --dataset kin8nm --score=CRPS
python3 -m experiments.regression --dataset naval --score=CRPS
python3 -m experiments.regression --dataset power --score=CRPS
python3 -m experiments.regression --dataset energy --score=CRPS
python3 -m experiments.regression --dataset protein --score=CRPS
