#!/usr/bin/bash
python3 -m experiments.regression --dataset housing --score=CRPS --verbose
python3 -m experiments.regression --dataset concrete --score=CRPS --verbose
python3 -m experiments.regression --dataset wine --score=CRPS --verbose
python3 -m experiments.regression --dataset kin8nm --score=CRPS --verbose
python3 -m experiments.regression --dataset naval --score=CRPS --verbose
python3 -m experiments.regression --dataset power --score=CRPS --verbose
python3 -m experiments.regression --dataset energy --score=CRPS --verbose
python3 -m experiments.regression --dataset protein --score=CRPS --verbose
