#!/usr/bin/bash
python3 -m experiments.regression --dataset housing --score=MLE
python3 -m experiments.regression --dataset concrete --score=MLE
python3 -m experiments.regression --dataset wine --score=MLE
python3 -m experiments.regression --dataset kin8nm --score=MLE
python3 -m experiments.regression --dataset naval --score=MLE
python3 -m experiments.regression --dataset power --score=MLE
python3 -m experiments.regression --dataset energy --score=MLE
python3 -m experiments.regression --dataset protein --score=MLE
