#!/usr/bin/bash
python3 -m experiments.regression --dataset housing --score=MLE --verbose
python3 -m experiments.regression --dataset concrete --score=MLE --verbose
python3 -m experiments.regression --dataset wine --score=MLE --verbose
python3 -m experiments.regression --dataset kin8nm --score=MLE --verbose
python3 -m experiments.regression --dataset naval --score=MLE --verbose
python3 -m experiments.regression --dataset power --score=MLE --verbose
python3 -m experiments.regression --dataset energy --score=MLE --verbose
python3 -m experiments.regression --dataset protein --score=MLE --verbose
python3 -m experiments.regression --dataset yacht --score=MLE --verbose
