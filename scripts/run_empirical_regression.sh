#!/bin/sh

python3 -m examples.experiments.regression_exp --dataset=housing  --score=MLE --natural --lr=0.0007 --n-est=5000
python3 -m examples.experiments.regression_exp --dataset=concrete  --score=MLE --natural  --lr=0.002 --n-est=5000
python3 -m examples.experiments.regression_exp --dataset=energy  --score=MLE --natural  --lr=0.002 --n-est=5000
python3 -m examples.experiments.regression_exp --dataset=kin8nm  --score=MLE --natural
python3 -m examples.experiments.regression_exp --dataset=naval  --score=MLE --natural
python3 -m examples.experiments.regression_exp --dataset=power  --score=MLE --natural
python3 -m examples.experiments.regression_exp --dataset=protein  --score=MLE --natural --n-splits=5
python3 -m examples.experiments.regression_exp --dataset=wine  --score=MLE --natural
python3 -m examples.experiments.regression_exp --dataset=yacht  --score=MLE --natural
python3 -m examples.experiments.regression_exp --dataset=msd  --score=MLE --natural
