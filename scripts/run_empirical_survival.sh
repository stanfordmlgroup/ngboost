#!/bin/sh

python3 -m examples.experiments.survival_exp --dataset=flchain  --score=MLE --natural  --distn=LogNormal --n-est=600 --lr=0.01
python3 -m examples.experiments.survival_exp --dataset=flchain  --score=MLE --natural  --distn=Exponential --n-est=600 --lr=0.01

python3 -m examples.experiments.survival_exp --dataset=support  --score=MLE --natural  --distn=LogNormal --n-est=1000 --lr=0.01
python3 -m examples.experiments.survival_exp --dataset=support  --score=MLE --natural  --distn=Exponential --n-est=1000 --lr=0.01

python3 -m examples.experiments.survival_exp --dataset=sprint  --score=MLE --natural  --distn=LogNormal --n-est=1000 --lr=0.01
python3 -m examples.experiments.survival_exp --dataset=sprint  --score=MLE --natural  --distn=Exponential --n-est=1000 --lr=0.01

