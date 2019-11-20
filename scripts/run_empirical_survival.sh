#!/bin/sh

python3 -m examples.experiments.survival_exp --dataset=flchain  --score=MLE --natural --verbose --n-est=1000 --lr=0.1  --distn=LogNormal

exit

python3 -m examples.experiments.survival_exp --dataset=support  --score=MLE --natural --verbose
python3 -m examples.experiments.survival_exp --dataset=sprint  --score=MLE --natural --verbose

exit

python3 -m examples.experiments.survival_exp --dataset=flchain  --score=CRPS --natural --verbose --distn=LogNormal
python3 -m examples.experiments.survival_exp --dataset=support  --score=CRPS --natural --verbose
python3 -m examples.experiments.survival_exp --dataset=sprint  --score=CRPS --natural --verbose
