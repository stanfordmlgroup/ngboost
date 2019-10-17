#!/bin/sh

python3 -m examples.empirical.survival --dataset=flchain  --score=MLE --natural --verbose --n-est=200 --lr=0.1  --distn=Exponential

exit

python3 -m examples.empirical.survival --dataset=support  --score=MLE --natural --verbose
python3 -m examples.empirical.survival --dataset=sprint  --score=MLE --natural --verbose

exit

python3 -m examples.empirical.survival --dataset=flchain  --score=CRPS --natural --verbose --distn=LogNormal
python3 -m examples.empirical.survival --dataset=support  --score=CRPS --natural --verbose
python3 -m examples.empirical.survival --dataset=sprint  --score=CRPS --natural --verbose
