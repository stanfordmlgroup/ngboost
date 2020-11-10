#!/bin/sh

python3 -m examples.experiments.survival_exp --dataset=flchain  --score=MLE --natural --verbose --distn=LogNormal
python3 -m examples.experiments.survival_exp --dataset=flchain  --score=MLE --natural --verbose --distn=Exponential
python3 -m examples.experiments.survival_exp --dataset=flchain  --score=MLE --natural --verbose --distn=MultivariateNormal

python3 -m examples.experiments.survival_exp --dataset=support  --score=MLE --natural --verbose --distn=LogNormal
python3 -m examples.experiments.survival_exp --dataset=support  --score=MLE --natural --verbose --distn=Exponential
python3 -m examples.experiments.survival_exp --dataset=support  --score=MLE --natural --verbose --distn=MultivariateNormal

python3 -m examples.experiments.survival_exp --dataset=sprint  --score=MLE --natural --verbose --distn=LogNormal
python3 -m examples.experiments.survival_exp --dataset=sprint  --score=MLE --natural --verbose --distn=Exponential
python3 -m examples.experiments.survival_exp --dataset=sprint  --score=MLE --natural --verbose --distn=MultivariateNormal
