#!/bin/sh


python3 -m examples.empirical.survival --dataset=flchain  --score=CRPS_SURV --natural --verbose
python3 -m examples.empirical.survival --dataset=support  --score=CRPS_SURV --natural --verbose
python3 -m examples.empirical.survival --dataset=sprint  --score=CRPS_SURV --natural --verbose

python3 -m examples.empirical.survival --dataset=flchain  --score=MLE_SURV --natural --verbose
python3 -m examples.empirical.survival --dataset=support  --score=MLE_SURV --natural --verbose
python3 -m examples.empirical.survival --dataset=sprint  --score=MLE_SURV --natural --verbose
