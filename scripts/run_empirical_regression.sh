#!/bin/sh
python3 -m examples.empirical.regression --dataset=concrete  --score=MLE --natural
python3 -m examples.empirical.regression --dataset=wine  --score=MLE --natural
python3 -m examples.empirical.regression --dataset=kin8nm  --score=MLE --natural
python3 -m examples.empirical.regression --dataset=naval  --score=MLE --natural
python3 -m examples.empirical.regression --dataset=power  --score=MLE --natural
python3 -m examples.empirical.regression --dataset=energy  --score=MLE --natural
python3 -m examples.empirical.regression --dataset=yacht  --score=MLE --natural
python3 -m examples.empirical.regression --dataset=housing  --score=MLE --natural

exit

python3 -m examples.empirical.regression --dataset=concrete  --score=CRPS --natural
python3 -m examples.empirical.regression --dataset=wine  --score=CRPS --natural
python3 -m examples.empirical.regression --dataset=kin8nm  --score=CRPS --natural
python3 -m examples.empirical.regression --dataset=naval  --score=CRPS --natural
python3 -m examples.empirical.regression --dataset=power  --score=CRPS --natural
python3 -m examples.empirical.regression --dataset=energy  --score=CRPS --natural
python3 -m examples.empirical.regression --dataset=yacht  --score=CRPS --natural
python3 -m examples.empirical.regression --dataset=housing  --score=CRPS --natural


python3 -m examples.empirical.regression --dataset=concrete  --score=MLE --distn=HomoskedasticNormal
python3 -m examples.empirical.regression --dataset=wine  --score=MLE --distn=HomoskedasticNormal
python3 -m examples.empirical.regression --dataset=kin8nm  --score=MLE --distn=HomoskedasticNormal
python3 -m examples.empirical.regression --dataset=naval  --score=MLE --distn=HomoskedasticNormal
python3 -m examples.empirical.regression --dataset=power  --score=MLE --distn=HomoskedasticNormal
python3 -m examples.empirical.regression --dataset=energy  --score=MLE --distn=HomoskedasticNormal
python3 -m examples.empirical.regression --dataset=yacht  --score=MLE --distn=HomoskedasticNormal
python3 -m examples.empirical.regression --dataset=housing  --score=MLE --distn=HomoskedasticNormal


python3 -m examples.empirical.regression --dataset=concrete  --score=CRPS --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=wine  --score=CRPS --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=kin8nm  --score=CRPS --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=naval  --score=CRPS --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=power  --score=CRPS --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=energy  --score=CRPS --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=yacht  --score=CRPS --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=housing  --score=CRPS --natural --distn=Laplace

python3 -m examples.empirical.regression --dataset=concrete  --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=wine  --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=kin8nm  --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=naval  --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=power  --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=energy  --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=yacht  --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=housing  --score=MLE --natural --distn=Laplace
