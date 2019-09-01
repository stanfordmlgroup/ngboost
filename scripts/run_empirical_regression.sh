#!/bin/sh
# python3 -m examples.empirical.regression --dataset=concrete --verbose --score=CRPS --natura
# python3 -m examples.empirical.regression --dataset=wine --verbose --score=CRPS --natural
# python3 -m examples.empirical.regression --dataset=kin8nm --verbose --score=CRPS --natural
# python3 -m examples.empirical.regression --dataset=naval --verbose --score=CRPS --natural
# python3 -m examples.empirical.regression --dataset=power --verbose --score=CRPS --natural
# python3 -m examples.empirical.regression --dataset=energy --verbose --score=CRPS --natural
# python3 -m examples.empirical.regression --dataset=yacht --verbose --score=CRPS --natural
# python3 -m examples.empirical.regression --dataset=housing --verbose --score=CRPS --natural
#
# python3 -m examples.empirical.regression --dataset=concrete --verbose --score=CRPS --natural --distn=Laplace
# python3 -m examples.empirical.regression --dataset=wine --verbose --score=CRPS --natural --distn=Laplace
# python3 -m examples.empirical.regression --dataset=kin8nm --verbose --score=CRPS --natural --distn=Laplace
# python3 -m examples.empirical.regression --dataset=naval --verbose --score=CRPS --natural --distn=Laplace
# python3 -m examples.empirical.regression --dataset=power --verbose --score=CRPS --natural --distn=Laplace
# python3 -m examples.empirical.regression --dataset=energy --verbose --score=CRPS --natural --distn=Laplace
# python3 -m examples.empirical.regression --dataset=yacht --verbose --score=CRPS --natural --distn=Laplace
# python3 -m examples.empirical.regression --dataset=housing --verbose --score=CRPS --natural --distn=Laplace

python3 -m examples.empirical.regression --dataset=concrete --verbose --score=MLE --natural
python3 -m examples.empirical.regression --dataset=wine --verbose --score=MLE --natural
python3 -m examples.empirical.regression --dataset=kin8nm --verbose --score=MLE --natural
python3 -m examples.empirical.regression --dataset=naval --verbose --score=MLE --natural
python3 -m examples.empirical.regression --dataset=power --verbose --score=MLE --natural
python3 -m examples.empirical.regression --dataset=energy --verbose --score=MLE --natural
python3 -m examples.empirical.regression --dataset=yacht --verbose --score=MLE --natural
python3 -m examples.empirical.regression --dataset=housing --verbose --score=MLE --natural

exit

python3 -m examples.empirical.regression --dataset=concrete --verbose --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=wine --verbose --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=kin8nm --verbose --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=naval --verbose --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=power --verbose --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=energy --verbose --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=yacht --verbose --score=MLE --natural --distn=Laplace
python3 -m examples.empirical.regression --dataset=housing --verbose --score=MLE --natural --distn=Laplace
