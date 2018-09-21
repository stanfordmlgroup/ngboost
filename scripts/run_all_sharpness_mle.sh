#!/usr/bin/bash
python3 -m comparison.sharpness --alpha=1.0 --score=MLE --verbose
python3 -m comparison.sharpness --alpha=0.9 --score=MLE --verbose
python3 -m comparison.sharpness --alpha=0.8 --score=MLE --verbose
python3 -m comparison.sharpness --alpha=0.7 --score=MLE --verbose
python3 -m comparison.sharpness --alpha=0.6 --score=MLE --verbose
python3 -m comparison.sharpness --alpha=0.5 --score=MLE --verbose