#!/usr/bin/bash
python3 -m comparison.sharpness --alpha=1.0 --score=CRPS --verbose
python3 -m comparison.sharpness --alpha=0.9 --score=CRPS --verbose
python3 -m comparison.sharpness --alpha=0.8 --score=CRPS --verbose
python3 -m comparison.sharpness --alpha=0.7 --score=CRPS --verbose
python3 -m comparison.sharpness --alpha=0.6 --score=CRPS --verbose
python3 -m comparison.sharpness --alpha=0.5 --score=CRPS --verbose