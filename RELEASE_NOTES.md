# RELEASE NOTES

## Version 0.5.5

* Add support for Weibull and HalfNormal Distributions

## Version 0.5.5

* Upgrade sklearn > 1.6

## Version 0.5.4

* Update to partial fit to respect validation data

## Version 0.5.3

* Allow NAN as input
* Poetry update

## Version 0.5.2

* Support for Numpy 2.0
* Value error fix
* Linting updates

## Version 0.5.1

* Adds support for NormalFixedMean distribution
* Updates to makefile for easier publishing

## Version 0.5.0

* Drops support for python 3.7 and 3.8
* Now supports Python 3.11 and 3.12
* Fixed issue with np.bool
* Optimized memory usage in pred-dist
* Removed declared pandas dependency
* Significant improvements to run times on tests during development
* Minor enhancements to github actions

## Version 0.4.2

* Fix deprecated numpy type alias. This was causing a warning with NumPy >=1.20 and an error with NumPy >=1.24
* Remove pandas as a declared dependency

## Version 0.4.1

### Added `partial_fit` method for incremental learning

NGBoost now includes a new `partial_fit` method that allows for incremental learning. This method appends new base models to the existing ones, which can be useful when new data becomes available over time or when the data is too large to fit in memory all at once.

The `partial_fit` method takes similar parameters to the `fit` method, including predictors `X`, outcomes `Y`, and validation sets `X_val` and `Y_val`. It also supports custom weights for the training and validation sets, as well as early stopping and custom loss monitoring.

Please note that the `partial_fit` method is not yet fully tested and may not work as expected in all cases. Use it with caution and thoroughly test its behavior in your specific use case before relying on it in production.

## Version 0.4.0

* Added support for the gamma distribution
* Added sklearn support to `set_params`
* Fixed off-by-one issue for max trees
* Upgraded version of `black` formatter to 22.8.0
