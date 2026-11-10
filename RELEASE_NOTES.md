# RELEASE NOTES

## Version 0.6.0

### Breaking changes

* Drop support for Python 3.9, which reached end-of-life in October 2025. The minimum supported version is now Python 3.10
* `NGBoost.score` now returns the negative of the average scoring rule (the mean log-likelihood for `LogScore`), so that higher is better as scikit-learn expects. Previously, `GridSearchCV` and `cross_val_score` without an explicit `scoring` argument selected the worst model (issue #418, PR #423)
* Base learners are now seeded from the model `random_state`, so fits are reproducible and independent of `n_jobs`. Output for the default base learner shifts relative to 0.5.x (PR #417)

### New features

* Add optional `n_jobs` parameter to fit base learners in parallel with threads (PR #416)
* Support a distinct base learner for each distribution parameter, including scikit-learn nested parameters and feature importances for all-tree configurations (issue #338, PR #411)
* Add `validation_fraction` and `early_stopping_rounds` to `NGBClassifier` and `NGBSurvival` for API parity with `NGBRegressor` (issue #402, PR #407)
* Report fitted status to scikit-learn's `check_is_fitted` via `__sklearn_is_fitted__`, fixing `NotFittedError` in `Pipeline.predict` on sklearn >= 1.8 (PR #429)
* Set scikit-learn classifier metadata on `NGBClassifier` so that scikit-learn correctly identifies it as a classifier (PR #413)
* Handle array-like sample weights positionally and validate them at fit time; weights with wrong length, shape, NaN, or inf now raise `ValueError` (PR #425)
* `MultivariateNormal` `params` now gives the covariance matrices under the key `cov`, the name that `scipy.stats.multivariate_normal` uses. The old key `scale` is deprecated: `params['scale']` and `params.get('scale')` still return the covariance, but give a `FutureWarning`. Everywhere else in ngboost `scale` is a standard deviation, so users read the covariance as one. `scale` is no longer in `params.keys()` (issue #168, PR #438)
* Document saving and loading models with `skops` and test the round trip (PR #427)
* Add `mu` and `sigma` to `LogNormal`, the mean and standard deviation of log(Y). `LogNormal.loc` and `LogNormal.scale` are deprecated and give a `FutureWarning`. `LogNormal.scale` was sigma, but `params['scale']` is exp(mu) as in `scipy.stats.lognorm`, and users mixed the two up. `params` does not change (issue #168, PR #433)
* Add the `NegativeBinomial` distribution for overdispersed counts, built with the SymPy factory. Its parameters are the mean `mu` and the dispersion `n`, and its variance is `mu + mu**2 / n` (issue #281, PR #446)
* `make_distribution` accepts expressions as values of `scipy_kwarg_map`, for distributions that scipy parameterizes differently, such as `p = n / (n + mu)` for `scipy.stats.nbinom` (PR #446)

### Bug fixes

* Fix the `CRPScore` metric of `Normal`, `NormalFixedMean`, and `LogNormal`, which was off by a factor of the predicted scale. Fits using `CRPScore` with these distributions now produce correct natural gradients (issue #421, PR #424)
* Fix the scale gradient of the `Laplace` `CRPScore`, which dropped a `0.75 * scale` term (PR #419)
* Fix censored scores and gradients for `Exponential` and `LogNormal`, which swapped censored and observed rows and had sign and extra-term errors (PR #420)
* Fix the log-scale Fisher information of `TFixedDf` and `Cauchy`, which was off by a factor of `4 * var` (PR #422)
* Raise `AttributeError` for an attribute that a distribution does not have. Before, the distributions built on `scipy.stats` returned `None`, so a typo such as `dist.s` failed silently. Predicted distributions can now be pickled and deep-copied (issue #168, PR #433)
* Reset validation iteration metadata on refit so that re-calling `fit` on the same model behaves identically to fitting a fresh one (PR #426)
* Keep `random_state` as given and seed a fresh generator in `fit`, so that `get_params` returns the original value and clones behave identically (PR #417)
* Replace the deprecated `friedman_mse` criterion with `squared_error` for the default tree learner (issue #408, PR #409)
* Stop the overflow warnings from trial steps that the line search rejects, such as `overflow encountered in exp` in `Normal`. They came most often from `natural_gradient=False` on a target with large units. Fitted models do not change (issue #352, PR #434)
* `fit` now raises a `ValueError` that says the fit diverged when the gradient has non-finite values, in place of the base learner's `Input y contains NaN` error (PR #434)
* Fix saving models that use a distribution made with the SymPy factory (`Beta`, `BetaBernoulli`, `BetaBinomial`, `BetaBinomialEstN`, `LogitNormal` or a user's own) with pickle, joblib and skops. The generated classes reported `ngboost.distns.sympy_utils` as their module, so saving failed with a `PicklingError`. `make_distribution` and `make_sympy_log_score` now set the module of a class to the module that creates it, and take a new `module` argument (PR #445)
* Fix `sample(m)` of the SymPy factory distributions that use a scipy class, such as `Beta`. It failed for more than one row, and now returns one draw for each row in each of the `m` samples, as the other distributions do (PR #446)

### Documentation and CI

* Build the user guide with Jupyter Book 2 in GitHub Actions and deploy on push to master (PR #428)
* Remove the old prebuilt docs site generated by jupyter-book 0.x in 2020 (PR #430)
* Run user guide notebooks during the docs build with `--execute`, replacing removed `load_boston` calls with `load_diabetes` (PR #431)
* Mark observed rows as events in the survival examples (PR #436)
* Add a section on overflow warnings and diverging fits to the tuning page of the user guide (PR #434)
* Run the test matrix once per commit and cancel stale PR runs via concurrency groups (PR #444)
* Fix scikit-learn wheel resolution on Python 3.14 (PR #414)

## Version 0.5.10

* Add `load_ngboost_model` compatibility loader for models saved with scikit-learn < 1.3 and loaded under newer scikit-learn versions (issue #389)
* Backfill `missing_go_to_left` during load and restore trees to standard sklearn `Tree` instances after deserialization
* Add targeted regression tests for old-style tree-node pickles and helper-level tests for `Y_from_censored` and compatibility loading paths in `tests/test_helpers.py`

## Version 0.5.9

* Add SymPy-powered distribution factory support for defining LogScore and NGBoost distributions from symbolic expressions or `sympy.stats` distributions
* Add new built-in distributions via the factory, including Beta, Beta-Bernoulli, Beta-Binomial, and Logit-Normal
* Add support for Python 3.14 and update CI matrix to test Python 3.14
* Bump development version to `0.5.9dev`

## Version 0.5.8

* Fix numpy2 natural gradient compatibility

## Version 0.5.7

* Support for Python 3.13
* Update to dev dependencies

## Version 0.5.6

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
