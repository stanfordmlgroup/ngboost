"The NGBoost library API"

# pylint: disable=too-many-arguments
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array

from ngboost.distns import (
    Bernoulli,
    ClassificationDistn,
    LogNormal,
    Normal,
    RegressionDistn,
)
from ngboost.distns.utils import SurvivalDistnClass
from ngboost.helpers import Y_from_censored
from ngboost.learners import default_tree_learner
from ngboost.ngboost import NGBoost
from ngboost.scores import LogScore


class NGBRegressor(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost regression models.

    NGBRegressor is a wrapper for the generic NGBoost class that facilitates regression.
    Use this class if you want to predict an outcome that could take an
    infinite number of (ordered) values.

    Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Normal
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner(s) to use in the boosting algorithm.
                            Pass a single instantiated sklearn regressor, e.g.
                            DecisionTreeRegressor(), to use the same learner for
                            every distribution parameter. To use different learners
                            per distribution parameter, pass a list/tuple of
                            instantiated sklearn regressors with length equal to
                            Dist.n_params. The sequence order matches the distribution
                            parameter order; for example, Normal uses
                            [loc_learner, scale_learner].
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
        validation_fraction: Proportion of training data to set
                             aside as validation data for early stopping.
        early_stopping_rounds:      The number of consecutive boosting iterations during which the
                                    loss has to increase before the algorithm stops early.
                                    Set to None to disable early stopping and validation.
                                    None enables running over the full data set.
        n_jobs            : see the ``NGBoost`` base class.

    Output:
        An NGBRegressor object that can be fit.
    """

    # pylint: disable=too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
        n_jobs=None,
    ):
        # Dist is stored unchanged, as scikit-learn's clone requires. A Dist
        # with only censored scores is uncensored in _make_dist_class.
        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
            validation_fraction,
            early_stopping_rounds,
            n_jobs,
        )

        self._estimator_type = "regressor"

    def _make_dist_class(self):
        assert issubclass(
            self.Dist, RegressionDistn
        ), f"{self.Dist.__name__} is not useable for regression."
        if not hasattr(self.Dist, "scores"):
            # Dist has only censored scores, so use them with every event observed
            return self.Dist.uncensor(self.Score)
        return self.Dist

    def __setstate__(self, state_dict):
        # Older pickles of a Dist with only censored scores set this flag. They
        # already store the user's Dist, which _make_dist_class uncensors.
        state_dict.pop("uncensor", None)
        super().__setstate__(state_dict)


class NGBClassifier(ClassifierMixin, NGBoost, BaseEstimator):
    """
    Constructor for NGBoost classification models.

    NGBRegressor is a wrapper for the generic NGBoost class that facilitates classification.
    Use this class if you want to predict an outcome that could take a discrete number of
    (unordered) values.

    Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Bernoulli
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner(s) to use in the boosting algorithm.
                            Pass a single instantiated sklearn regressor, e.g.
                            DecisionTreeRegressor(), to use the same learner for
                            every distribution parameter. To use different learners
                            per distribution parameter, pass a list/tuple of
                            instantiated sklearn regressors with length equal to
                            Dist.n_params. The sequence order matches the distribution
                            parameter order; for example, Normal uses
                            [loc_learner, scale_learner].
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
        validation_fraction: Proportion of training data to set
                             aside as validation data for early stopping.
        early_stopping_rounds:      The number of consecutive boosting iterations during which the
                                    loss has to increase before the algorithm stops early.
                                    Set to None to disable early stopping and validation.
                                    None enables running over the full data set.
        n_jobs            : see the ``NGBoost`` base class.
    Output:
        An NGBClassifier object that can be fit.
    """

    # pylint: disable=too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        Dist=Bernoulli,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
        n_jobs=None,
    ):
        assert issubclass(
            Dist, ClassificationDistn
        ), f"{Dist.__name__} is not useable for classification."
        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
            validation_fraction,
            early_stopping_rounds,
            n_jobs,
        )
        self._estimator_type = "classifier"

    def _fit_label_encoder(self, Y):
        le = LabelEncoder().fit(Y)
        self._le = le  # pylint: disable=attribute-defined-outside-init
        self.classes_ = le.classes_  # pylint: disable=attribute-defined-outside-init
        n_classes = self.Dist.n_params + 1
        if len(self.classes_) != n_classes:
            raise ValueError(
                "NGBClassifier Dist expects "
                f"{n_classes} classes, got {len(self.classes_)}."
            )

    def __setstate__(
        self, state_dict
    ):  # pylint: disable=attribute-defined-outside-init
        super().__setstate__(state_dict)
        if not hasattr(self, "classes_"):
            self.classes_ = np.arange(self.Dist.n_params + 1)
        if not hasattr(self, "_le"):
            self._le = LabelEncoder()
            self._le.classes_ = self.classes_

    # fit and partial_fit pass their arguments on to NGBoost unchanged, so
    # pylint reads the calls as copies of the one in NGBoost.fit.
    # pylint: disable=too-many-positional-arguments,duplicate-code
    def fit(
        self,
        X,
        Y,
        X_val=None,
        Y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
    ):
        # Learn the classes on every fit. NGBoost.fit then calls partial_fit,
        # which encodes Y and Y_val with them.
        self._fit_label_encoder(Y)
        super().fit(
            X,
            Y,
            X_val=X_val,
            Y_val=Y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )
        return self

    # pylint: disable=too-many-positional-arguments
    def partial_fit(
        self,
        X,
        Y,
        X_val=None,
        Y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
    ):
        if not hasattr(self, "classes_"):
            self._fit_label_encoder(Y)
        Y = self._le.transform(Y)
        if Y_val is not None:
            Y_val = self._le.transform(Y_val)
        super().partial_fit(
            X,
            Y,
            X_val=X_val,
            Y_val=Y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )
        return self

    def predict(self, X, max_iter=None):
        return self.classes_[super().predict(X, max_iter=max_iter)]

    def staged_predict(self, X, max_iter=None):
        return [
            self.classes_[pred] for pred in super().staged_predict(X, max_iter=max_iter)
        ]

    def score(self, X, Y):  # for sklearn
        # Call NGBoost.score directly: super() would reach ClassifierMixin.score,
        # which returns accuracy.
        return NGBoost.score(self, X, self._le.transform(Y))

    def predict_proba(self, X, max_iter=None):
        """
        Probability prediction of Y at the points X=x

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : get the prediction at the specified number of boosting iterations

        Output:
            Numpy array of the estimates of P(Y=k|X=x). Will have shape (n, K)
        """
        return self.pred_dist(X, max_iter=max_iter).class_probs()

    def staged_predict_proba(self, X, max_iter=None):
        """
        Probability prediction of Y at the points X=x at multiple boosting iterations

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : largest number of boosting iterations to get the prediction for

        Output:
            A list of of the estimates of P(Y=k|X=x) of shape (n, K),
            one per boosting stage up to max_iter
        """
        return [
            dist.class_probs() for dist in self.staged_pred_dist(X, max_iter=max_iter)
        ]


class NGBSurvival(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost survival models.

    NGBSurvival is a wrapper for the generic NGBoost class that facilitates survival analysis.
    Use this class if you want to predict an outcome that could take an infinite number of
    (ordered) values, but right-censoring is present in the observed data.

     Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. LogNormal
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner(s) to use in the boosting algorithm.
                            Pass a single instantiated sklearn regressor, e.g.
                            DecisionTreeRegressor(), to use the same learner for
                            every distribution parameter. To use different learners
                            per distribution parameter, pass a list/tuple of
                            instantiated sklearn regressors with length equal to
                            Dist.n_params. The sequence order matches the distribution
                            parameter order; for example, Normal uses
                            [loc_learner, scale_learner].
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
        validation_fraction: Proportion of training data to set
                             aside as validation data for early stopping.
        early_stopping_rounds:      The number of consecutive boosting iterations during which the
                                    loss has to increase before the algorithm stops early.
                                    Set to None to disable early stopping and validation.
                                    None enables running over the full data set.
        n_jobs            : see the ``NGBoost`` base class.
    Output:
        An NGBSurvival object that can be fit.
    """

    # pylint: disable=too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        Dist=LogNormal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
        n_jobs=None,
    ):
        # Dist is stored unchanged, as scikit-learn's clone requires. The class
        # with the censored scores is built from it in _make_dist_class.
        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
            validation_fraction,
            early_stopping_rounds,
            n_jobs,
        )

    def _make_dist_class(self):
        assert issubclass(
            self.Dist, RegressionDistn
        ), f"{self.Dist.__name__} is not useable for regression."
        if not hasattr(self.Dist, "censored_scores"):
            raise ValueError(
                f"The {self.Dist.__name__} distribution does not have any "
                "censored scores implemented."
            )
        return SurvivalDistnClass(self.Dist)

    def __setstate__(self, state_dict):
        # Older pickles store the user's Dist under _basedist
        if "_basedist" in state_dict:
            state_dict["Dist"] = state_dict.pop("_basedist")
        super().__setstate__(state_dict)

    def score(self, X, T, E=None):  # pylint: disable=arguments-renamed
        """
        Score the predicted distributions at the points X=x against the censored
        outcomes. Higher is better: as in NGBoost.score, this is the negative of
        the average (censored) scoring rule.

        scikit-learn model selection calls score(X, T) without E by default. To
        pass E, call sklearn.set_config(enable_metadata_routing=True) and
        request E with set_fit_request(E=True) and set_score_request(E=True).

        Parameters:
            X : DataFrame object or List or
                numpy array of predictors (n x p) in numeric format
            T : numpy array of times to event or censoring (n), or the
                structured array returned by ngboost.helpers.Y_from_censored
            E : numpy array of event indicators (n).
                E[i] = 1 <=> T[i] is the time of an event, else censoring time.
                Required unless T is a structured array.

        Output:
            A float, the negative of the average censored score
        """
        if E is None and np.asarray(T).dtype.names is None:
            raise ValueError(
                "NGBSurvival.score needs the event indicators E. scikit-learn model "
                "selection passes E to score only with metadata routing: call "
                "sklearn.set_config(enable_metadata_routing=True), then "
                "set_fit_request(E=True) and set_score_request(E=True) on the estimator."
            )
        return super().score(X, Y_from_censored(T, E))

    # pylint: disable=too-many-positional-arguments
    def fit(self, X, T, E, X_val=None, T_val=None, E_val=None, **kwargs):
        """Fits an NGBoost survival model to the data.
        For additional parameters see ngboost.NGboost.fit

        Parameters:
            X                     : DataFrame object or List or
                                    numpy array of predictors (n x p) in Numeric format
            T                     : DataFrame object or List or
                                    numpy array of times to event or censoring (n) (floats).
            E                     : DataFrame object or List or
                                    numpy array of event indicators (n).
                                    E[i] = 1 <=> T[i] is the time of an event, else censoring time
            T_val                 : DataFrame object or List or
                                    validation-set times, in numeric format if any
            E_val                 : DataFrame object or List or
                                    validation-set event idicators, in numeric format if any
        """

        X = check_array(X, accept_sparse=True)

        if X_val is not None:
            X_val = check_array(X_val, accept_sparse=True)

        return super().fit(
            X,
            Y_from_censored(T, E),
            X_val=X_val,
            Y_val=Y_from_censored(T_val, E_val),
            **kwargs,
        )
