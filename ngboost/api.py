"The NGBoost library API"
# pylint: disable=too-many-arguments
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from ngboost.distns import (
    Bernoulli,
    ClassificationDistn,
    Normal,
    RegressionDistn,
)
from ngboost.learners import default_tree_learner
from ngboost.ngboost import NGBoost
from ngboost.scores import LogScore
from ngboost.censored import CensoredOutcome


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
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
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
    Output:
        An NGBRegressor object that can be fit.
    """

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
        censored=False,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):
        assert issubclass(
            Dist, RegressionDistn
        ), f"{Dist.__name__} is not useable for regression."

        super().__init__(
            Dist=Dist,
            Score=Score,
            Base=Base,
            natural_gradient=natural_gradient,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            censored=censored,
            verbose=verbose,
            verbose_eval=verbose_eval,
            tol=tol,
            random_state=random_state,
        )


class NGBCensored(NGBRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, censored=True)

    def check_X_y(self, X, Y):
        X = check_array(X)
        assert (
            type(Y) is CensoredOutcome
        ), f"Y must be a ngboost.censored.CensoredOutcome object"

        return X, Y


class NGBClassifier(NGBoost, BaseEstimator):
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
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
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
    Output:
        An NGBRegressor object that can be fit.
    """

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
    ):
        assert issubclass(
            Dist, ClassificationDistn
        ), f"{Dist.__name__} is not useable for classification."
        super().__init__(
            Dist=Dist,
            Score=Score,
            Base=Base,
            natural_gradient=natural_gradient,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            verbose=verbose,
            verbose_eval=verbose_eval,
            tol=tol,
            random_state=random_state,
        )

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
