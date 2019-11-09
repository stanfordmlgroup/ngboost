import numpy as np
from shap import *
from shap.explainers.tree import TreeEnsemble, TreeExplainer, Tree, feature_dependence_codes
from shap.common import DenseData


class NGBoostEnsemble(TreeEnsemble):
    """ An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """
    def __init__(self, model, param_idx=0 ,data=None, data_missing=None):
        self.model_type = "internal"
        self.trees = None
        less_than_or_equal = True
        self.base_offset = 0
        self.objective = None # what we explain when explaining the loss of the model
        self.tree_output = None # what are the units of the values in the leaves of the trees
        self.internal_dtype = np.float64
        self.input_dtype = np.float64 # for sklearn we need to use np.float32 to always get exact matches to their predictions
        self.data = data
        self.data_missing = data_missing
        self.fully_defined_weighting = True # does the background dataset land in every leaf (making it valid for the tree_path_dependent method)
        self.tree_limit = None # used for limiting the number of trees we use by default (like from early stopping) 

        # we use names like keras
        objective_name_map = {
            "mse": "squared_error",
            "friedman_mse": "squared_error",
            "reg:linear": "squared_error",
            "regression": "squared_error",
            "regression_l2": "squared_error",
            "mae": "absolute_error",
            "gini": "binary_crossentropy",
            "entropy": "binary_crossentropy",
            "binary:logistic": "binary_crossentropy",
            "binary_logloss": "binary_crossentropy",
            "binary": "binary_crossentropy"
        }

        if str(type(model)).endswith(("ngboost.ngboost.NGBoost'>","ngboost.sklearn_api.NGBRegressor'>","ngboost.sklearn_api.NGBClassifier'>")):
            assert model.base_models, "Model has empty `base_models`! Have you called `model.fit`?"
            assert str(type(model.base_models[0][param_idx])).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"), "You must use default_tree_learner!"
            self.internal_dtype = model.base_models[0][param_idx].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = - model.learning_rate * np.array(model.scalings) # output is average of trees
            self.trees = [Tree(e[param_idx].tree_, scaling=s, data=data, data_missing=data_missing) for e,s in zip(model.base_models,scaling)]
            self.objective = objective_name_map.get(model.base_models[0][param_idx].criterion, None)
            self.tree_output = "raw_value"
        else:
            raise Exception("Model type not yet supported by TreeExplainer: " + str(type(model)))

                # build a dense numpy version of all the tree objects
        if self.trees is not None and self.trees:
            max_nodes = np.max([len(t.values) for t in self.trees])
            assert len(np.unique([t.values.shape[1] for t in self.trees])) == 1, "All trees in the ensemble must have the same output dimension!"
            ntrees = len(self.trees)
            self.n_outputs = self.trees[0].values.shape[1]

            # important to be -1 in unused sections!! This way we can tell which entries are valid.
            self.children_left = -np.ones((ntrees, max_nodes), dtype=np.int32)
            self.children_right = -np.ones((ntrees, max_nodes), dtype=np.int32)
            self.children_default = -np.ones((ntrees, max_nodes), dtype=np.int32)
            self.features = -np.ones((ntrees, max_nodes), dtype=np.int32)

            self.thresholds = np.zeros((ntrees, max_nodes), dtype=self.internal_dtype)
            self.values = np.zeros((ntrees, max_nodes, self.trees[0].values.shape[1]), dtype=self.internal_dtype)
            self.node_sample_weight = np.zeros((ntrees, max_nodes), dtype=self.internal_dtype)
            
            for i in range(ntrees):
                l = len(self.trees[i].features)
                self.children_left[i,:l] = self.trees[i].children_left
                self.children_right[i,:l] = self.trees[i].children_right
                self.children_default[i,:l] = self.trees[i].children_default
                self.features[i,:l] = self.trees[i].features
                self.thresholds[i,:l] = self.trees[i].thresholds
                self.values[i,:l,:] = self.trees[i].values
                self.node_sample_weight[i,:l] = self.trees[i].node_sample_weight

                # ensure that the passed background dataset lands in every leaf
                if np.min(self.trees[i].node_sample_weight) <= 0:
                    self.fully_defined_weighting = False
            
            self.num_nodes = np.array([len(t.values) for t in self.trees], dtype=np.int32)
            self.max_depth = np.max([t.max_depth for t in self.trees])


class TreeExplainer(TreeExplainer):
    """Uses Tree SHAP algorithms to explain the output of ensemble tree models.

    Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles of trees,
    under several different possible assumptions about feature dependence. It depends on fast C++
    implementations either inside an externel model package or in the local compiled C extention.

    Parameters
    ----------
    model : model object
        The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.

    data : numpy.array or pandas.DataFrame
        The background dataset to use for integrating out features. This argument is optional when
        feature_dependence="tree_path_dependent", since in that case we can use the number of training
        samples that went down each tree path as our background dataset (this is recorded in the model object).

    param_idx: integer
        The index of parameter.

    feature_dependence : "tree_path_dependent" (default) or "independent"
        Since SHAP values rely on conditional expectations we need to decide how to handle correlated
        (or otherwise dependent) input features. The default "tree_path_dependent" approach is to just
        follow the trees and use the number of training examples that went down each leaf to represent
        the background distribution. This approach repects feature dependecies along paths in the trees.
        However, for non-linear marginal transforms (like explaining the model loss)  we don't yet
        have fast algorithms that respect the tree path dependence, so instead we offer an "independent"
        approach that breaks the dependencies between features, but allows us to explain non-linear
        transforms of the model's output. Note that the "independent" option requires a background
        dataset and its runtime scales linearly with the size of the background dataset you use. Anywhere
        from 100 to 1000 random background samples are good sizes to use.
    
    model_output : "margin", "probability", or "log_loss"
        What output of the model should be explained. If "margin" then we explain the raw output of the
        trees, which varies by model (for binary classification in XGBoost this is the log odds ratio).
        If "probability" then we explain the output of the model transformed into probability space
        (note that this means the SHAP values now sum to the probability output of the model). If "log_loss"
        then we explain the log base e of the model loss function, so that the SHAP values sum up to the
        log loss of the model for each sample. This is helpful for breaking down model performance by feature.
        Currently the probability and log_loss options are only supported when feature_dependence="independent".
    """
    def __init__(self, model, param_idx=0 ,data = None, model_output = "margin", feature_dependence = "tree_path_dependent"):
        assert param_idx < model.Dist.n_params, "No such parameter index!"

        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            self.data = data.values
        elif isinstance(data, DenseData):
            self.data = data.data
        else:
            self.data = data
        self.data_missing = None if self.data is None else np.isnan(self.data)
        self.model_output = model_output
        self.feature_dependence = feature_dependence
        self.expected_value = None
        self.model = NGBoostEnsemble(model, param_idx, self.data, self.data_missing)

        assert feature_dependence in feature_dependence_codes, "Invalid feature_dependence option!"

        # check for unsupported combinations of feature_dependence and model_outputs
        if feature_dependence == "tree_path_dependent":
            assert model_output == "margin", "Only margin model_output is supported for feature_dependence=\"tree_path_dependent\""
        else:   
            assert data is not None, "A background dataset must be provided unless you are using feature_dependence=\"tree_path_dependent\"!"

        if model_output != "margin":
            if self.model.objective is None and self.model.tree_output is None:
                raise Exception("Model does not have a known objective or output type! When model_output is " \
                                "not \"margin\" then we need to know the model's objective or link function.")

        # compute the expected value if we have a parsed tree for the cext
        if self.model_output == "logloss":
            self.expected_value = self.__dynamic_expected_value
        elif data is not None:
            self.expected_value = self.model.predict(self.data, output=model_output).mean(0)
            if hasattr(self.expected_value, '__len__') and len(self.expected_value) == 1:
                self.expected_value = self.expected_value[0]
        elif hasattr(self.model, "node_sample_weight"):
            self.expected_value = self.model.values[:,0].sum(0)
            if self.expected_value.size == 1:
                self.expected_value = self.expected_value[0]
            self.expected_value += self.model.base_offset
        