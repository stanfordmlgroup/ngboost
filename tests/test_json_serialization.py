"""Tests for JSON and UBJ serialization of NGBoost models."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import MultivariateNormal

try:
    import ubjson  # noqa: F401
except ImportError:
    ubjson = None


@pytest.fixture(name="simple_regressor")
def fixture_simple_regressor(california_housing_data):
    """Create a simple fitted regressor for testing."""
    X_train, _, Y_train, _ = california_housing_data
    ngb = NGBRegressor(verbose=False, n_estimators=10)
    ngb.fit(X_train, Y_train)
    return ngb, X_train


@pytest.fixture(name="simple_classifier")
def fixture_simple_classifier(breast_cancer_data):
    """Create a simple fitted classifier for testing."""
    X_train, _, Y_train, _ = breast_cancer_data
    ngb = NGBClassifier(verbose=False, n_estimators=10)
    ngb.fit(X_train, Y_train)
    return ngb, X_train


@pytest.fixture(name="simple_survival")
def fixture_simple_survival(california_housing_survival_data):
    """Create a simple fitted survival model for testing."""
    X_train, _, T_train, E_train, _ = california_housing_survival_data
    ngb = NGBSurvival(verbose=False, n_estimators=10)
    ngb.fit(X_train, T_train, E_train)
    return ngb, X_train


def test_to_dict_regressor(simple_regressor):
    """Test that to_dict() works for regressor."""
    ngb, X_train = simple_regressor

    model_dict = ngb.to_dict()

    assert "version" in model_dict
    assert "model_type" in model_dict
    assert model_dict["model_type"] == "NGBRegressor"
    assert "base_models" in model_dict
    assert "scalings" in model_dict
    assert "col_idxs" in model_dict
    assert "init_params" in model_dict
    assert len(model_dict["base_models"]) == len(ngb.base_models)


def test_from_dict_regressor(simple_regressor):
    """Test that from_dict() reconstructs regressor correctly."""
    ngb, X_train = simple_regressor

    # Get original predictions
    original_preds = ngb.predict(X_train)
    original_dists = ngb.pred_dist(X_train)

    # Serialize and deserialize
    model_dict = ngb.to_dict()
    ngb_loaded = NGBRegressor.from_dict(model_dict)

    # Check predictions match
    loaded_preds = ngb_loaded.predict(X_train)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)

    # Check distribution parameters match
    loaded_dists = ngb_loaded.pred_dist(X_train)
    # Compare params dict values
    if isinstance(original_dists.params, dict):
        for key in original_dists.params:
            np.testing.assert_array_almost_equal(
                original_dists.params[key], loaded_dists.params[key], decimal=5
            )
    else:
        np.testing.assert_array_almost_equal(
            original_dists.params, loaded_dists.params, decimal=5
        )


def test_save_load_json_regressor(simple_regressor):
    """Test save_json() and load_json() for regressor."""
    ngb, X_train = simple_regressor

    original_preds = ngb.predict(X_train)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        ngb.save_json(filepath)

        ngb_loaded = NGBRegressor.load_json(filepath)
        loaded_preds = ngb_loaded.predict(X_train)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)
    finally:
        Path(filepath).unlink()


def test_to_dict_classifier(simple_classifier):
    """Test that to_dict() works for classifier."""
    ngb, X_train = simple_classifier

    model_dict = ngb.to_dict()

    assert model_dict["model_type"] == "NGBClassifier"
    assert "base_models" in model_dict


def test_from_dict_classifier(simple_classifier):
    """Test that from_dict() reconstructs classifier correctly."""
    ngb, X_train = simple_classifier

    original_preds = ngb.predict(X_train)
    original_proba = ngb.predict_proba(X_train)

    model_dict = ngb.to_dict()
    ngb_loaded = NGBClassifier.from_dict(model_dict)

    loaded_preds = ngb_loaded.predict(X_train)
    loaded_proba = ngb_loaded.predict_proba(X_train)

    np.testing.assert_array_equal(original_preds, loaded_preds)
    np.testing.assert_array_almost_equal(original_proba, loaded_proba, decimal=5)


def test_to_dict_survival(simple_survival):
    """Test that to_dict() works for survival model."""
    ngb, X_train = simple_survival

    model_dict = ngb.to_dict()

    assert model_dict["model_type"] == "NGBSurvival"
    assert "base_models" in model_dict


def test_from_dict_survival(simple_survival):
    """Test that from_dict() reconstructs survival model correctly."""
    ngb, X_train = simple_survival

    original_preds = ngb.predict(X_train)

    model_dict = ngb.to_dict()
    ngb_loaded = NGBSurvival.from_dict(model_dict)

    loaded_preds = ngb_loaded.predict(X_train)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)


def test_exclude_non_essential(simple_regressor):
    """Test that include_non_essential=False excludes optional data."""
    ngb, _ = simple_regressor

    # Force computation of feature_importances_
    _ = ngb.feature_importances_

    model_dict_with = ngb.to_dict(include_non_essential=True)
    model_dict_without = ngb.to_dict(include_non_essential=False)

    assert "feature_importances_" in model_dict_with
    assert "feature_importances_" not in model_dict_without


def test_multivariate_normal(california_housing_survival_data):
    """Test serialization with MultivariateNormal distribution."""
    X_surv_train, _, T_surv_train, E_surv_train, _ = california_housing_survival_data

    ngb = NGBRegressor(Dist=MultivariateNormal(2), n_estimators=10, verbose=False)
    Y_mvn = np.vstack((T_surv_train, E_surv_train)).T
    ngb.fit(X_surv_train, Y_mvn)

    original_preds = ngb.predict(X_surv_train)

    model_dict = ngb.to_dict()
    ngb_loaded = NGBRegressor.from_dict(model_dict)

    loaded_preds = ngb_loaded.predict(X_surv_train)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)


def test_ubj_serialization(simple_regressor):
    """Test UBJ serialization if ubjson is available."""
    if ubjson is None:
        pytest.skip("ubjson package not available")

    ngb, X_train = simple_regressor

    original_preds = ngb.predict(X_train)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".ubj", delete=False) as f:
        filepath = f.name

    try:
        ngb.save_ubj(filepath)

        ngb_loaded = NGBRegressor.load_ubj(filepath)
        loaded_preds = ngb_loaded.predict(X_train)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)
    finally:
        Path(filepath).unlink()


def test_ubj_import_error(simple_regressor):
    """Test that save_ubj raises ImportError when ubjson is not available."""
    # pylint: disable=import-outside-toplevel
    import ngboost.ngboost as ngb_module

    ngb, _ = simple_regressor

    # Temporarily disable UBJSON
    original_available = ngb_module.UBJSON_AVAILABLE
    ngb_module.UBJSON_AVAILABLE = False

    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".ubj", delete=False) as f:
            filepath = f.name

        try:
            with pytest.raises(ImportError, match="ubjson"):
                ngb.save_ubj(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)
    finally:
        ngb_module.UBJSON_AVAILABLE = original_available


def test_json_file_size(simple_regressor):
    """Test that excluding non-essential data reduces file size."""
    ngb, _ = simple_regressor

    # Force computation of feature_importances_
    _ = ngb.feature_importances_

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath_with = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath_without = f.name

    try:
        ngb.save_json(filepath_with, include_non_essential=True)
        ngb.save_json(filepath_without, include_non_essential=False)

        size_with = Path(filepath_with).stat().st_size
        size_without = Path(filepath_without).stat().st_size

        # File without non-essential data should be smaller or equal
        assert size_without <= size_with
    finally:
        Path(filepath_with).unlink(missing_ok=True)
        Path(filepath_without).unlink(missing_ok=True)


def test_to_dict_unfitted_model():
    """Test that to_dict() raises error for unfitted model."""
    ngb = NGBRegressor()

    with pytest.raises(ValueError, match="Model must be fitted"):
        ngb.to_dict()


def test_from_dict_missing_keys():
    """Test that from_dict() raises error for invalid dictionary."""
    invalid_dict = {"version": "1.0"}

    with pytest.raises(ValueError, match="missing required keys"):
        NGBRegressor.from_dict(invalid_dict)


def test_from_dict_corrupted_tree(simple_regressor):
    """Test that from_dict() handles corrupted tree data gracefully."""
    ngb, _ = simple_regressor

    model_dict = ngb.to_dict()
    # Corrupt a tree dictionary
    model_dict["base_models"][0][0]["_pickle"] = "invalid_base64"

    with pytest.raises(ValueError, match="Failed to decode"):
        NGBRegressor.from_dict(model_dict)


def test_from_dict_version_check(simple_regressor):
    """Test that from_dict() checks version compatibility."""
    ngb, _ = simple_regressor

    model_dict = ngb.to_dict()
    model_dict["version"] = "2.0"  # Future version

    with pytest.raises(ValueError, match="Unsupported model version"):
        NGBRegressor.from_dict(model_dict)
