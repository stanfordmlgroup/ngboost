#!/usr/bin/env python3
"""
Test for NumPy 2.x compatibility issues with natural gradients.

This test reproduces the issue described in GitHub issue #384 and verifies the fix.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, Normal
from ngboost.manifold import manifold
from ngboost.scores import LogScore


def test_natural_gradient_numpy2_compatibility():
    """Test that natural gradients work with NumPy 2.x"""
    print(f"NumPy version: {np.__version__}")

    # Test classification
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, random_state=42
    )

    model = NGBClassifier(
        Dist=Bernoulli, natural_gradient=True, n_estimators=10, verbose=False
    )

    # This should not raise a dimension mismatch error
    model.fit(X, y)

    # Test predictions
    pred_dist = model.pred_dist(X[:5])
    assert pred_dist is not None
    print("✓ Classification with natural gradient works")


def test_natural_gradient_regression_numpy2_compatibility():
    """Test that natural gradients work with regression on NumPy 2.x"""
    print(f"NumPy version: {np.__version__}")

    # Test regression (unpack 2 values for broad sklearn compatibility)
    X, y = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42
    )

    model = NGBRegressor(
        Dist=Normal, natural_gradient=True, n_estimators=10, verbose=False
    )

    # This should not raise a dimension mismatch error
    model.fit(X, y)

    # Test predictions
    pred_dist = model.pred_dist(X[:5])
    assert pred_dist is not None
    print("✓ Regression with natural gradient works")


def test_natural_gradient_disabled_works():
    """Test that disabling natural gradient works around any issues"""
    print(f"NumPy version: {np.__version__}")

    # Test classification with natural gradient disabled
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, random_state=42
    )

    model = NGBClassifier(
        Dist=Bernoulli, natural_gradient=False, n_estimators=10, verbose=False
    )

    model.fit(X, y)

    # Test predictions
    pred_dist = model.pred_dist(X[:5])
    assert pred_dist is not None
    print("✓ Classification with disabled natural gradient works")


def test_gradient_computation_dimensions():
    """Test that gradient computation handles dimensions correctly"""
    print(f"NumPy version: {np.__version__}")

    # Create a test case
    n_samples = 10
    n_params = 2

    params = np.random.randn(n_params, n_samples)
    Manifold = manifold(LogScore, Normal)
    manifold_dist = Manifold(params)
    y = np.random.randn(n_samples)

    # Test gradient computation
    grad = manifold_dist.grad(y, natural=True)

    # Check dimensions
    assert grad.shape == (n_samples, n_params)
    print(f"✓ Gradient computation works: {grad.shape}")


def test_dimension_mismatch_error_reproduction():
    """Test that reproduces the dimension mismatch error for testing purposes"""
    print(f"NumPy version: {np.__version__}")

    # Create a test case
    n_samples = 5
    n_params = 2

    params = np.random.randn(n_params, n_samples)
    Manifold = manifold(LogScore, Normal)
    manifold_dist = Manifold(params)
    y = np.random.randn(n_samples)

    # Get d_score and metric
    d_score_result = manifold_dist.d_score(y)
    metric_result = manifold_dist.metric()

    # Test the current approach (should work)
    try:
        grad_expanded = d_score_result[..., None]
        result = np.linalg.solve(metric_result, grad_expanded)
        final_grad = result[..., 0]
        print(f"✓ Current approach works: {final_grad.shape}")
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"✗ Current approach fails: {e}")
        if "mismatch in its core dimension" in str(e):
            print("*** FOUND THE DIMENSION MISMATCH ISSUE! ***")
            pytest.fail("Dimension mismatch error found")

    # Test with incorrect dimensions (should fail gracefully)
    try:
        # Create incorrect gradient shape
        wrong_grad = d_score_result.flatten()
        result = np.linalg.solve(metric_result, wrong_grad[..., None])
        print("✗ Unexpected: wrong dimensions worked")
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"✓ Expected failure with wrong dimensions: {e}")
        if "mismatch in its core dimension" in str(e):
            print("✓ Correctly caught dimension mismatch")


def test_large_dataset_natural_gradient():
    """Test natural gradient with larger datasets"""
    print(f"NumPy version: {np.__version__}")

    # Test with larger dataset
    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2, random_state=42
    )

    model = NGBClassifier(
        Dist=Bernoulli, natural_gradient=True, n_estimators=5, verbose=False
    )

    # This should not raise a dimension mismatch error
    model.fit(X, y)

    # Test predictions
    pred_dist = model.pred_dist(X[:10])
    assert pred_dist is not None
    print("✓ Large dataset with natural gradient works")


if __name__ == "__main__":
    print("Testing NumPy 2.x compatibility...")
    print("=" * 60)

    # Run all tests
    test_natural_gradient_numpy2_compatibility()
    test_natural_gradient_regression_numpy2_compatibility()
    test_natural_gradient_disabled_works()
    test_gradient_computation_dimensions()
    test_dimension_mismatch_error_reproduction()
    test_large_dataset_natural_gradient()

    print("\n" + "=" * 60)
    print("All NumPy 2.x compatibility tests passed!")
