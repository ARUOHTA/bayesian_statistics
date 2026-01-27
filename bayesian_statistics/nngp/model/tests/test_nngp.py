"""Tests for NNGP factor computation."""
import numpy as np
import pytest
from bayesian_statistics.nngp.model.nngp import build_nngp_factors, build_cross_factors
from bayesian_statistics.nngp.model.sample import LocalNNGPKernel


class TestBuildNNGPFactors:
    """Test build_nngp_factors function."""

    def test_basic_functionality(self):
        """Test that factors are computed correctly."""
        np.random.seed(42)
        coords = np.random.randn(20, 2)
        kernel = LocalNNGPKernel(lengthscale=0.5, variance=1.0)
        factors, order = build_nngp_factors(coords, M=5, kernel=kernel)

        assert len(factors.neighbor_idx) == 20
        assert len(factors.a_rows) == 20
        assert factors.d.shape == (20,)
        assert np.all(factors.d >= 0)

    def test_performance(self):
        """Test that computation is fast (scipy.linalg)."""
        import time
        np.random.seed(42)
        coords = np.random.randn(50, 2)
        kernel = LocalNNGPKernel(lengthscale=0.5, variance=1.0)

        t0 = time.time()
        factors, _ = build_nngp_factors(coords, M=10, kernel=kernel)
        elapsed = time.time() - t0

        # Should complete in < 1 second (with scipy.linalg)
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"


class TestBuildCrossFactors:
    """Test build_cross_factors function."""

    def test_basic_functionality(self):
        """Test that cross factors are computed correctly."""
        np.random.seed(42)
        prior_coords = np.random.randn(20, 2)
        target_coords = np.random.randn(10, 2)
        kernel = LocalNNGPKernel(lengthscale=0.5, variance=1.0)

        factors = build_cross_factors(prior_coords, target_coords, M=5, kernel=kernel)

        assert len(factors.neighbor_idx) == 10
        assert len(factors.a_rows) == 10
        assert factors.d.shape == (10,)
        assert np.all(factors.d >= 0)

    def test_performance(self):
        """Test that computation is fast (scipy.linalg)."""
        import time
        np.random.seed(42)
        prior_coords = np.random.randn(50, 2)
        target_coords = np.random.randn(100, 2)
        kernel = LocalNNGPKernel(lengthscale=0.5, variance=1.0)

        t0 = time.time()
        factors = build_cross_factors(prior_coords, target_coords, M=10, kernel=kernel)
        elapsed = time.time() - t0

        # Should complete in < 1 second (with scipy.linalg)
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"
