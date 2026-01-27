"""Tests for fixed intensity coefficient functions."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from bayesian_statistics.nngp.model.marked_point_process.intensity import (
    compute_eta_intensity,
    compute_eta_intensity_fixed,
    update_beta_intensity_fixed,
)


class TestComputeEtaIntensityFixed:
    """Test compute_eta_intensity_fixed function."""

    def test_basic_computation(self):
        """eta = W @ beta for fixed coefficients."""
        # Simple case: 3 points, 2 features (intercept + covariate)
        W = np.array([
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, -0.5],
        ])
        beta = np.array([1.0, 2.0])  # intercept=1, slope=2

        eta = compute_eta_intensity_fixed(beta, W)

        # Expected: [1 + 2*0.5, 1 + 2*1.0, 1 + 2*(-0.5)] = [2, 3, 0]
        expected = np.array([2.0, 3.0, 0.0])
        assert_array_almost_equal(eta, expected)

    def test_intercept_only(self):
        """eta = beta[0] for intercept-only model."""
        n = 5
        W = np.ones((n, 1))
        beta = np.array([0.5])

        eta = compute_eta_intensity_fixed(beta, W)

        expected = np.full(n, 0.5)
        assert_array_almost_equal(eta, expected)

    def test_different_from_spatial_version(self):
        """Fixed version differs from spatial version (which uses element-wise product)."""
        n = 3
        p = 2
        W = np.array([
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, -0.5],
        ])
        # Fixed: 1D beta
        beta_fixed = np.array([1.0, 2.0])

        # Spatial: 2D beta (p, n) - different coefficients per site
        beta_spatial = np.array([
            [1.0, 1.0, 1.0],  # intercept same at all sites
            [2.0, 3.0, 4.0],  # slope varies by site
        ])

        eta_fixed = compute_eta_intensity_fixed(beta_fixed, W)
        eta_spatial = compute_eta_intensity(beta_spatial, W)

        # Should be different since spatial has varying slopes
        assert not np.allclose(eta_fixed, eta_spatial)


class TestUpdateBetaIntensityFixed:
    """Test update_beta_intensity_fixed function."""

    def test_output_shape(self):
        """Output beta should have shape (p,)."""
        n = 10
        p = 3
        W = np.hstack([np.ones((n, 1)), np.random.randn(n, p - 1)])
        omega = np.abs(np.random.randn(n)) + 0.1
        kappa = np.random.randn(n)
        rng = np.random.default_rng(42)

        beta_new, eta_new = update_beta_intensity_fixed(
            beta_int=np.zeros(p),
            W=W,
            omega=omega,
            kappa=kappa,
            prior_mean=0.0,
            prior_variance=10.0,
            rng=rng,
        )

        assert beta_new.shape == (p,)
        assert eta_new.shape == (n,)

    def test_eta_consistent_with_beta(self):
        """eta_new should equal W @ beta_new."""
        n = 10
        p = 2
        W = np.hstack([np.ones((n, 1)), np.random.randn(n, 1)])
        omega = np.abs(np.random.randn(n)) + 0.1
        kappa = np.random.randn(n)
        rng = np.random.default_rng(42)

        beta_new, eta_new = update_beta_intensity_fixed(
            beta_int=np.zeros(p),
            W=W,
            omega=omega,
            kappa=kappa,
            prior_mean=0.0,
            prior_variance=10.0,
            rng=rng,
        )

        expected_eta = W @ beta_new
        assert_array_almost_equal(eta_new, expected_eta)

    def test_prior_influence(self):
        """Strong prior should pull beta toward prior mean."""
        n = 5
        p = 1
        W = np.ones((n, 1))
        omega = np.full(n, 0.001)  # Very small omega -> weak data signal
        kappa = np.zeros(n)
        rng = np.random.default_rng(42)

        # Strong prior at mean=5.0, small variance
        beta_new, _ = update_beta_intensity_fixed(
            beta_int=np.zeros(p),
            W=W,
            omega=omega,
            kappa=kappa,
            prior_mean=5.0,
            prior_variance=0.1,
            rng=rng,
        )

        # Beta should be close to prior mean (allowing for sampling variance)
        assert abs(beta_new[0] - 5.0) < 1.0

    def test_data_influence(self):
        """Strong data signal should override prior."""
        n = 100
        p = 1
        W = np.ones((n, 1))
        # Strong data: all y=1 (sites), so kappa = 0.5
        omega = np.ones(n)  # Strong omega
        kappa = np.full(n, 0.5)  # All sites (y=1)
        rng = np.random.default_rng(42)

        # Weak prior at 0, large variance
        beta_new, _ = update_beta_intensity_fixed(
            beta_int=np.zeros(p),
            W=W,
            omega=omega,
            kappa=kappa,
            prior_mean=0.0,
            prior_variance=100.0,
            rng=rng,
        )

        # Beta should be positive (data says y=1)
        assert beta_new[0] > 0

    def test_reproducibility_with_seed(self):
        """Same seed should give same results."""
        n = 10
        p = 2
        W = np.hstack([np.ones((n, 1)), np.random.randn(n, 1)])
        omega = np.abs(np.random.randn(n)) + 0.1
        kappa = np.random.randn(n)

        beta1, _ = update_beta_intensity_fixed(
            beta_int=np.zeros(p),
            W=W,
            omega=omega,
            kappa=kappa,
            prior_mean=0.0,
            prior_variance=10.0,
            rng=np.random.default_rng(42),
        )

        beta2, _ = update_beta_intensity_fixed(
            beta_int=np.zeros(p),
            W=W,
            omega=omega,
            kappa=kappa,
            prior_mean=0.0,
            prior_variance=10.0,
            rng=np.random.default_rng(42),
        )

        assert_array_almost_equal(beta1, beta2)
