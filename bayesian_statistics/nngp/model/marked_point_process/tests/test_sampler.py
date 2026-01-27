"""Tests for MarkedPointProcessSampler with fixed intensity coefficients."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from bayesian_statistics.nngp.model.marked_point_process.config import (
    MarkedPointProcessConfig,
)
from bayesian_statistics.nngp.model.marked_point_process.dataset import (
    MarkedPointProcessDataset,
)
from bayesian_statistics.nngp.model.marked_point_process.sampler import (
    MarkedPointProcessSampler,
)


def create_simple_dataset(n_sites=20, n_categories=3, seed=42):
    """Create a simple dataset for testing."""
    rng = np.random.default_rng(seed)

    # Random site coordinates in [0, 1]^2
    site_coords = rng.uniform(0, 1, size=(n_sites, 2))

    # Random counts for K categories
    counts = rng.poisson(10, size=(n_sites, n_categories))

    # Category names
    origins = [f"category_{i}" for i in range(n_categories)]

    # Simple grid
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    return MarkedPointProcessDataset(
        site_coords=site_coords,
        counts=counts,
        origins=origins,
        grid_coords=grid_coords,
        region=[[0.0, 1.0], [0.0, 1.0]],  # volume = 1.0
    )


class TestFixedIntensityCoefficientsSampler:
    """Test sampler with fixed_intensity_coefficients=True."""

    def test_beta_int_shape_fixed_mode(self):
        """beta_int should have shape (p_int,) in fixed mode, not (p_int, n_sites)."""
        dataset = create_simple_dataset()
        config = MarkedPointProcessConfig(
            n_iter=10,
            burn_in=5,
            fixed_intensity_coefficients=True,
            seed=42,
        )
        sampler = MarkedPointProcessSampler(dataset, config)
        sampler._init_state()

        # In fixed mode, beta_int_sites should be 1D
        p_int = dataset.design_matrix_intensity.shape[1]
        assert sampler.beta_int_sites.shape == (p_int,)

    def test_beta_int_shape_spatial_mode(self):
        """beta_int should have shape (p_int, n_sites) in spatial mode."""
        dataset = create_simple_dataset()
        config = MarkedPointProcessConfig(
            n_iter=10,
            burn_in=5,
            fixed_intensity_coefficients=False,
            seed=42,
        )
        sampler = MarkedPointProcessSampler(dataset, config)
        sampler._init_state()

        # In spatial mode, beta_int_sites should be 2D
        p_int = dataset.design_matrix_intensity.shape[1]
        n_sites = dataset.num_sites()
        assert sampler.beta_int_sites.shape == (p_int, n_sites)

    def test_run_fixed_mode_returns_correct_shape(self):
        """run() in fixed mode should return beta_int_samples with shape (n_saved, p_int)."""
        dataset = create_simple_dataset()
        config = MarkedPointProcessConfig(
            n_iter=20,
            burn_in=10,
            thinning=2,
            fixed_intensity_coefficients=True,
            seed=42,
        )
        sampler = MarkedPointProcessSampler(dataset, config)
        results = sampler.run(show_progress=False)

        n_saved = config.n_saved()
        p_int = dataset.design_matrix_intensity.shape[1]

        # In fixed mode: (n_saved, p_int)
        assert results.beta_int_samples.shape == (n_saved, p_int)

    def test_run_spatial_mode_returns_correct_shape(self):
        """run() in spatial mode should return beta_int_samples with shape (n_saved, p_int, n_sites)."""
        dataset = create_simple_dataset()
        config = MarkedPointProcessConfig(
            n_iter=20,
            burn_in=10,
            thinning=2,
            fixed_intensity_coefficients=False,
            seed=42,
        )
        sampler = MarkedPointProcessSampler(dataset, config)
        results = sampler.run(show_progress=False)

        n_saved = config.n_saved()
        p_int = dataset.design_matrix_intensity.shape[1]
        n_sites = dataset.num_sites()

        # In spatial mode: (n_saved, p_int, n_sites)
        assert results.beta_int_samples.shape == (n_saved, p_int, n_sites)

    def test_lambda_star_samples_valid(self):
        """lambda_star samples should be positive."""
        dataset = create_simple_dataset()
        config = MarkedPointProcessConfig(
            n_iter=20,
            burn_in=10,
            fixed_intensity_coefficients=True,
            seed=42,
        )
        sampler = MarkedPointProcessSampler(dataset, config)
        results = sampler.run(show_progress=False)

        assert np.all(results.lambda_star_samples > 0)

    def test_predict_intensity_fixed_mode(self):
        """predict_intensity should work in fixed mode."""
        dataset = create_simple_dataset()
        config = MarkedPointProcessConfig(
            n_iter=20,
            burn_in=10,
            fixed_intensity_coefficients=True,
            seed=42,
        )
        sampler = MarkedPointProcessSampler(dataset, config)
        results = sampler.run(show_progress=False)

        # Predict at grid
        intensity_grid = results.predict_intensity(location="grid")
        n_grid = dataset.num_grid()

        assert intensity_grid.shape == (n_grid,)
        # Intensity should be non-negative (except NaN for invalid grids)
        valid_intensity = intensity_grid[~np.isnan(intensity_grid)]
        assert np.all(valid_intensity >= 0)

    def test_predict_intensity_sites_fixed_mode(self):
        """predict_intensity at sites should work in fixed mode."""
        dataset = create_simple_dataset()
        config = MarkedPointProcessConfig(
            n_iter=20,
            burn_in=10,
            fixed_intensity_coefficients=True,
            seed=42,
        )
        sampler = MarkedPointProcessSampler(dataset, config)
        results = sampler.run(show_progress=False)

        # Predict at sites
        intensity_sites = results.predict_intensity(location="sites")
        n_sites = dataset.num_sites()

        assert intensity_sites.shape == (n_sites,)
        assert np.all(intensity_sites >= 0)


class TestBackwardCompatibility:
    """Test that spatial mode (default) still works correctly."""

    def test_default_mode_is_spatial(self):
        """Default config should use spatial coefficients."""
        config = MarkedPointProcessConfig()
        assert config.fixed_intensity_coefficients is False

    def test_spatial_mode_sampler_runs(self):
        """Sampler should run in spatial mode without errors."""
        dataset = create_simple_dataset()
        config = MarkedPointProcessConfig(
            n_iter=10,
            burn_in=5,
            seed=42,
        )
        sampler = MarkedPointProcessSampler(dataset, config)
        results = sampler.run(show_progress=False)

        assert results.lambda_star_samples is not None
        assert results.beta_mark_samples is not None
        assert results.beta_int_samples is not None
