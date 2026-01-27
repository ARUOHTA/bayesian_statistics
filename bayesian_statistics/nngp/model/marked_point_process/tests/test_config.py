"""Tests for MarkedPointProcessConfig fixed intensity coefficients."""

import pytest
from bayesian_statistics.nngp.model.marked_point_process.config import (
    MarkedPointProcessConfig,
)


class TestFixedIntensityCoefficientConfig:
    """Test fixed_intensity_coefficients configuration."""

    def test_default_is_false(self):
        """fixed_intensity_coefficients should default to False."""
        config = MarkedPointProcessConfig()
        assert config.fixed_intensity_coefficients is False

    def test_can_set_true(self):
        """fixed_intensity_coefficients can be set to True."""
        config = MarkedPointProcessConfig(fixed_intensity_coefficients=True)
        assert config.fixed_intensity_coefficients is True

    def test_intensity_prior_mean_default(self):
        """intensity_prior_mean should default to 0.0."""
        config = MarkedPointProcessConfig()
        assert config.intensity_prior_mean == 0.0

    def test_intensity_prior_variance_default(self):
        """intensity_prior_variance should default to 10.0."""
        config = MarkedPointProcessConfig()
        assert config.intensity_prior_variance == 10.0

    def test_can_customize_prior_hyperparameters(self):
        """Prior hyperparameters can be customized."""
        config = MarkedPointProcessConfig(
            fixed_intensity_coefficients=True,
            intensity_prior_mean=-1.0,
            intensity_prior_variance=5.0,
        )
        assert config.fixed_intensity_coefficients is True
        assert config.intensity_prior_mean == -1.0
        assert config.intensity_prior_variance == 5.0
