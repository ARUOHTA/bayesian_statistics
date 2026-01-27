"""Configuration for the marked point process model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class MarkedPointProcessConfig:
    """Configuration for the marked point process MCMC sampler.

    This configuration controls both the point process (intensity) component
    and the mark (composition) component of the joint model.

    Attributes
    ----------
    n_iter : int
        Total number of MCMC iterations.
    burn_in : int
        Number of initial iterations to discard.
    thinning : int
        Keep every thinning-th sample after burn-in.
    seed : Optional[int]
        Random seed for reproducibility.
    neighbor_count : int
        Number of neighbors for NNGP approximation.
    intensity_kernel_lengthscale : float
        RBF kernel lengthscale for the intensity (point process) component.
    intensity_kernel_variance : float
        RBF kernel variance for the intensity component.
    mark_kernel_lengthscale : float
        RBF kernel lengthscale for the mark (composition) component.
    mark_kernel_variance : float
        RBF kernel variance for the mark component.
    mark_kernel_lengthscale_by_feature : Optional[Sequence[float]]
        Per-feature lengthscales for mark component. If provided, overrides
        mark_kernel_lengthscale.
    mark_kernel_variance_by_feature : Optional[Sequence[float]]
        Per-feature variances for mark component. If provided, overrides
        mark_kernel_variance.
    lambda_prior_shape : float
        Shape parameter (m_0) for the Gamma prior on lambda*.
    lambda_prior_rate : float
        Rate parameter (r_0) for the Gamma prior on lambda*.
    tau : float
        Temperature parameter for distance-based weighted inverse softmax.
    alpha : float
        Importance weight exponent for distance prior.
    source_weights : Optional[Sequence[float]]
        Source importance weights for distance prior (length K-1).
    lambda_fixed : Optional[Sequence[float]]
        Fixed scaling factors for distance prior mean (length K-1).
    """

    # MCMC settings
    n_iter: int = 2000
    burn_in: int = 500
    thinning: int = 1
    seed: Optional[int] = None

    # NNGP settings
    neighbor_count: int = 25

    # Intensity (point process) kernel
    intensity_kernel_lengthscale: float = 0.1
    intensity_kernel_variance: float = 1.0

    # Mark (composition) kernel
    mark_kernel_lengthscale: float = 0.1
    mark_kernel_variance: float = 1.0
    mark_kernel_lengthscale_by_feature: Optional[Sequence[float]] = None
    mark_kernel_variance_by_feature: Optional[Sequence[float]] = None

    # Lambda* prior: Gamma(shape, rate)
    lambda_prior_shape: float = 2.0
    lambda_prior_rate: float = 1.0

    # Distance prior hyperparameters (for mark intercepts)
    tau: float = 1.0  # Temperature for weighted inverse softmax
    alpha: float = 1.0  # Weight on source importance
    source_weights: Optional[Sequence[float]] = None  # Importance weights (K-1,)
    lambda_fixed: Optional[Sequence[float]] = None  # Fixed scaling (K-1,) or None

    # Fixed intensity coefficients mode (non-spatial)
    fixed_intensity_coefficients: bool = False  # True for fixed (non-spatial) coefficients
    intensity_prior_mean: float = 0.0  # Prior mean for fixed coefficients (scalar or per-coef vector supported at runtime)
    intensity_prior_variance: float = 10.0  # Prior variance for fixed coefficients (scalar or per-coef vector supported at runtime)

    def n_saved(self) -> int:
        """Compute the number of posterior samples that will be saved.

        Returns
        -------
        int
            Number of saved samples after burn-in and thinning.

        Raises
        ------
        ValueError
            If thinning is less than 1.
        """
        if self.thinning < 1:
            raise ValueError("thinning must be at least 1")
        usable = max(self.n_iter - self.burn_in, 0)
        return usable // self.thinning
