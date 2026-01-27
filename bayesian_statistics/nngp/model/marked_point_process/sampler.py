"""Gibbs sampler for the marked point process model.

This module implements the joint Gibbs sampler that alternates between:
- Point process (intensity) updates
- Mark (composition) updates

The sampler follows the algorithm described in sec7.tex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from bayesian_statistics.nngp.model.nngp import (
    FactorCache,
    NNGPFactors,
    build_grid_interpolation_matrix,
    build_nngp_factors,
    order_points_morton,
)
from bayesian_statistics.nngp.model.sample import (
    LocalNNGPKernel,
    PolyaGammaSampler,
    compute_eta,
    logratio_to_probs,
    softmax_with_baseline,
    update_beta_category,
)

from .composition import compute_kappa_tilde_mark, sample_xi_mark
from .config import MarkedPointProcessConfig
from .dataset import MarkedPointProcessDataset
from .intensity import (
    compute_eta_intensity,
    compute_eta_intensity_fixed,
    compute_kappa_intensity,
    compute_q,
    sample_omega_intensity,
    sample_pseudo_absence,
    update_beta_intensity,
    update_beta_intensity_fixed,
    update_lambda_star,
)

if TYPE_CHECKING:
    from bayesian_statistics.nngp.model.nngp import FactorCache


@dataclass
class MarkedPointProcessResults:
    """Results from the marked point process MCMC sampler.

    Attributes
    ----------
    lambda_star_samples : np.ndarray
        Samples of λ* intensity parameter, shape (n_saved,).
    beta_mark_samples : np.ndarray
        Samples of mark coefficients, shape (n_saved, K-1, p+1, n_sites).
    beta_int_samples : np.ndarray
        Samples of intensity coefficients.
        - Fixed mode: shape (n_saved, p_int)
        - Spatial mode: shape (n_saved, p_int, n_sites)
    n_pseudo_absence : List[int]
        Number of pseudo-absence points at each saved iteration.
    dataset : MarkedPointProcessDataset
        The dataset used for MCMC.
    config : MarkedPointProcessConfig
        The configuration used for MCMC.
    factor_cache : Optional[FactorCache]
        Cached NNGP factors for grid prediction (None if no grid).
    """

    lambda_star_samples: np.ndarray
    beta_mark_samples: np.ndarray
    beta_int_samples: np.ndarray
    n_pseudo_absence: List[int]
    dataset: MarkedPointProcessDataset
    config: "MarkedPointProcessConfig"
    factor_cache: Optional["FactorCache"] = None

    def _project_beta_to_grid(self, beta_sites: np.ndarray) -> np.ndarray:
        """Project site-level beta coefficients to grid using NNGP conditional.

        This method handles the prior mean adjustment for intercept when
        distance prior is used. The adjustment ensures that the spatial
        interpolation correctly accounts for the non-zero prior mean.

        Parameters
        ----------
        beta_sites : np.ndarray
            Site-level coefficients with shape (K-1, p+1, n_sites).

        Returns
        -------
        np.ndarray
            Grid-level coefficients with shape (K-1, p+1, n_grid).

        Raises
        ------
        ValueError
            If factor_cache is not available.
        """
        if self.factor_cache is None:
            raise ValueError("Grid prediction not available (no factor_cache)")

        K_minus_1, p_plus_1, _ = beta_sites.shape
        n_grid = self.dataset.num_grid()
        A_grid_list = self.factor_cache.A_grid_all()

        beta_grid = np.zeros((K_minus_1, p_plus_1, n_grid))

        for k in range(K_minus_1):
            for j in range(p_plus_1):
                A_grid = A_grid_list[j]
                beta_grid[k, j] = A_grid @ beta_sites[k, j]

                # Adjust for prior mean on intercept when distance prior is used
                if j == 0 and self.dataset.prior_mean_intercept_grid is not None:
                    prior_mean_grid = self.dataset.prior_mean_intercept_grid[k]
                    prior_mean_sites = self.dataset.prior_mean_intercept_sites[k]
                    # Correct for the difference between grid prior mean and
                    # the interpolated site prior mean
                    beta_grid[k, j] += prior_mean_grid - (A_grid @ prior_mean_sites)

        return beta_grid

    def predict_probabilities(
        self,
        location: str = "sites",
        sample_conditional: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Predict category probabilities at sites or grid.

        Parameters
        ----------
        location
            Either "sites" or "grid".
        sample_conditional
            If True and location="grid", sample from GP conditional.
            If False, use posterior mean projection.
        seed
            Random seed for conditional sampling.

        Returns
        -------
        np.ndarray
            Probability array with shape (K, n_locations), where K includes baseline.
        """
        K_minus_1 = self.beta_mark_samples.shape[1]
        n_samples = self.beta_mark_samples.shape[0]

        if location == "sites":
            W = self.dataset.design_matrix_marks
            beta_mean = self.beta_mark_samples.mean(axis=0)  # (K-1, p+1, n_sites)
        elif location == "grid":
            if self.factor_cache is None:
                raise ValueError("No grid factors available for grid prediction")
            if self.dataset.grid_coords is None:
                raise ValueError("No grid coordinates in dataset")

            W = self.dataset.design_matrix_grid_marks
            if W is None:
                # Use intercept-only if no design matrix provided
                n_grid = len(self.dataset.grid_coords)
                W = np.ones((n_grid, 1))

            if sample_conditional:
                # Sample beta_grid from GP conditional
                n_loc = W.shape[0]
                A_grid_list = self.factor_cache.A_grid_all()
                d_grid_list = self.factor_cache.d_grid_all()
                rng = np.random.default_rng(seed)
                beta_grid_samples = np.zeros((n_samples, K_minus_1, W.shape[1], n_loc))

                for sample_idx in range(n_samples):
                    beta_sites = self.beta_mark_samples[sample_idx]
                    for k in range(K_minus_1):
                        for j in range(W.shape[1]):
                            A_grid = A_grid_list[j]
                            d_grid = d_grid_list[j]
                            beta_sites_j = beta_sites[k, j]
                            mu_cond = A_grid @ beta_sites_j

                            # Apply prior mean adjustment for intercept
                            if j == 0 and self.dataset.prior_mean_intercept_grid is not None:
                                prior_mean_grid = self.dataset.prior_mean_intercept_grid[k]
                                prior_mean_sites = self.dataset.prior_mean_intercept_sites[k]
                                mu_cond = mu_cond + prior_mean_grid - (A_grid @ prior_mean_sites)

                            std_cond = np.sqrt(np.maximum(d_grid, 1e-12))
                            beta_grid_samples[sample_idx, k, j] = rng.normal(
                                mu_cond, std_cond
                            )
                beta_mean = beta_grid_samples.mean(axis=0)
            else:
                # Use posterior mean projection with prior mean adjustment
                beta_sites_mean = self.beta_mark_samples.mean(axis=0)
                beta_mean = self._project_beta_to_grid(beta_sites_mean)
        else:
            raise ValueError(f"location must be 'sites' or 'grid', got {location}")

        # Compute eta and probabilities
        eta = compute_eta(beta_mean, W)
        probs = softmax_with_baseline(eta)
        return probs

    def predict_intensity(
        self,
        location: str = "grid",
    ) -> np.ndarray:
        """Predict site existence intensity at grid points.

        The intensity function λ(s) = λ* × q(s) where q(s) = sigmoid(η_int(s)).

        For sites, η_int is directly available from the posterior samples.
        For grid points:
        - Fixed coefficient mode: η_int = W_grid @ beta (direct computation)
        - Spatial coefficient mode: η_int is interpolated using NNGP conditional mean

        Parameters
        ----------
        location
            Either "sites" or "grid".

        Returns
        -------
        np.ndarray
            Intensity values with shape (n_locations,).
        """
        lambda_star_mean = self.lambda_star_samples.mean()

        if location == "sites":
            W = self.dataset.design_matrix_intensity
            beta_int_mean = self.beta_int_samples.mean(axis=0)

            if self.config.fixed_intensity_coefficients:
                # Fixed mode: beta_int_mean has shape (p_int,)
                eta_sites = compute_eta_intensity_fixed(beta_int_mean, W)
            else:
                # Spatial mode: beta_int_mean has shape (p_int, n_sites)
                eta_sites = compute_eta_intensity(beta_int_mean, W)

            q = compute_q(eta_sites)
            return lambda_star_mean * q

        elif location == "grid":
            if self.dataset.grid_coords is None:
                raise ValueError("No grid coordinates in dataset")

            n_grid = len(self.dataset.grid_coords)

            # Get grid design matrix
            W_grid = self.dataset.design_matrix_grid_intensity
            if W_grid is None:
                W_grid = np.ones((n_grid, 1))

            # Handle NaN in design matrix (invalid grid points)
            W_grid_clean = np.nan_to_num(W_grid, nan=0.0)

            if self.config.fixed_intensity_coefficients:
                # Fixed mode: direct computation, no NNGP interpolation needed
                beta_int_mean = self.beta_int_samples.mean(axis=0)  # (p_int,)
                eta_grid = compute_eta_intensity_fixed(beta_int_mean, W_grid_clean)
            else:
                # Spatial mode: NNGP interpolation from sites to grid
                n_samples = self.beta_int_samples.shape[0]
                p_int_plus_1 = self.beta_int_samples.shape[1]

                kernel = LocalNNGPKernel(
                    lengthscale=self.config.intensity_kernel_lengthscale,
                    variance=self.config.intensity_kernel_variance,
                )
                A_grid = build_grid_interpolation_matrix(
                    site_coords=self.dataset.site_coords,
                    grid_coords=self.dataset.grid_coords,
                    kernel=kernel,
                    M=self.config.neighbor_count,
                )

                eta_grid_samples = np.zeros((n_samples, n_grid))
                for sample_idx in range(n_samples):
                    beta_sites = self.beta_int_samples[sample_idx]  # (p_int, n_sites)
                    beta_grid = np.zeros((p_int_plus_1, n_grid))

                    for j in range(p_int_plus_1):
                        beta_grid[j] = A_grid @ beta_sites[j]

                    eta_grid_samples[sample_idx] = compute_eta_intensity(
                        beta_grid, W_grid_clean
                    )

                eta_grid = eta_grid_samples.mean(axis=0)

            q = compute_q(eta_grid)
            intensity = lambda_star_mean * q

            # Mark invalid grid points (where original W_grid had NaN) as NaN
            invalid_mask = np.any(np.isnan(W_grid), axis=1)
            intensity[invalid_mask] = np.nan

            return intensity

        else:
            raise ValueError(f"location must be 'sites' or 'grid', got {location}")

    def decompose_effects(
        self,
        location: str = "sites",
    ) -> dict[str, np.ndarray]:
        """Decompose mark probabilities into distance vs data-driven effects.

        This method separates the contribution of distance prior from data-driven
        adjustments, allowing interpretation of how much the model relies on
        distance information versus observed composition patterns.

        Parameters
        ----------
        location
            Either "sites" or "grid".

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of effects, each with shape (K, n_locations):
            - "distance": pure distance effect (no data influence)
            - "intercept_adjustment": how data pulls away from distance baseline
            - "intercept": combined intercept effect (distance + adjustment)
            - "full": complete model including all covariates

        Raises
        ------
        ValueError
            If location is not "sites" or "grid".
            If distance prior is not available in the dataset.
        """
        if self.dataset.distance_features_sites is None:
            raise ValueError(
                "Distance prior not available in dataset. "
                "Please prepare dataset with distance_column_names parameter."
            )

        K_minus_1 = self.beta_mark_samples.shape[1]
        beta_mean = self.beta_mark_samples.mean(axis=0)  # (K-1, p+1, n_sites)

        if location == "sites":
            W = self.dataset.design_matrix_marks
            n_loc = self.dataset.num_sites()
            g = self.dataset.distance_features_sites.T  # (K-1, n_sites)
        elif location == "grid":
            W = self.dataset.design_matrix_grid_marks
            if W is None:
                # If no mark covariates specified, use intercept only
                n_loc = self.dataset.num_grid()
                W = np.ones((n_loc, 1))
            else:
                n_loc = W.shape[0]
            g = self.dataset.distance_features_grid.T  # (K-1, n_grid)

            # Project beta to grid using NNGP conditional with prior mean adjustment
            beta_mean = self._project_beta_to_grid(beta_mean)
        else:
            raise ValueError(f"location must be 'sites' or 'grid', got {location}")

        effects = {}

        # Helper function: softmax without numerical stabilization
        def softmax_no_stabilization(eta: np.ndarray) -> np.ndarray:
            """Softmax with baseline, without max-subtraction."""
            exp_eta = np.exp(eta)
            sum_exp = exp_eta.sum(axis=0, keepdims=True)
            baseline = 1.0 + sum_exp
            probs_non_baseline = exp_eta / baseline
            probs_baseline = 1.0 / baseline
            return np.vstack([probs_non_baseline, probs_baseline])

        # Get lambda values from config or use defaults
        if self.config.lambda_fixed is not None:
            lambda_vec = np.array(self.config.lambda_fixed)
        else:
            lambda_vec = np.ones(K_minus_1)

        # 1. Distance prior effect only (baseline expectation)
        logratio_dist = lambda_vec[:, np.newaxis] * g  # (K-1, n_loc)
        effects["distance"] = logratio_to_probs(logratio_dist)

        # 2. Data-driven adjustment to distance prior
        eta_intercept = beta_mean[:, 0, :]  # (K-1, n_loc)
        eta_adjustment = eta_intercept - logratio_dist  # delta_k(s)
        effects["intercept_adjustment"] = softmax_no_stabilization(eta_adjustment)

        # 3. Full intercept (distance + adjustment)
        effects["intercept"] = softmax_no_stabilization(eta_intercept)

        # 4. Individual covariates (if any beyond intercept)
        p_plus_1 = beta_mean.shape[1]
        if p_plus_1 > 1 and W is not None:
            for j in range(1, p_plus_1):
                # Note: mark variables might not have names, use index
                w_j = W[:, j]
                eta_j = beta_mean[:, j, :] * w_j
                effects[f"covariate_{j}"] = softmax_no_stabilization(eta_j)

            # 5. All covariates only (excluding intercept)
            eta_covariates_only = compute_eta(beta_mean[:, 1:, :], W[:, 1:])
            effects["covariates_only"] = softmax_no_stabilization(eta_covariates_only)

        # 6. Full model (distance + adjustment + covariates)
        eta_full = compute_eta(beta_mean, W)
        effects["full"] = softmax_no_stabilization(eta_full)

        return effects


class MarkedPointProcessSampler:
    """Gibbs sampler for the marked point process model.

    Parameters
    ----------
    dataset : MarkedPointProcessDataset
        Data for the model.
    config : MarkedPointProcessConfig
        Configuration for MCMC.
    """

    def __init__(
        self,
        dataset: MarkedPointProcessDataset,
        config: MarkedPointProcessConfig,
    ):
        self.dataset = dataset
        self.config = config

        # Initialize RNG
        if config.seed is not None:
            self.rng = np.random.default_rng(config.seed)
        else:
            self.rng = np.random.default_rng()

        # PG sampler
        self.pg = PolyaGammaSampler()

        # Build NNGP factors for marks (fixed since X is fixed)
        self._build_mark_factors()

    def _build_mark_factors(self):
        """Build NNGP factors for the mark component."""
        n_sites = self.dataset.num_sites()
        K = self.dataset.num_categories()
        K_minus_1 = K - 1

        # Morton ordering for better locality
        self.mark_order = order_points_morton(self.dataset.site_coords)

        # For update_beta_category, we need one factor per feature (p+1)
        p_plus_1 = self.dataset.design_matrix_marks.shape[1]

        # Build factors with per-feature kernels if provided
        if self.config.mark_kernel_lengthscale_by_feature is not None:
            lengthscales = list(self.config.mark_kernel_lengthscale_by_feature)
            if len(lengthscales) != p_plus_1:
                raise ValueError(
                    f"mark_kernel_lengthscale_by_feature must have length {p_plus_1}, "
                    f"got {len(lengthscales)}"
                )
        else:
            lengthscales = [self.config.mark_kernel_lengthscale] * p_plus_1

        if self.config.mark_kernel_variance_by_feature is not None:
            variances = list(self.config.mark_kernel_variance_by_feature)
            if len(variances) != p_plus_1:
                raise ValueError(
                    f"mark_kernel_variance_by_feature must have length {p_plus_1}, "
                    f"got {len(variances)}"
                )
        else:
            variances = [self.config.mark_kernel_variance] * p_plus_1

        # Build one factor per feature
        self.mark_factors_by_feature = []
        kernels_for_cache = []
        for j in range(p_plus_1):
            kernel = LocalNNGPKernel(lengthscale=lengthscales[j], variance=variances[j])
            factors, _ = build_nngp_factors(
                self.dataset.site_coords,
                self.config.neighbor_count,
                kernel,
                order=self.mark_order,
            )
            self.mark_factors_by_feature.append(factors)
            kernels_for_cache.append(kernel)

        # Build FactorCache for grid predictions if grid is available
        self.factor_cache: Optional[FactorCache] = None
        if self.dataset.grid_coords is not None and len(self.dataset.grid_coords) > 0:
            self.factor_cache = FactorCache(
                s_coords=self.dataset.site_coords,
                grid_points=self.dataset.grid_coords,
                M=self.config.neighbor_count,
                kernels=kernels_for_cache,
            )

    def _init_state(self):
        """Initialize MCMC state."""
        n_sites = self.dataset.num_sites()
        K = self.dataset.num_categories()
        K_minus_1 = K - 1
        p_mark = self.dataset.design_matrix_marks.shape[1]
        p_int = self.dataset.design_matrix_intensity.shape[1]

        # Initialize lambda*
        self.lambda_star = (
            float(n_sites) / self.dataset.volume if self.dataset.volume > 0 else 1.0
        )

        # Initialize mark coefficients: beta[k, j, i] for category k, feature j, site i
        # Start with zeros
        self.beta_marks = np.zeros((K_minus_1, p_mark, n_sites))

        # Initialize eta (linear predictor) for marks
        self.eta_marks = compute_eta(self.beta_marks, self.dataset.design_matrix_marks)

        # Initialize intensity coefficients
        # In fixed mode: beta_int[j] for feature j (shared across all sites)
        # In spatial mode: beta_int[j, i] for feature j, site i
        if self.config.fixed_intensity_coefficients:
            self.beta_int_sites = np.zeros(p_int)
            self.eta_int_sites = np.zeros(n_sites)
        else:
            self.beta_int_sites = np.zeros((p_int, n_sites))
            self.eta_int_sites = np.zeros(n_sites)

    def _sample_marks(self, iteration: int):
        """Gibbs step (e)-(f): Update mark parameters.

        Steps:
        (e) Sample ξ_ik ~ PG(N_i, η_ik) for each category k
        (f) Update (β_k, u_k) using conjugate Gaussian update
        """
        n_sites = self.dataset.num_sites()
        K = self.dataset.num_categories()
        K_minus_1 = K - 1
        total_counts = self.dataset.total_counts
        W = self.dataset.design_matrix_marks
        p_plus_1 = W.shape[1]

        # Prepare prior means for each category (distance prior for intercept)
        for k in range(K_minus_1):
            # (e) Sample PG variables
            xi_k = sample_xi_mark(self.eta_marks[k], total_counts, self.rng)

            # Compute κ̃_ik = y_ik - N_i/2
            counts_k = self.dataset.counts[:, k]
            kappa_tilde = compute_kappa_tilde_mark(counts_k, total_counts)

            # Prepare prior means for this category
            # prior_mean_by_feature[j] = None/array for each feature j
            prior_mean_by_feature = []
            for j in range(p_plus_1):
                if j == 0 and self.dataset.prior_mean_intercept_sites is not None:
                    # Intercept: use distance prior
                    prior_mean_by_feature.append(
                        self.dataset.prior_mean_intercept_sites[k]  # (n_sites,)
                    )
                else:
                    # Other features: no prior mean (defaults to 0)
                    prior_mean_by_feature.append(None)

            # (f) Update coefficients using Gibbs with distance prior
            self.beta_marks[k], self.eta_marks[k] = update_beta_category(
                beta_k=self.beta_marks[k],
                eta_k=self.eta_marks[k],
                W=W,
                factors_by_feature=self.mark_factors_by_feature,
                order=self.mark_order,
                omega=xi_k,
                kappa_tilde=kappa_tilde,
                rng=self.rng,
                prior_mean_by_feature=prior_mean_by_feature,
            )

    def _sample_intensity(self, iteration: int) -> int:
        """Gibbs steps (a)-(d): Update intensity parameters.

        Implements the full intensity Gibbs update from sec7.tex:
        (a) U ~ IPP(λ*(1-q)) via Poisson thinning
        (b) ω_i ~ PG(1, η_int,i) for i=1,...,n_X+n_U
        (c) (β_int, u_int) ~ N(m_int, V_int) at X ∪ U
        (d) λ* ~ Gamma(m_0 + n, r_0 + |D|)

        Returns the number of pseudo-absence points sampled.
        """
        if self.config.fixed_intensity_coefficients:
            return self._sample_intensity_fixed()
        else:
            return self._sample_intensity_spatial()

    def _sample_intensity_fixed(self) -> int:
        """Fixed coefficient mode: beta is shared across all locations."""
        n_sites = self.dataset.num_sites()
        p_int = self.dataset.design_matrix_intensity.shape[1]

        n_U = 0
        U_indices = np.array([], dtype=int)

        # Get grid design matrix
        W_grid = self.dataset.design_matrix_grid_intensity
        if W_grid is None and self.dataset.grid_coords is not None:
            W_grid = np.ones((len(self.dataset.grid_coords), 1))

        if self.dataset.grid_coords is not None and len(self.dataset.grid_coords) > 0:
            # (a) Sample pseudo-absence U ~ IPP(λ*(1-q))
            n_grid = len(self.dataset.grid_coords)
            valid_mask = (
                self.dataset.valid_grids
                if self.dataset.valid_grids is not None
                else np.ones(n_grid, dtype=bool)
            )

            # Fixed mode: eta_grid = W_grid @ beta (direct computation)
            W_grid_clean = np.nan_to_num(W_grid, nan=0.0)
            eta_grid = compute_eta_intensity_fixed(self.beta_int_sites, W_grid_clean)

            U_indices = sample_pseudo_absence(
                lambda_star=self.lambda_star,
                eta_grid=eta_grid,
                valid_mask=valid_mask,
                region_volume=self.dataset.volume,
                rng=self.rng,
            )
            n_U = len(U_indices)

        # (b)-(c) Update β_int using standard PG regression
        W_sites = self.dataset.design_matrix_intensity
        if n_U > 0:
            # Use cleaned grid design rows to avoid NaNs
            W_U = W_grid_clean[U_indices]
            W_combined = np.vstack([W_sites, W_U])
        else:
            W_combined = W_sites

        n_combined = n_sites + n_U

        # Compute eta for combined points
        eta_combined = compute_eta_intensity_fixed(self.beta_int_sites, W_combined)

        # y = 1 for sites, y = 0 for pseudo-absence
        y_combined = np.concatenate([np.ones(n_sites), np.zeros(n_U)])

        # (b) Sample ω ~ PG(1, η)
        omega_combined = sample_omega_intensity(eta_combined, self.rng)

        # Compute κ = y - 0.5
        kappa_combined = compute_kappa_intensity(y_combined)

        # (c) Update β using fixed coefficient update
        self.beta_int_sites, eta_combined = update_beta_intensity_fixed(
            beta_int=self.beta_int_sites,
            W=W_combined,
            omega=omega_combined,
            kappa=kappa_combined,
            prior_mean=self.config.intensity_prior_mean,
            prior_variance=self.config.intensity_prior_variance,
            rng=self.rng,
        )

        # Update eta at sites
        self.eta_int_sites = compute_eta_intensity_fixed(
            self.beta_int_sites, W_sites
        )

        # (d) Update λ*
        n_total = n_sites + n_U
        self.lambda_star = update_lambda_star(
            n_total=n_total,
            region_volume=self.dataset.volume,
            prior_shape=self.config.lambda_prior_shape,
            prior_rate=self.config.lambda_prior_rate,
            rng=self.rng,
        )

        return n_U

    def _sample_intensity_spatial(self) -> int:
        """Spatial coefficient mode: beta varies by location (NNGP)."""
        n_sites = self.dataset.num_sites()
        p_int = self.dataset.design_matrix_intensity.shape[1]
        kernel = LocalNNGPKernel(
            lengthscale=self.config.intensity_kernel_lengthscale,
            variance=self.config.intensity_kernel_variance,
        )

        n_U = 0
        U_coords = np.empty((0, 2))

        # Get grid design matrix
        W_grid = self.dataset.design_matrix_grid_intensity
        if W_grid is None and self.dataset.grid_coords is not None:
            W_grid = np.ones((len(self.dataset.grid_coords), 1))

        if self.dataset.grid_coords is not None and len(self.dataset.grid_coords) > 0:
            # (a) Sample pseudo-absence U ~ IPP(λ*(1-q))
            # Use current eta at grid points (interpolated from sites)
            n_grid = len(self.dataset.grid_coords)
            valid_mask = (
                self.dataset.valid_grids
                if self.dataset.valid_grids is not None
                else np.ones(n_grid, dtype=bool)
            )

            # Interpolate eta_int to grid using NNGP conditional mean matching intensity kernel
            A_grid = build_grid_interpolation_matrix(
                site_coords=self.dataset.site_coords,
                grid_coords=self.dataset.grid_coords,
                kernel=kernel,
                M=self.config.neighbor_count,
            )
            beta_grid = np.zeros((p_int, n_grid))
            for j in range(p_int):
                beta_grid[j] = A_grid @ self.beta_int_sites[j]
            eta_grid = compute_eta_intensity(beta_grid, W_grid)

            U_indices = sample_pseudo_absence(
                lambda_star=self.lambda_star,
                eta_grid=eta_grid,
                valid_mask=valid_mask,
                region_volume=self.dataset.volume,
                rng=self.rng,
            )
            n_U = len(U_indices)

            if n_U > 0:
                U_coords = self.dataset.grid_coords[U_indices]

        # (b)-(c) Update β_int at X ∪ U
        if n_U > 0:
            # Combine X and U coordinates
            combined_coords = np.vstack([self.dataset.site_coords, U_coords])
            n_combined = n_sites + n_U

            # Create design matrix for combined points
            W_sites = self.dataset.design_matrix_intensity
            # Use grid design matrix for U points (at their sampled indices)
            if W_grid is not None and W_grid.shape[1] == p_int:
                W_U = np.nan_to_num(W_grid, nan=0.0)[U_indices]
            else:
                W_U = np.ones((n_U, p_int))
            W_combined = np.vstack([W_sites, W_U])

            # Build NNGP factors for combined points (X ∪ U)
            order_combined = order_points_morton(combined_coords)
            factors_combined, _ = build_nngp_factors(
                combined_coords,
                M=self.config.neighbor_count,
                kernel=kernel,
                order=order_combined,
            )
            factors_by_feature = [factors_combined] * p_int

            # Initialize beta_int for combined points
            # Use current values at sites, zeros at U
            beta_int_combined = np.zeros((p_int, n_combined))
            beta_int_combined[:, :n_sites] = self.beta_int_sites

            # Compute eta for combined points
            eta_int_combined = compute_eta_intensity(beta_int_combined, W_combined)

            # y = 1 for sites (X), y = 0 for pseudo-absence (U)
            y_combined = np.concatenate([np.ones(n_sites), np.zeros(n_U)])

            # (b) Sample ω ~ PG(1, η_int) for all combined points
            omega_combined = sample_omega_intensity(eta_int_combined, self.rng)

            # Compute κ = y - 0.5
            kappa_combined = compute_kappa_intensity(y_combined)

            # (c) Update β_int at combined points
            beta_int_combined, eta_int_combined = update_beta_intensity(
                beta_int=beta_int_combined,
                eta_int=eta_int_combined,
                W=W_combined,
                factors_by_feature=factors_by_feature,
                order=order_combined,
                omega=omega_combined,
                kappa=kappa_combined,
                rng=self.rng,
            )

            # Store updated values at sites (first n_sites entries)
            self.beta_int_sites = beta_int_combined[:, :n_sites]
            self.eta_int_sites = eta_int_combined[:n_sites]
        else:
            # No pseudo-absence: update only at sites
            order_sites = order_points_morton(self.dataset.site_coords)
            factors_sites, _ = build_nngp_factors(
                self.dataset.site_coords,
                M=self.config.neighbor_count,
                kernel=kernel,
                order=order_sites,
            )
            factors_by_feature = [factors_sites] * p_int

            W_sites = self.dataset.design_matrix_intensity
            y_sites = np.ones(n_sites)

            omega_sites = sample_omega_intensity(self.eta_int_sites, self.rng)
            kappa_sites = compute_kappa_intensity(y_sites)

            self.beta_int_sites, self.eta_int_sites = update_beta_intensity(
                beta_int=self.beta_int_sites,
                eta_int=self.eta_int_sites,
                W=W_sites,
                factors_by_feature=factors_by_feature,
                order=order_sites,
                omega=omega_sites,
                kappa=kappa_sites,
                rng=self.rng,
            )

        # (d) Update λ* ~ Gamma(m_0 + n, r_0 + |D|)
        n_total = n_sites + n_U
        self.lambda_star = update_lambda_star(
            n_total=n_total,
            region_volume=self.dataset.volume,
            prior_shape=self.config.lambda_prior_shape,
            prior_rate=self.config.lambda_prior_rate,
            rng=self.rng,
        )

        return n_U

    def run(self, show_progress: bool = True) -> MarkedPointProcessResults:
        """Run the MCMC sampler.

        Parameters
        ----------
        show_progress : bool
            Whether to show a progress bar.

        Returns
        -------
        MarkedPointProcessResults
            Results containing posterior samples.
        """
        # Initialize state
        self._init_state()

        # Storage for samples
        n_saved = self.config.n_saved()
        K_minus_1 = self.dataset.num_categories() - 1
        p_mark = self.dataset.design_matrix_marks.shape[1]
        p_int = self.dataset.design_matrix_intensity.shape[1]
        n_sites = self.dataset.num_sites()

        lambda_star_samples = np.zeros(n_saved)
        beta_mark_samples = np.zeros((n_saved, K_minus_1, p_mark, n_sites))
        # Fixed mode: (n_saved, p_int), Spatial mode: (n_saved, p_int, n_sites)
        if self.config.fixed_intensity_coefficients:
            beta_int_samples = np.zeros((n_saved, p_int))
        else:
            beta_int_samples = np.zeros((n_saved, p_int, n_sites))
        n_pseudo_absence = []

        save_idx = 0
        iterator = range(self.config.n_iter)
        if show_progress:
            iterator = tqdm(iterator, desc="MCMC")

        for iteration in iterator:
            # === Intensity (point process) updates ===
            n_U = self._sample_intensity(iteration)

            # === Mark (composition) updates ===
            self._sample_marks(iteration)

            # Save samples after burn-in and thinning
            if iteration >= self.config.burn_in:
                if (iteration - self.config.burn_in) % self.config.thinning == 0:
                    if save_idx < n_saved:
                        lambda_star_samples[save_idx] = self.lambda_star
                        beta_mark_samples[save_idx] = self.beta_marks.copy()
                        beta_int_samples[save_idx] = self.beta_int_sites.copy()
                        n_pseudo_absence.append(n_U)
                        save_idx += 1

        return MarkedPointProcessResults(
            lambda_star_samples=lambda_star_samples,
            beta_mark_samples=beta_mark_samples,
            beta_int_samples=beta_int_samples,
            n_pseudo_absence=n_pseudo_absence,
            dataset=self.dataset,
            config=self.config,
            factor_cache=self.factor_cache,
        )
