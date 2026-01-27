"""MMCP (Marked Multinomial Composition Process) model runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from ..config import ExperimentConfig
from ..output import ProgressManager, get_progress_manager

if TYPE_CHECKING:
    from bayesian_statistics.models.preprocessing.data_preprocessor import (
        ObsidianDataPreprocessor,
    )
    from bayesian_statistics.nngp.model.marked_point_process import (
        MarkedPointProcessDataset,
        MarkedPointProcessResults,
    )


@dataclass
class MMCPRunnerResult:
    """Result container for MMCP runner.

    Attributes
    ----------
    results : MarkedPointProcessResults
        Raw MCMC results from the sampler.
    dataset : MarkedPointProcessDataset
        Dataset used for sampling.
    site_probs : np.ndarray
        Predicted probabilities at site locations (K, n_sites).
    grid_probs : np.ndarray
        Predicted probabilities at grid locations (K, n_grid).
    effects : dict
        Decomposed effects (distance, intercept_adjustment, full, etc.).
    period : int
        Time period index.
    """

    results: Any
    dataset: Any
    site_probs: np.ndarray
    grid_probs: np.ndarray
    effects: Dict[str, np.ndarray]
    period: int


class MMCPRunner:
    """Runner for MMCP model.

    This class wraps the MarkedPointProcessSampler to provide a consistent
    interface for running experiments.

    Attributes
    ----------
    config : ExperimentConfig
        Experiment configuration.
    progress : ProgressManager
        Progress manager for output control.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        progress: ProgressManager | None = None,
    ):
        """Initialize MMCPRunner.

        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration.
        progress : ProgressManager, optional
            Progress manager for output control. If None, uses global default.
        """
        self.config = config
        self.progress = progress or get_progress_manager()

    def _create_mmcp_config(self):
        """Create MarkedPointProcessConfig from ExperimentConfig.

        Returns
        -------
        MarkedPointProcessConfig
            Configuration for the MCMC sampler.
        """
        from bayesian_statistics.nngp.model.marked_point_process import (
            MarkedPointProcessConfig,
        )

        return MarkedPointProcessConfig(
            n_iter=self.config.n_iter,
            burn_in=self.config.burn_in,
            thinning=self.config.thinning,
            seed=42,
            neighbor_count=self.config.neighbor_count,
            intensity_kernel_lengthscale=self.config.intensity_lengthscale,
            intensity_kernel_variance=self.config.intensity_variance,
            mark_kernel_lengthscale=self.config.mark_lengthscale,
            mark_kernel_variance=self.config.mark_variance,
            lambda_prior_shape=self.config.lambda_prior_shape,
            lambda_prior_rate=self.config.lambda_prior_rate,
            tau=self.config.tau,
            alpha=self.config.alpha,
            source_weights=self.config.source_weights,
            lambda_fixed=self.config.lambda_fixed,
            fixed_intensity_coefficients=self.config.fixed_intensity_coefficients,
            intensity_prior_mean=self.config.intensity_prior_mean,
            intensity_prior_variance=self.config.intensity_prior_variance,
        )

    def _prepare_dataset(
        self,
        preprocessor: "ObsidianDataPreprocessor",
        period: int,
    ) -> "MarkedPointProcessDataset":
        """Prepare dataset for a single period.

        Parameters
        ----------
        preprocessor : ObsidianDataPreprocessor
            Data preprocessor.
        period : int
            Time period index (0-4).

        Returns
        -------
        MarkedPointProcessDataset
            Prepared dataset.
        """
        from bayesian_statistics.nngp.model.marked_point_process import (
            prepare_marked_point_process_dataset,
        )

        dataset = prepare_marked_point_process_dataset(
            preprocessor=preprocessor,
            period=period,
            origins=self.config.origins,
            grid_subsample_ratio=self.config.grid_subsample_ratio,
            drop_zero_total_sites=True,
            intensity_variable_names=self.config.intensity_variable_names,
            mark_variable_names=self.config.mark_variable_names,
            distance_column_names=self.config.distance_column_names,
            source_weights=self.config.source_weights,
            lambda_fixed=self.config.lambda_fixed,
            tau=self.config.tau,
            alpha=self.config.alpha,
        )

        return dataset

    def _run_mcmc(
        self,
        preprocessor: "ObsidianDataPreprocessor",
        period: int,
        show_progress: bool = True,
    ) -> tuple:
        """Run MCMC sampling.

        Parameters
        ----------
        preprocessor : ObsidianDataPreprocessor
            Data preprocessor.
        period : int
            Time period index.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        tuple
            (results, dataset)
        """
        from bayesian_statistics.nngp.model.marked_point_process import (
            MarkedPointProcessSampler,
        )

        dataset = self._prepare_dataset(preprocessor, period)
        config = self._create_mmcp_config()
        sampler = MarkedPointProcessSampler(dataset, config)
        results = sampler.run(show_progress=show_progress)

        return results, dataset

    def run_single_period(
        self,
        preprocessor: "ObsidianDataPreprocessor",
        period: int,
    ) -> Dict[str, Any]:
        """Run MMCP model for a single period.

        Parameters
        ----------
        preprocessor : ObsidianDataPreprocessor
            Data preprocessor.
        period : int
            Time period index (0-4).

        Returns
        -------
        dict
            Dictionary containing:
            - results: Raw MCMC results
            - dataset: Dataset used for sampling
            - site_probs: Predicted probabilities at sites (K, n_sites)
            - grid_probs: Predicted probabilities at grid (K, n_grid)
            - effects: Decomposed effects dict
        """
        show_progress = self.progress.should_show_progress()
        results, dataset = self._run_mcmc(preprocessor, period, show_progress)

        # Predict probabilities
        site_probs = results.predict_probabilities(location="sites")
        grid_probs = results.predict_probabilities(
            location="grid", sample_conditional=False
        )

        # Decompose effects
        effects = results.decompose_effects(location="grid")

        return {
            "results": results,
            "dataset": dataset,
            "site_probs": site_probs,
            "grid_probs": grid_probs,
            "effects": effects,
            "period": period,
        }

    def run_all_periods(
        self,
        preprocessor: "ObsidianDataPreprocessor",
        periods: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Run MMCP model for all specified periods.

        Parameters
        ----------
        preprocessor : ObsidianDataPreprocessor
            Data preprocessor.
        periods : list of int, optional
            List of period indices to run. If None, runs all 5 periods.

        Returns
        -------
        dict
            Dictionary mapping period index to result dict.
        """
        if periods is None:
            periods = list(self.config.time_periods.keys())

        all_results = {}
        for period in periods:
            period_name = self.config.time_periods[period]
            self.progress.subsection(f"Period {period}: {period_name}")

            with self.progress.nested():
                result = self.run_single_period(preprocessor, period)
            all_results[period] = result

            self.progress.info(
                f"Sites: {result['dataset'].num_sites()}, "
                f"Saved samples: {len(result['results'].lambda_star_samples)}"
            )

        return all_results

    def compute_true_ratios(self, dataset: "MarkedPointProcessDataset") -> np.ndarray:
        """Compute true (observed) ratios from counts.

        Parameters
        ----------
        dataset : MarkedPointProcessDataset
            Dataset containing counts.

        Returns
        -------
        np.ndarray
            True ratios (K, n_sites).
        """
        counts = dataset.counts
        total = counts.sum(axis=1, keepdims=True)
        ratios = np.nan_to_num(counts / total)
        return ratios.T  # (K, n_sites)
