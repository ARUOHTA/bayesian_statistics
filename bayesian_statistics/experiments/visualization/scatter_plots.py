"""Scatter plot visualizations for model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .style import FIGURE_SIZE_GRID_2X2, TITLE_FONTSIZE


def plot_estimated_vs_observed(
    estimated: np.ndarray,
    observed: np.ndarray,
    origins: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot estimated vs observed scatter plots for all origins.

    Parameters
    ----------
    estimated : np.ndarray
        Estimated probabilities (K, n_sites).
    observed : np.ndarray
        Observed ratios (K, n_sites).
    origins : list of str
        Names of origins (first 4 used).
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : np.ndarray
        Array of matplotlib axes.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_GRID_2X2)
    axes_flat = axes.flatten()

    for k in range(4):  # First 4 origins
        ax = axes_flat[k]

        # Check for valid data
        obs_k = observed[k]
        est_k = estimated[k]
        valid_mask = ~(np.isnan(obs_k) | np.isnan(est_k))
        n_valid = valid_mask.sum()

        ax.plot([0, 1], [0, 1], "r--", label="y=x")
        ax.set_xlabel("実測比率")
        ax.set_ylabel("推定確率")
        ax.set_title(origins[k])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")

        if n_valid == 0:
            ax.text(
                0.5,
                0.5,
                "データなし",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            continue

        ax.scatter(obs_k[valid_mask], est_k[valid_mask], alpha=0.6, s=30)

        # Correlation and RMSE (need at least 2 points for correlation)
        if n_valid >= 2:
            corr = np.corrcoef(obs_k[valid_mask], est_k[valid_mask])[0, 1]
            rmse = np.sqrt(np.mean((est_k[valid_mask] - obs_k[valid_mask]) ** 2))
            ax.text(
                0.05,
                0.9,
                f"r = {corr:.3f}\nRMSE = {rmse:.3f}\nn = {n_valid}",
                fontsize=10,
                transform=ax.transAxes,
            )
        else:
            ax.text(
                0.05,
                0.9,
                f"n = {n_valid}\n(相関計算不可)",
                fontsize=10,
                transform=ax.transAxes,
            )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def compute_metrics(
    estimated: np.ndarray,
    observed: np.ndarray,
    origins: List[str],
) -> dict:
    """Compute evaluation metrics for all origins.

    Parameters
    ----------
    estimated : np.ndarray
        Estimated probabilities (K, n_sites).
    observed : np.ndarray
        Observed ratios (K, n_sites).
    origins : list of str
        Names of origins.

    Returns
    -------
    dict
        Dictionary of metrics for each origin.
    """
    metrics = {}

    for k, origin in enumerate(origins[:-1]):  # Exclude "その他"
        rmse = np.sqrt(np.mean((estimated[k] - observed[k]) ** 2))
        mae = np.mean(np.abs(estimated[k] - observed[k]))
        corr = np.corrcoef(estimated[k], observed[k])[0, 1]

        metrics[origin] = {
            "rmse": rmse,
            "mae": mae,
            "correlation": corr,
        }

    return metrics


def print_metrics_summary(metrics: dict) -> None:
    """Print metrics summary to console.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics for each origin.
    """
    print("=" * 50)
    print("推定精度の評価")
    print("=" * 50)
    print()

    for origin, m in metrics.items():
        print(f"{origin}:")
        print(f"  RMSE: {m['rmse']:.4f}")
        print(f"  MAE:  {m['mae']:.4f}")
        print(f"  相関: {m['correlation']:.4f}")
        print()


def plot_loocv_comparison(
    loocv_results: dict,
    time_periods: dict,
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot LOOCV comparison bar chart for NW and MMCP.

    Parameters
    ----------
    loocv_results : dict
        Dictionary with 'mmcp' and 'nw' keys, each containing
        period -> {'mean_aitchison_distance': float, 'std_aitchison_distance': float}.
    time_periods : dict
        Mapping from period index to period name.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    periods = list(time_periods.keys())
    period_names = [time_periods[p] for p in periods]

    mmcp_means = []
    mmcp_stds = []
    nw_means = []
    nw_stds = []

    for p in periods:
        mmcp_result = loocv_results.get("mmcp", {}).get(p, {})
        nw_result = loocv_results.get("nw", {}).get(p, {})

        mmcp_means.append(mmcp_result.get("mean_aitchison_distance", float("nan")))
        mmcp_stds.append(mmcp_result.get("std_aitchison_distance", float("nan")))
        nw_means.append(nw_result.get("mean_aitchison_distance", float("nan")))
        nw_stds.append(nw_result.get("std_aitchison_distance", float("nan")))

    x = np.arange(len(periods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Check if all data is NaN
    all_nan = all(np.isnan(m) for m in mmcp_means + nw_means)

    if all_nan:
        ax.text(
            0.5,
            0.5,
            "LOOCVデータなし\n(--run-loocv を実行してください)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=TITLE_FONTSIZE,
        )
        ax.set_xlabel("時期")
        ax.set_ylabel("Aitchison距離（LOOCV）")
        ax.set_xticks(x)
        ax.set_xticklabels(period_names)
        # No legend for empty plot
    else:
        bars1 = ax.bar(
            x - width / 2,
            nw_means,
            width,
            yerr=nw_stds,
            label="NW推定量",
            color="steelblue",
            capsize=3,
        )
        bars2 = ax.bar(
            x + width / 2,
            mmcp_means,
            width,
            yerr=mmcp_stds,
            label="MMCP（提案手法）",
            color="coral",
            capsize=3,
        )

        ax.set_xlabel("時期")
        ax.set_ylabel("Aitchison距離（LOOCV）")
        ax.set_xticks(x)
        ax.set_xticklabels(period_names)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, mean in zip(bars1, nw_means):
            if not np.isnan(mean):
                ax.annotate(
                    f"{mean:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        for bar, mean in zip(bars2, mmcp_means):
            if not np.isnan(mean):
                ax.annotate(
                    f"{mean:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax
