"""Spatial distribution plots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .map_template import MapPlotter
from .style import (
    CMAP_INTENSITY,
    CMAP_PROBABILITY,
    FIGURE_SIZE_GRID_1X2,
    FIGURE_SIZE_GRID_2X2,
    FIGURE_SIZE_SINGLE,
    TITLE_FONTSIZE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_single_origin_map(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    grid_probs: np.ndarray,
    site_coords: np.ndarray,
    true_ratio: np.ndarray,
    origin_name: str,
    period_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot composition map for a single origin.

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    grid_probs : np.ndarray
        Grid probabilities (n_grid,).
    site_coords : np.ndarray
        Site coordinates (n_sites, 2).
    true_ratio : np.ndarray
        True observed ratios at sites (n_sites,).
    origin_name : str
        Name of the obsidian origin.
    period_name : str
        Name of the time period.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    fig, ax = plotter.create_figure(1, 1, figsize=FIGURE_SIZE_SINGLE)

    # Plot grid probabilities
    sc = plotter.plot_grid_scatter(
        ax=ax,
        coords=grid_coords,
        values=grid_probs,
    )

    # Plot site true ratios
    plotter.plot_site_scatter(
        ax=ax,
        coords=site_coords,
        values=true_ratio,
    )

    # Plot boundary
    plotter.plot_boundary(ax)

    # Add colorbar and labels
    plotter.add_colorbar(fig, sc, ax, label="確率")
    plotter.set_axis_style(
        ax,
        title=f"{origin_name}産黒曜石の組成比（{period_name}）",
        xlabel="経度",
        ylabel="緯度",
    )

    if output_path:
        plotter.save_figure(fig, output_path)

    return fig, ax


def plot_all_origins_grid(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    grid_probs: np.ndarray,
    site_coords: np.ndarray,
    true_ratios: np.ndarray,
    origins: List[str],
    period_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot composition maps for all origins in a 2x2 grid.

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    grid_probs : np.ndarray
        Grid probabilities (K, n_grid).
    site_coords : np.ndarray
        Site coordinates (n_sites, 2).
    true_ratios : np.ndarray
        True observed ratios at sites (K, n_sites).
    origins : list of str
        Names of obsidian origins (first 4 only used).
    period_name : str
        Name of the time period.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : np.ndarray
        Array of matplotlib axes.
    """
    fig, axes = plotter.create_figure(2, 2, figsize=FIGURE_SIZE_GRID_2X2)
    axes_flat = axes.flatten()

    for idx in range(4):  # 4 main origins
        ax = axes_flat[idx]

        # Plot grid probabilities
        sc = plotter.plot_grid_scatter(
            ax=ax,
            coords=grid_coords,
            values=grid_probs[idx],
            s=8,
        )

        # Plot site true ratios
        plotter.plot_site_scatter(
            ax=ax,
            coords=site_coords,
            values=true_ratios[idx],
            s=30,
            linewidths=0.3,
        )

        # Plot boundary
        plotter.plot_boundary(ax)

        # Labels
        plotter.add_colorbar(fig, sc, ax, label="確率")
        plotter.set_axis_style(ax, title=origins[idx])

    fig.suptitle(
        f"黒曜石産地組成比のグリッド予測（{period_name}）",
        fontsize=TITLE_FONTSIZE,
    )

    if output_path:
        plotter.save_figure(fig, output_path)

    return fig, axes


def plot_intensity_map(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    intensity: np.ndarray,
    site_coords: np.ndarray,
    period_name: str,
    lambda_star_mean: float,
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot intensity map.

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    intensity : np.ndarray
        Intensity values at grid (n_grid,).
    site_coords : np.ndarray
        Site coordinates (n_sites, 2).
    period_name : str
        Name of the time period.
    lambda_star_mean : float
        Mean lambda star value.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    fig, ax = plotter.create_figure(1, 1, figsize=FIGURE_SIZE_SINGLE)

    # Plot intensity
    sc = ax.scatter(
        grid_coords[plotter.is_land, 0],
        grid_coords[plotter.is_land, 1],
        c=intensity[plotter.is_land],
        cmap=CMAP_INTENSITY,
        s=10,
        alpha=0.8,
    )

    # Plot sites
    ax.scatter(
        site_coords[:, 0],
        site_coords[:, 1],
        c="blue",
        s=30,
        marker="^",
        edgecolors="white",
        linewidths=0.5,
        alpha=0.8,
        label=f"遺跡 (n={len(site_coords)})",
    )

    # Plot boundary
    plotter.plot_boundary(ax)

    # Add colorbar and labels
    plotter.add_colorbar(fig, sc, ax, label="強度 λ(s)")
    plotter.set_axis_style(
        ax,
        title=f"遺跡存在強度（{period_name}）\nλ* = {lambda_star_mean:.2f}",
        xlabel="経度",
        ylabel="緯度",
    )
    ax.legend(loc="upper right")

    if output_path:
        plotter.save_figure(fig, output_path)

    return fig, ax


def plot_comparison(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    grid_probs: np.ndarray,
    site_coords: np.ndarray,
    site_probs: np.ndarray,
    true_ratio: np.ndarray,
    origin_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot estimated vs observed comparison.

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    grid_probs : np.ndarray
        Grid probabilities (n_grid,).
    site_coords : np.ndarray
        Site coordinates (n_sites, 2).
    site_probs : np.ndarray
        Estimated probabilities at sites (n_sites,).
    true_ratio : np.ndarray
        True observed ratios at sites (n_sites,).
    origin_name : str
        Name of the obsidian origin.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : np.ndarray
        Array of matplotlib axes.
    """
    fig, axes = plotter.create_figure(1, 2, figsize=(16, 6))

    # Estimated
    ax = axes[0]
    sc = plotter.plot_grid_scatter(ax=ax, coords=grid_coords, values=grid_probs)
    plotter.plot_site_scatter(ax=ax, coords=site_coords, values=site_probs)
    plotter.plot_boundary(ax)
    plotter.add_colorbar(fig, sc, ax, label="確率")
    plotter.set_axis_style(ax, title=f"{origin_name}: 推定値（事後平均）")

    # Observed
    ax = axes[1]
    sc = plotter.plot_site_scatter(ax=ax, coords=site_coords, values=true_ratio)
    plotter.plot_boundary(ax)
    plotter.add_colorbar(fig, sc, ax, label="比率")
    plotter.set_axis_style(ax, title=f"{origin_name}: 実測値")

    if output_path:
        plotter.save_figure(fig, output_path)

    return fig, axes


def plot_distance_prior(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    p0_grid: np.ndarray,
    origins: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot distance-based prior probability p0 for main origins (神津島, 信州).

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    p0_grid : np.ndarray
        Distance-based prior probabilities (n_grid, K).
    origins : list of str
        Names of origins (first 2 used: 神津島, 信州).
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : np.ndarray
        Array of matplotlib axes.
    """
    import matplotlib.pyplot as plt

    from .style import CMAP_PRIOR, SUPTITLE_FONTSIZE, set_spine_width

    # Only show first 2 origins (神津島, 信州)
    n_origins_to_plot = 2

    fig, axes = plt.subplots(
        1, n_origins_to_plot, figsize=FIGURE_SIZE_GRID_1X2, constrained_layout=True
    )

    for idx in range(n_origins_to_plot):
        ax = axes[idx]

        sc = ax.scatter(
            grid_coords[plotter.is_land, 0],
            grid_coords[plotter.is_land, 1],
            c=p0_grid[plotter.is_land, idx],
            cmap=CMAP_PRIOR,
            s=8,
            alpha=0.8,
            vmin=0,
            vmax=1,
        )
        plotter.plot_boundary(ax)

        plt.colorbar(sc, ax=ax, label="事前確率", shrink=0.8)
        ax.set_title(origins[idx], fontsize=TITLE_FONTSIZE)
        ax.set_xticks([])
        ax.set_yticks([])
        set_spine_width(ax, 0.3)

    fig.suptitle(
        r"距離ベース事前分布 $p_0(s)$",
        fontsize=SUPTITLE_FONTSIZE,
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_model_comparison_map(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    mmcp_probs: np.ndarray,
    nw_probs: np.ndarray,
    site_coords: np.ndarray,
    true_ratio: np.ndarray,
    origins: List[str],
    period_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot NW vs MMCP comparison for all origins.

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    mmcp_probs : np.ndarray
        MMCP probabilities (K, n_grid).
    nw_probs : np.ndarray
        NW probabilities (K, n_grid).
    site_coords : np.ndarray
        Site coordinates (n_sites, 2).
    true_ratio : np.ndarray
        True observed ratios (K, n_sites).
    origins : list of str
        Names of origins (first 4 used).
    period_name : str
        Name of the time period.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : np.ndarray
        Array of matplotlib axes.
    """
    import matplotlib.pyplot as plt

    from .style import SUPTITLE_FONTSIZE, set_spine_width

    # 4 rows (origins) x 2 cols (NW, MMCP)
    fig, axes = plt.subplots(4, 2, figsize=(12, 20), constrained_layout=True)

    for origin_idx in range(4):
        # NW (left column)
        ax = axes[origin_idx, 0]
        sc = ax.scatter(
            grid_coords[plotter.is_land, 0],
            grid_coords[plotter.is_land, 1],
            c=nw_probs[origin_idx, plotter.is_land],
            cmap=CMAP_PROBABILITY,
            s=6,
            alpha=0.8,
            vmin=0,
            vmax=1,
        )
        ax.scatter(
            site_coords[:, 0],
            site_coords[:, 1],
            c=true_ratio[origin_idx],
            cmap=CMAP_PROBABILITY,
            s=20,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.8,
            vmin=0,
            vmax=1,
        )
        plotter.plot_boundary(ax)

        if origin_idx == 0:
            ax.set_title("NW推定量", fontsize=12)
        ax.set_ylabel(origins[origin_idx], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        set_spine_width(ax, 0.3)

        # MMCP (right column)
        ax = axes[origin_idx, 1]
        sc = ax.scatter(
            grid_coords[plotter.is_land, 0],
            grid_coords[plotter.is_land, 1],
            c=mmcp_probs[origin_idx, plotter.is_land],
            cmap=CMAP_PROBABILITY,
            s=6,
            alpha=0.8,
            vmin=0,
            vmax=1,
        )
        ax.scatter(
            site_coords[:, 0],
            site_coords[:, 1],
            c=true_ratio[origin_idx],
            cmap=CMAP_PROBABILITY,
            s=20,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.8,
            vmin=0,
            vmax=1,
        )
        plotter.plot_boundary(ax)

        if origin_idx == 0:
            ax.set_title("MMCP（提案手法）", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        set_spine_width(ax, 0.3)

    fig.colorbar(sc, ax=axes, label="確率", shrink=0.6, pad=0.02)
    fig.suptitle(
        f"モデル比較: NW vs MMCP（{period_name}）",
        fontsize=SUPTITLE_FONTSIZE,
        y=1.01,
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_site_probability_map(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    site_probability: np.ndarray,
    site_coords: np.ndarray,
    period_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot site existence probability map q(s) = sigmoid(η_int).

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    site_probability : np.ndarray
        Site existence probability q(s) at grid (n_grid,).
    site_coords : np.ndarray
        Site coordinates (n_sites, 2).
    period_name : str
        Name of the time period.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    fig, ax = plotter.create_figure(1, 1, figsize=FIGURE_SIZE_SINGLE)

    # Plot site probability (magma colormap, 0-1 range)
    sc = ax.scatter(
        grid_coords[plotter.is_land, 0],
        grid_coords[plotter.is_land, 1],
        c=site_probability[plotter.is_land],
        cmap="magma",
        s=10,
        alpha=0.8,
        vmin=0,
        vmax=1,
    )

    # Plot sites as white points with black edges
    ax.scatter(
        site_coords[:, 0],
        site_coords[:, 1],
        c="white",
        s=30,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
        label=f"遺跡 (n={len(site_coords)})",
    )

    plotter.plot_boundary(ax)
    plotter.add_colorbar(fig, sc, ax, label="存在確率 q(s)")
    plotter.set_axis_style(
        ax,
        title=f"遺跡の存在確率（{period_name}）",
        xlabel="経度",
        ylabel="緯度",
    )
    ax.legend(loc="upper right")

    if output_path:
        plotter.save_figure(fig, output_path)

    return fig, ax


def plot_uncertainty_map(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    posterior_std: np.ndarray,
    site_coords: np.ndarray,
    origins: List[str],
    period_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot posterior standard deviation (uncertainty) maps for main origins.

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    posterior_std : np.ndarray
        Posterior standard deviation (K, n_grid).
    site_coords : np.ndarray
        Site coordinates (n_sites, 2).
    origins : list of str
        Names of origins (first 2 used: 神津島, 信州).
    period_name : str
        Name of the time period.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : np.ndarray
        Array of matplotlib axes.
    """
    import matplotlib.pyplot as plt

    from .style import CMAP_INTENSITY, SUPTITLE_FONTSIZE, set_spine_width

    # Only show first 2 origins (神津島, 信州)
    n_origins_to_plot = 2

    fig, axes = plt.subplots(
        1, n_origins_to_plot, figsize=FIGURE_SIZE_GRID_1X2, constrained_layout=True
    )

    # Check for valid data (not all NaN)
    valid_data = posterior_std[:n_origins_to_plot, plotter.is_land]
    all_nan = np.all(np.isnan(valid_data))

    if all_nan:
        for idx in range(n_origins_to_plot):
            ax = axes[idx]
            ax.text(
                0.5,
                0.5,
                "データなし\n(MCMCを十分なイテレーションで\n実行してください)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            # Mark site locations only
            ax.scatter(
                site_coords[:, 0],
                site_coords[:, 1],
                c="blue",
                s=15,
                marker="^",
                alpha=0.6,
                label="遺跡",
            )
            plotter.plot_boundary(ax)
            ax.set_title(origins[idx], fontsize=TITLE_FONTSIZE)
            ax.set_xticks([])
            ax.set_yticks([])
            set_spine_width(ax, 0.3)
    else:
        # Find global max for consistent colorbar (ignoring NaN)
        vmax = np.nanmax(valid_data)
        if np.isnan(vmax) or vmax == 0:
            vmax = 0.1  # Default fallback

        for idx in range(n_origins_to_plot):
            ax = axes[idx]

            sc = ax.scatter(
                grid_coords[plotter.is_land, 0],
                grid_coords[plotter.is_land, 1],
                c=posterior_std[idx, plotter.is_land],
                cmap=CMAP_INTENSITY,
                s=8,
                alpha=0.8,
                vmin=0,
                vmax=vmax,
            )

            # Mark site locations
            ax.scatter(
                site_coords[:, 0],
                site_coords[:, 1],
                c="blue",
                s=15,
                marker="^",
                alpha=0.6,
                label="遺跡",
            )

            plotter.plot_boundary(ax)

            plt.colorbar(sc, ax=ax, label="事後標準偏差", shrink=0.8)
            ax.set_title(origins[idx], fontsize=TITLE_FONTSIZE)
            ax.set_xticks([])
            ax.set_yticks([])
            set_spine_width(ax, 0.3)

    fig.suptitle(
        f"推定の不確実性（{period_name}）",
        fontsize=SUPTITLE_FONTSIZE,
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes
