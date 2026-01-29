"""Period x Origin grid visualization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .map_template import MapPlotter
from .style import (
    CMAP_DIVERGING,
    CMAP_PROBABILITY,
    LABEL_FONTSIZE,
    SPINE_WIDTH_THIN,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    set_spine_width,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_effect_by_periods_and_origins(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    all_effects: Dict[int, Dict[str, np.ndarray]],
    all_datasets: Dict[int, Any],
    effect_name: str,
    periods: List[int],
    origin_indices: List[int],
    origins: List[str],
    time_periods: Dict[int, str],
    scatter: bool = False,
    cmap: str = CMAP_PROBABILITY,
    vcenter: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
    weight_by: Optional[Dict[int, np.ndarray]] = None,
    title_suffix: str = "",
) -> tuple:
    """Plot effect comparison across periods and origins.

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    all_effects : dict
        Dictionary mapping period -> effect dict.
    all_datasets : dict
        Dictionary mapping period -> dataset.
    effect_name : str
        Name of effect to plot (e.g., "distance", "intercept_adjustment", "full").
    periods : list of int
        List of period indices to plot.
    origin_indices : list of int
        List of origin indices to plot.
    origins : list of str
        Names of all origins.
    time_periods : dict
        Mapping from period index to period name.
    scatter : bool
        Whether to overlay site scatter points.
    cmap : str
        Colormap name.
    vcenter : float, optional
        Center value for TwoSlopeNorm.
    output_path : str or Path, optional
        Path to save the figure.
    weight_by : dict, optional
        Dictionary mapping period -> weight array (n_grid,).
        If provided, effect values are multiplied element-wise by weights.
    title_suffix : str, optional
        Suffix to append to the figure title.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : np.ndarray
        Array of matplotlib axes.
    """
    n_periods = len(periods)
    n_origins = len(origin_indices)

    fig, axes = plt.subplots(
        n_origins,
        n_periods,
        figsize=(5 * n_periods, 5 * n_origins),
        constrained_layout=True,
        squeeze=False,
    )

    # Compute global min/max for consistent colormap
    max_val = -np.inf
    min_val = np.inf
    for p_idx in periods:
        effect_data = all_effects[p_idx][effect_name]  # (K, n_grid)
        for origin_idx in origin_indices:
            effect_k = effect_data[origin_idx]  # (n_grid,)
            # Apply weights if provided (mask deviation from vcenter)
            if weight_by is not None and p_idx in weight_by:
                if vcenter is not None:
                    deviation = effect_k - vcenter
                    effect_k = vcenter + deviation * weight_by[p_idx]
                else:
                    effect_k = effect_k * weight_by[p_idx]
            valid_values = effect_k[plotter.is_land]
            if len(valid_values) > 0:
                max_val = max(max_val, np.nanmax(valid_values))
                min_val = min(min_val, np.nanmin(valid_values))

    # Setup normalization
    if vcenter is not None:
        # Center at vcenter with symmetric range
        max_diff = max(abs(max_val - vcenter), abs(vcenter - min_val))
        max_val = vcenter + max_diff
        min_val = vcenter - max_diff
        norm = TwoSlopeNorm(vmin=min_val, vcenter=vcenter, vmax=max_val)
        cmap = CMAP_DIVERGING
    else:
        norm = None

    for row_idx, origin_idx in enumerate(origin_indices):
        for col_idx, p_idx in enumerate(periods):
            ax = axes[row_idx, col_idx]

            # Get effect data and apply weights if provided (mask deviation from vcenter)
            effect_data = all_effects[p_idx][effect_name]
            effect_k = effect_data[origin_idx]  # (n_grid,)
            if weight_by is not None and p_idx in weight_by:
                if vcenter is not None:
                    deviation = effect_k - vcenter
                    effect_k = vcenter + deviation * weight_by[p_idx]
                else:
                    effect_k = effect_k * weight_by[p_idx]

            # Plot grid scatter
            scatter_kwargs = {
                "c": effect_k[plotter.is_land],
                "cmap": cmap,
                "s": 8,
                "alpha": 0.8,
            }
            if norm is not None:
                scatter_kwargs["norm"] = norm
            else:
                scatter_kwargs["vmin"] = min_val
                scatter_kwargs["vmax"] = max_val

            sc = ax.scatter(
                grid_coords[plotter.is_land, 0],
                grid_coords[plotter.is_land, 1],
                **scatter_kwargs,
            )

            # Overlay sites if requested or if effect is "full"
            if effect_name == "full" or scatter:
                dataset_p = all_datasets[p_idx]
                true_ratio_p = dataset_p.counts / dataset_p.counts.sum(
                    axis=1, keepdims=True
                )
                true_ratio_p = np.nan_to_num(true_ratio_p).T

                site_kwargs = {
                    "c": true_ratio_p[origin_idx],
                    "cmap": cmap,
                    "s": 20,
                    "edgecolors": "black",
                    "linewidths": 0.3,
                    "alpha": 0.8,
                }
                if norm is not None:
                    site_kwargs["norm"] = norm
                else:
                    site_kwargs["vmin"] = min_val
                    site_kwargs["vmax"] = max_val

                # Get site coordinates from dataset
                if hasattr(dataset_p, "site_coords"):
                    site_coords = dataset_p.site_coords
                elif hasattr(dataset_p, "coords"):
                    site_coords = dataset_p.coords
                else:
                    site_coords = None

                if site_coords is not None:
                    ax.scatter(
                        site_coords[:, 0],
                        site_coords[:, 1],
                        **site_kwargs,
                    )

            # Plot boundary
            plotter.plot_boundary(ax)

            # Labels
            if row_idx == 0:
                ax.set_title(time_periods[p_idx], fontsize=TITLE_FONTSIZE)
            if col_idx == 0:
                ax.set_ylabel(origins[origin_idx], fontsize=LABEL_FONTSIZE)

            ax.set_xticks([])
            ax.set_yticks([])
            set_spine_width(ax, SPINE_WIDTH_THIN)

    # Common colorbar
    fig.colorbar(sc, ax=axes, label="確率", shrink=0.7, pad=0.02)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_all_periods_for_origin(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    all_grid_probs: Dict[int, np.ndarray],
    all_datasets: Dict[int, Any],
    origin_index: int,
    origins: List[str],
    time_periods: Dict[int, str],
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot all periods for a single origin in a horizontal row.

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    all_grid_probs : dict
        Dictionary mapping period -> grid probabilities (K, n_grid).
    all_datasets : dict
        Dictionary mapping period -> dataset.
    origin_index : int
        Index of the origin to plot.
    origins : list of str
        Names of all origins.
    time_periods : dict
        Mapping from period index to period name.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : list
        List of matplotlib axes.
    """
    n_periods = len(time_periods)
    fig, axes = plt.subplots(
        1, n_periods, figsize=(5 * n_periods, 5), constrained_layout=True
    )

    for p_idx, ax in enumerate(axes):
        grid_probs_p = all_grid_probs[p_idx]
        dataset_p = all_datasets[p_idx]

        # True ratio
        true_ratio_p = dataset_p.counts / dataset_p.counts.sum(axis=1, keepdims=True)
        true_ratio_p = np.nan_to_num(true_ratio_p).T

        # Plot grid probabilities
        sc = plotter.plot_grid_scatter(
            ax=ax,
            coords=grid_coords,
            values=grid_probs_p[origin_index],
        )

        # Plot site true ratios
        if hasattr(dataset_p, "site_coords"):
            site_coords = dataset_p.site_coords
        elif hasattr(dataset_p, "coords"):
            site_coords = dataset_p.coords
        else:
            site_coords = None

        if site_coords is not None:
            plotter.plot_site_scatter(
                ax=ax,
                coords=site_coords,
                values=true_ratio_p[origin_index],
                s=25,
                linewidths=0.5,
            )

        plotter.plot_boundary(ax)

        ax.set_title(f"{time_periods[p_idx]} (n={dataset_p.num_sites()})")
        ax.set_xlabel("経度")
        ax.set_ylabel("緯度")
        set_spine_width(ax, SPINE_WIDTH_THIN)

    # Common colorbar
    fig.colorbar(sc, ax=axes, label="事後平均", shrink=0.8)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_all_origins_all_periods(
    plotter: MapPlotter,
    grid_coords: np.ndarray,
    all_grid_probs: Dict[int, np.ndarray],
    all_datasets: Dict[int, Any],
    origins: List[str],
    time_periods: Dict[int, str],
    output_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Plot all origins x all periods in a grid (4 rows x 5 cols).

    Parameters
    ----------
    plotter : MapPlotter
        Map plotter instance.
    grid_coords : np.ndarray
        Grid coordinates (n_grid, 2).
    all_grid_probs : dict
        Dictionary mapping period -> grid probabilities (K, n_grid).
    all_datasets : dict
        Dictionary mapping period -> dataset.
    origins : list of str
        Names of all origins (first 4 used).
    time_periods : dict
        Mapping from period index to period name.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : np.ndarray
        Array of matplotlib axes.
    """
    n_origins = 4  # First 4 origins
    n_periods = len(time_periods)

    fig, axes = plt.subplots(
        n_origins, n_periods, figsize=(25, 20), constrained_layout=True
    )

    for origin_idx in range(n_origins):
        for p_idx in range(n_periods):
            ax = axes[origin_idx, p_idx]
            grid_probs_p = all_grid_probs[p_idx]
            dataset_p = all_datasets[p_idx]

            # True ratio
            true_ratio_p = dataset_p.counts / dataset_p.counts.sum(
                axis=1, keepdims=True
            )
            true_ratio_p = np.nan_to_num(true_ratio_p).T

            # Plot grid probabilities
            sc = ax.scatter(
                grid_coords[plotter.is_land, 0],
                grid_coords[plotter.is_land, 1],
                c=grid_probs_p[origin_idx, plotter.is_land],
                cmap=CMAP_PROBABILITY,
                s=5,
                alpha=0.8,
                vmin=0,
                vmax=1,
            )

            # Plot sites
            if hasattr(dataset_p, "site_coords"):
                site_coords = dataset_p.site_coords
            elif hasattr(dataset_p, "coords"):
                site_coords = dataset_p.coords
            else:
                site_coords = None

            if site_coords is not None:
                ax.scatter(
                    site_coords[:, 0],
                    site_coords[:, 1],
                    c=true_ratio_p[origin_idx],
                    cmap=CMAP_PROBABILITY,
                    s=15,
                    edgecolors="black",
                    linewidths=0.3,
                    alpha=0.8,
                    vmin=0,
                    vmax=1,
                )

            plotter.plot_boundary(ax)

            # Row label (origin name) on left only
            if p_idx == 0:
                ax.set_ylabel(origins[origin_idx], fontsize=LABEL_FONTSIZE)

            # Column label (period name) on top only
            if origin_idx == 0:
                ax.set_title(time_periods[p_idx], fontsize=TICK_FONTSIZE)

            ax.set_xticks([])
            ax.set_yticks([])
            set_spine_width(ax, 0.1)

    # Common colorbar
    fig.colorbar(sc, ax=axes, label="事後平均確率", shrink=0.6, pad=0.02)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes
