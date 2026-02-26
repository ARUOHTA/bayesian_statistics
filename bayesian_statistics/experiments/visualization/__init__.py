"""Visualization modules for spatial plots and diagnostics."""

from .diagnostics import (
    plot_beta_trace,
    plot_lambda_star_diagnostics,
    print_lambda_star_summary,
    print_pseudo_absence_summary,
)
from .effect_decomposition import (
    plot_all_origins_single_effect,
    plot_effect_comparison,
    plot_effect_decomposition_grid,
)
from .map_template import MapPlotter
from .period_origin_grid import (
    plot_all_origins_all_periods,
    plot_all_periods_for_origin,
    plot_effect_by_periods_and_origins,
)
from .scatter_plots import (
    compute_metrics,
    plot_estimated_vs_observed,
    plot_loocv_comparison,
    print_metrics_summary,
)
from .spatial_plots import (
    plot_all_origins_grid,
    plot_comparison,
    plot_distance_prior,
    plot_intensity_map,
    plot_model_comparison_map,
    plot_single_origin_map,
    plot_site_probability_map,
    plot_study_area_map,
    plot_study_area_with_sources,
    plot_uncertainty_map,
)
from .style import (
    CMAP_DIVERGING,
    CMAP_INTENSITY,
    CMAP_PRIOR,
    CMAP_PROBABILITY,
    FIGURE_DPI,
    FIGURE_FORMAT,
    apply_japanese_font,
    setup_matplotlib_style,
)

__all__ = [
    # Map template
    "MapPlotter",
    # Style
    "apply_japanese_font",
    "setup_matplotlib_style",
    "FIGURE_DPI",
    "FIGURE_FORMAT",
    "CMAP_PROBABILITY",
    "CMAP_INTENSITY",
    "CMAP_PRIOR",
    "CMAP_DIVERGING",
    # Spatial plots
    "plot_single_origin_map",
    "plot_all_origins_grid",
    "plot_intensity_map",
    "plot_comparison",
    # Effect decomposition
    "plot_effect_comparison",
    "plot_all_origins_single_effect",
    "plot_effect_decomposition_grid",
    # Period x Origin grid
    "plot_effect_by_periods_and_origins",
    "plot_all_periods_for_origin",
    "plot_all_origins_all_periods",
    # Scatter plots
    "plot_estimated_vs_observed",
    "compute_metrics",
    "print_metrics_summary",
    "plot_loocv_comparison",
    # Additional spatial plots
    "plot_distance_prior",
    "plot_model_comparison_map",
    "plot_site_probability_map",
    "plot_study_area_map",
    "plot_study_area_with_sources",
    "plot_uncertainty_map",
    # Diagnostics
    "plot_lambda_star_diagnostics",
    "print_lambda_star_summary",
    "print_pseudo_absence_summary",
    "plot_beta_trace",
]
