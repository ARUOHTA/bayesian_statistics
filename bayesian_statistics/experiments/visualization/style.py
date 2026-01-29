"""Common style constants for visualization."""

from __future__ import annotations

from typing import Any

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Default figure sizes
FIGURE_SIZE_SINGLE = (10, 8)
FIGURE_SIZE_COMPARISON = (16, 6)
FIGURE_SIZE_GRID_1X2 = (12, 5)
FIGURE_SIZE_GRID_2X2 = (12, 10)
FIGURE_SIZE_GRID_4X5 = (25, 20)

# Color maps
CMAP_PROBABILITY = "Blues"
CMAP_INTENSITY = "YlOrRd"
CMAP_PRIOR = "Reds"
CMAP_DIVERGING = "RdBu"

# Scatter plot settings
GRID_SCATTER_SIZE = 10
GRID_SCATTER_ALPHA = 0.8
SITE_SCATTER_SIZE = 40
SITE_SCATTER_ALPHA = 0.9
SITE_EDGE_WIDTH = 0.5
BOUNDARY_SIZE = 0.001
BOUNDARY_COLOR = "grey"

# Axis settings
SPINE_WIDTH = 0.5
SPINE_WIDTH_THIN = 0.3

# Text settings (統一されたフォントサイズ)
TITLE_FONTSIZE = 11  # サブプロットタイトル/列・行ヘッダー用
LABEL_FONTSIZE = 10  # 軸ラベル用
TICK_FONTSIZE = 9    # 目盛りラベル用
LEGEND_FONTSIZE = 9  # 凡例用
SUPTITLE_FONTSIZE = 16  # (未使用: suptitleは論文では削除)


def setup_matplotlib_style() -> None:
    """matplotlib のグローバルスタイル設定.

    論文用の図を生成する際に呼び出し、フォントサイズを統一する。
    """
    import matplotlib as mpl

    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titlesize"] = TITLE_FONTSIZE
    mpl.rcParams["axes.labelsize"] = LABEL_FONTSIZE
    mpl.rcParams["xtick.labelsize"] = TICK_FONTSIZE
    mpl.rcParams["ytick.labelsize"] = TICK_FONTSIZE
    mpl.rcParams["legend.fontsize"] = LEGEND_FONTSIZE

# Value ranges
PROB_VMIN = 0.0
PROB_VMAX = 1.0


def apply_japanese_font() -> None:
    """Apply Japanese font settings for matplotlib."""
    import japanize_matplotlib  # noqa: F401


def get_scatter_style(scatter_type: str = "grid") -> dict[str, Any]:
    """Get scatter plot style settings.

    Parameters
    ----------
    scatter_type : str
        Type of scatter plot: "grid", "site", or "boundary".

    Returns
    -------
    dict
        Dictionary of matplotlib scatter plot kwargs.
    """
    if scatter_type == "grid":
        return {
            "s": GRID_SCATTER_SIZE,
            "alpha": GRID_SCATTER_ALPHA,
            "cmap": CMAP_PROBABILITY,
            "vmin": PROB_VMIN,
            "vmax": PROB_VMAX,
        }
    elif scatter_type == "site":
        return {
            "s": SITE_SCATTER_SIZE,
            "alpha": SITE_SCATTER_ALPHA,
            "cmap": CMAP_PROBABILITY,
            "vmin": PROB_VMIN,
            "vmax": PROB_VMAX,
            "edgecolors": "black",
            "linewidths": SITE_EDGE_WIDTH,
        }
    elif scatter_type == "boundary":
        return {
            "s": BOUNDARY_SIZE,
            "c": BOUNDARY_COLOR,
        }
    else:
        raise ValueError(f"Unknown scatter type: {scatter_type}")


def set_spine_width(ax, width: float = SPINE_WIDTH) -> None:
    """Set the width of all spines on an axis.

    Parameters
    ----------
    ax
        Matplotlib axis object.
    width : float
        Width of the spines.
    """
    for spine in ax.spines.values():
        spine.set_linewidth(width)
