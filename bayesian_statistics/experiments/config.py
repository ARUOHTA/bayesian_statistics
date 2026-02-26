"""Experiment configuration for master thesis Chapter 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for the experiment.

    Attributes
    ----------
    data_dir : Path
        Directory containing input data files.
    output_dir : Path
        Directory for output files (figures, tables, results).
    time_periods : dict[int, str]
        Mapping from period index to period name.
    origins : list[str]
        List of obsidian origin names.
    distance_column_names : list[str]
        Column names for distance features.
    source_weights : list[float]
        Importance weights for each origin.
    tau : float
        Temperature parameter for distance prior.
    alpha : float
        Importance exponent for distance prior.
    n_iter : int
        Total number of MCMC iterations.
    burn_in : int
        Number of burn-in iterations to discard.
    thinning : int
        Thinning interval for saved samples.
    neighbor_count : int
        Number of neighbors for NNGP.
    mark_lengthscale : float
        Lengthscale for mark kernel.
    mark_variance : float
        Variance for mark kernel.
    intensity_lengthscale : float
        Lengthscale for intensity kernel.
    intensity_variance : float
        Variance for intensity kernel.
    loocv_n_samples : int
        Number of subsamples for LOOCV.
    loocv_seed : int
        Random seed for LOOCV subsampling.
    figure_dpi : int
        DPI for saved figures.
    figure_format : str
        Format for saved figures.
    """

    # パス設定
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(
        default_factory=lambda: Path("bayesian_statistics/experiments/output")
    )

    # 時期・産地定義
    time_periods: dict[int, str] = field(
        default_factory=lambda: {
            0: "早期・早々期",
            1: "前期",
            2: "中期",
            3: "後期",
            4: "晩期",
        }
    )
    origins: List[str] = field(
        default_factory=lambda: ["神津島", "信州", "箱根", "高原山", "その他"]
    )
    # 産地の座標 (経度, 緯度) - 10_2_calculate_distance.pyより
    origin_coords: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "神津島": (139.1517940530124, 34.214991203764754),
            "信州": (138.1431305514158, 36.14658805493071),
            "箱根": (139.0446125901514, 35.221867157762105),
            "高原山": (139.7766240432928, 36.900342242149065),
        }
    )
    distance_column_names: List[str] = field(
        default_factory=lambda: [
            "cost_kouzu",
            "cost_shinshu",
            "cost_hakone",
            "cost_takahara",
        ]
    )
    source_weights: List[float] = field(default_factory=lambda: [2, 0.5, 0.01, 0.01])

    # MMCPハイパーパラメータ
    tau: float = 0.5
    alpha: float = 1.0
    n_iter: int = 500
    burn_in: int = 100
    thinning: int = 2
    neighbor_count: int = 25

    # カーネルパラメータ（マーク用）
    mark_lengthscale: float = 0.2
    mark_variance: float = 0.1

    # カーネルパラメータ（強度用）
    intensity_lengthscale: float = 0.1
    intensity_variance: float = 1.0

    # 固定係数モード（点過程の強度係数）
    fixed_intensity_coefficients: bool = True  # デフォルトでTrue（非空間変動係数）
    intensity_prior_mean: float = 0.0
    intensity_prior_variance: float = 10.0

    # λ*の事前分布（Gamma(shape, rate)）
    lambda_prior_shape: float = 2.0
    lambda_prior_rate: float = 0.1

    # λの固定値（K-1=4つの非ベースラインカテゴリに対応）
    lambda_fixed: List[float] = field(default_factory=lambda: [1, 1, 1, 1])

    # グリッドサブサンプリング
    grid_subsample_ratio: float = 0.1

    # ==========================================================================
    # 使用可能な共変量一覧（ObsidianDataPreprocessorのスキーマより）
    # --------------------------------------------------------------------------
    # 地形関連:
    #   - average_elevation: 平均標高 (m)
    #   - maximum_elevation: 最大標高 (m)
    #   - minimum_elevation: 最小標高 (m)
    #   - average_slope_angle: 平均傾斜角 (度)
    #   - maximum_slope_angle: 最大傾斜角 (度)
    #   - minimum_slope_angle: 最小傾斜角 (度)
    #
    # Tobler距離（各産地までの歩行時間ベースの距離）:
    #   - cost_kouzu: 神津島までのTobler距離
    #   - cost_shinshu: 信州までのTobler距離
    #   - cost_hakone: 箱根までのTobler距離
    #   - cost_takahara: 高原山までのTobler距離
    #   - cost_river: 最寄り河川までの距離
    #
    # 移動関連:
    #   - walking_velocity: 歩行速度 (km/h)
    #   - travel_time: 移動時間
    # ==========================================================================

    # 強度モデルの共変量（点過程部分）
    intensity_variable_names: List[str] = field(
        default_factory=lambda: ["average_elevation", "average_slope_angle"]
    )

    # マークモデルの共変量（産地構成比部分）
    # None: 距離事前分布のみ使用（共変量なし）
    # 例: ["average_elevation"] で標高を共変量として使用
    mark_variable_names: List[str] | None = None

    # Nadaraya-Watsonパラメータ
    nw_sigma: float = 500.0  # グリッド間のカーネルバンド幅
    nw_sigma_for_sites: float = 0.1  # 遺跡間のカーネルバンド幅
    nw_variable_names: List[str] = field(
        default_factory=lambda: ["average_elevation"]  # 空間共変量（説明変数）
    )

    # LOOCVサブサンプル設定
    loocv_n_samples: int = 20
    loocv_seed: int = 42

    # 図設定
    figure_dpi: int = 300
    figure_format: str = "png"

    # 出力制御
    verbosity: int = 1  # 0=quiet, 1=normal, 2=verbose

    def n_saved(self) -> int:
        """Calculate number of saved samples after burn-in and thinning."""
        return max(0, (self.n_iter - self.burn_in) // self.thinning)

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings.

        Returns
        -------
        list[str]
            List of warning messages. Empty if configuration is valid.
        """
        warnings = []

        if self.n_iter <= self.burn_in:
            warnings.append(
                f"n_iter ({self.n_iter}) <= burn_in ({self.burn_in}): "
                f"サンプルが保存されません。n_iter > burn_in となるよう設定してください。"
            )
        elif self.n_saved() < 10:
            warnings.append(
                f"n_saved={self.n_saved()} は少なすぎる可能性があります。"
                f"推奨: n_iter >= {self.burn_in + 10 * self.thinning}"
            )

        if self.grid_subsample_ratio <= 0 or self.grid_subsample_ratio > 1:
            warnings.append(
                f"grid_subsample_ratio ({self.grid_subsample_ratio}) "
                f"は (0, 1] の範囲である必要があります。"
            )

        return warnings

    def adjust_for_testing(self, n_iter: int) -> None:
        """Adjust burn_in and thinning for testing with small n_iter.

        This ensures at least some samples are saved even with small n_iter.

        Parameters
        ----------
        n_iter : int
            Number of iterations to use.
        """
        self.n_iter = n_iter
        # Ensure burn_in is at most 50% of n_iter
        self.burn_in = min(self.burn_in, n_iter // 2)
        # Ensure at least 10 samples
        max_thinning = max(1, (n_iter - self.burn_in) // 10)
        self.thinning = min(self.thinning, max_thinning)
