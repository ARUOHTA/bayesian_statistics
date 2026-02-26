"""Tests for experiment configuration."""

from pathlib import Path

import pytest


def test_config_default_values():
    """デフォルト値が正しく設定されることを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig()

    # パス設定
    assert config.data_dir == Path("data")
    assert config.output_dir == Path("bayesian_statistics/experiments/output")

    # 時期・産地定義
    assert len(config.time_periods) == 5
    assert config.time_periods[0] == "早期・早々期"
    assert config.time_periods[2] == "中期"
    assert len(config.origins) == 5
    assert "神津島" in config.origins

    # MMCPハイパーパラメータ
    assert config.tau == 0.5
    assert config.alpha == 1.0
    assert config.n_iter == 500  # 開発用デフォルト値
    assert config.burn_in == 100  # 開発用デフォルト値
    assert config.thinning == 2
    assert config.neighbor_count == 25

    # LOOCVサブサンプル設定
    assert config.loocv_n_samples == 20
    assert config.loocv_seed == 42

    # 図設定
    assert config.figure_dpi == 300
    assert config.figure_format == "png"


def test_config_time_periods():
    """時期定義が5つあることを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig()

    assert len(config.time_periods) == 5
    expected_periods = {
        0: "早期・早々期",
        1: "前期",
        2: "中期",
        3: "後期",
        4: "晩期",
    }
    assert config.time_periods == expected_periods


def test_config_origins():
    """産地が正しく定義されていることを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig()

    expected_origins = ["神津島", "信州", "箱根", "高原山", "その他"]
    assert config.origins == expected_origins


def test_config_distance_columns():
    """距離列名が正しく定義されていることを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig()

    expected_columns = [
        "cost_kouzu",
        "cost_shinshu",
        "cost_hakone",
        "cost_takahara",
    ]
    assert config.distance_column_names == expected_columns


def test_config_source_weights():
    """産地の重要度が正しく定義されていることを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig()

    assert config.source_weights == [2, 0.5, 0.01, 0.01]


def test_config_custom_values():
    """カスタム値が正しく設定されることを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig(
        n_iter=500,
        burn_in=100,
        tau=0.3,
        loocv_n_samples=10,
    )

    assert config.n_iter == 500
    assert config.burn_in == 100
    assert config.tau == 0.3
    assert config.loocv_n_samples == 10


def test_config_n_saved():
    """保存サンプル数の計算が正しいことを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig(n_iter=1000, burn_in=200, thinning=2)
    assert config.n_saved() == 400

    config2 = ExperimentConfig(n_iter=500, burn_in=100, thinning=4)
    assert config2.n_saved() == 100


def test_config_kernel_params():
    """カーネルパラメータが正しく定義されていることを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig()

    assert config.mark_lengthscale == 0.2
    assert config.mark_variance == 0.1
    assert config.intensity_lengthscale == 0.1
    assert config.intensity_variance == 1.0


def test_config_verbosity_default():
    """verbosityのデフォルト値を確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig()
    assert config.verbosity == 1


def test_config_verbosity_custom():
    """verbosityのカスタム値を確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig(verbosity=0)
    assert config.verbosity == 0

    config2 = ExperimentConfig(verbosity=2)
    assert config2.verbosity == 2


def test_config_validate_n_iter_less_than_burn_in():
    """n_iter <= burn_in の場合に警告を返すことを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    # n_iter == burn_in: サンプル数が0
    config = ExperimentConfig(n_iter=50, burn_in=50)
    warnings = config.validate()
    assert len(warnings) >= 1
    assert "サンプルが保存されません" in warnings[0]

    # n_iter < burn_in: サンプル数が負（0に丸められる）
    config2 = ExperimentConfig(n_iter=30, burn_in=50)
    warnings2 = config2.validate()
    assert len(warnings2) >= 1
    assert "サンプルが保存されません" in warnings2[0]


def test_config_validate_low_n_saved():
    """n_savedが少ない場合に警告を返すことを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    # n_saved = (60 - 50) // 2 = 5 < 10
    config = ExperimentConfig(n_iter=60, burn_in=50, thinning=2)
    warnings = config.validate()
    assert len(warnings) >= 1
    assert "少なすぎる" in warnings[0]


def test_config_validate_valid_config():
    """有効な設定で警告がないことを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig(n_iter=300, burn_in=50, thinning=2)
    warnings = config.validate()
    assert len(warnings) == 0


def test_config_adjust_for_testing():
    """adjust_for_testingがburn_inとthinningを適切に調整することを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig(n_iter=1000, burn_in=500, thinning=10)

    # n_iter=50に調整
    config.adjust_for_testing(50)

    assert config.n_iter == 50
    # burn_in は n_iter // 2 = 25 以下
    assert config.burn_in <= 25
    # thinning は (n_iter - burn_in) // 10 以下、かつ少なくとも1
    assert config.thinning >= 1
    # 少なくとも10サンプルは保存される
    assert config.n_saved() >= 10


def test_config_adjust_for_testing_small_n_iter():
    """非常に小さいn_iterでもサンプルが保存されることを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig()
    config.adjust_for_testing(20)

    assert config.n_iter == 20
    assert config.n_saved() >= 1  # 少なくとも1サンプルは保存


def test_config_n_saved_returns_non_negative():
    """n_saved()が負の値を返さないことを確認"""
    from bayesian_statistics.experiments.config import ExperimentConfig

    config = ExperimentConfig(n_iter=10, burn_in=100, thinning=2)
    assert config.n_saved() >= 0
