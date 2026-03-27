# 修士論文 実験結果 再現手順書

**最終更新**: 2026-03-27

---

## 1. データの配置

データは Git リポジトリに含まれていないため、別途共有します。
以下の4点をプロジェクトルートの `data/` ディレクトリに配置してください。

```
data/
├── 11_gdf_elevation.csv                          (767 MB)
├── 11_gdf_obsidian.csv                           (5.3 MB)
├── 11_gdf_sites.csv                              (44 KB)
└── 16_tobler_distance_with_coast_50_average/
    ├── distance_siteID_0
    ├── distance_siteID_1
    │   ...
    └── distance_siteID_273                       (274ファイル, 計 ~2.8 GB)
```

距離行列は tar.gz で圧縮して共有しています。展開は以下のコマンドで行えます:

```bash
tar -xzf 16_tobler_distance_with_coast_50_average.tar.gz -C data/
```

---

## 2. 環境構築と実行

```bash
# Python 3.9 と uv (https://docs.astral.sh/uv/) が必要です

git clone https://github.com/ARUOHTA/bayesian_statistics.git
cd bayesian_statistics

# 依存関係のインストール
uv sync

# 動作確認（62件のテストが全パスすればOKです）
uv run python -m pytest -q

# 修論 第5章の全実験を一括実行
uv run python -m bayesian_statistics.experiments.run_all --all
```

全出力は `bayesian_statistics/experiments/output/` に保存されます。
全実行には数時間かかります（MCMCとLOOCVが支配的です）。

個別に実行する場合:

```bash
uv run python -m bayesian_statistics.experiments.run_all --run-models       # モデル推定のみ
uv run python -m bayesian_statistics.experiments.run_all --run-loocv        # LOOCV評価のみ
uv run python -m bayesian_statistics.experiments.run_all --generate-figures # 図表生成のみ
```

---

## 3. 出力と修論の対応

### 表

| ファイル | 修論 | 内容 |
|---------|------|------|
| `tables/table_5_1_data_summary.csv` | 表5.1 | 時期別の遺跡数・出土数 |
| `tables/table_5_2_loocv.csv` | 表5.2 | LOOCV Aitchison距離（MMCP vs NW） |
| `tables/table_5_3_lambda_posterior.csv` | 表5.3 | λ* の事後統計量 |

### 図

| ファイル | 修論 | 内容 |
|---------|------|------|
| `fig_5_0a_study_area_sources.png` | 図5.0a | 研究領域と黒曜石産地 |
| `fig_5_0b_study_area_sites.png` | 図5.0b | 研究領域と遺跡位置 |
| `fig_5_1_all_origins_periods.png` | 図5.1 | 全産地×全時期の推定構成比 |
| `fig_5_2_effect_distance.png` | 図5.2 | 距離事前分布の効果 |
| `fig_5_2b_effect_intercept_adjustment.png` | 図5.2b | データによる事前分布からの調整量 |
| `fig_5_3_estimated_vs_observed.png` | 図5.3 | 推定値 vs 観測値 |
| `fig_5_4_lambda_diagnostics.png` | 図5.4 | λ* のMCMC診断 |
| `fig_5_5_distance_prior.png` | 図5.5 | 距離事前分布の空間分布 |
| `fig_5_7_uncertainty.png` | 図5.7 | 事後不確実性マップ |
| `fig_5_8_loocv_comparison.png` | 図5.8 | LOOCV比較（MMCP vs NW） |
| `fig_5_9_site_probability.png` | 図5.9 | 遺跡存在確率 |
| `fig_5_10_weighted_intercept_adjustment.png` | 図5.10 | 重み付き切片調整量 |

---

## 4. ハイパーパラメータ

全設定は `bayesian_statistics/experiments/config.py` の `ExperimentConfig` に定義されています。

| カテゴリ | パラメータ | 値 |
|---------|-----------|-----|
| MCMC | `n_iter` / `burn_in` / `thinning` | 500 / 100 / 2 |
| NNGP | `neighbor_count` | 25 |
| グリッド | `grid_subsample_ratio` | 0.1 |
| マークカーネル | `mark_lengthscale` / `mark_variance` | 0.2 / 0.1 |
| 強度カーネル | `intensity_lengthscale` / `intensity_variance` | 0.1 / 1.0 |
| 距離事前分布 | `tau` / `alpha` | 0.5 / 1.0 |
| 距離事前分布 | `source_weights`（神津島, 信州, 箱根, 高原山） | [2, 0.5, 0.01, 0.01] |
| NW | `nw_sigma` / `nw_sigma_for_sites` | 500.0 / 0.1 |
| LOOCV | `loocv_n_samples` / `loocv_seed` | 20 / 42 |

