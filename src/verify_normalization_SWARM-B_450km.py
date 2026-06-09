"""
verify_normalization_SWARM-B_450km.py

目的:
    MSIS正規化が正しく機能しているか4つの観点から検証する。

検証項目:
    ① norm_ratio のヒストグラム／統計チェック
       - norm_ratio = rho_model_ref / rho_model_real
       - 物理的に妥当な範囲（例: 0.5–5.0）に収まっているか

    ② F10.7・Ap・高度と密度の相関チェック
       - 正規化前後でF10.7/Ap/高度との相関係数が減少するか

    ③ 緯度プロファイルの前後比較（non-SSW期間: DOY35-40）
       - 正規化後、高度や太陽活動のlatitudinal biasが消えるか

    ④ MSISモデル vs 観測の散布図
       - rho_obs vs rho_model_real が一直線になるか（モデル精度確認）

入力:
    integrateddata/swarm_dnsbpod_2018_DOY20-80.parquet
    normalizeddata/swarm_dnsbpod_2018_normalized_DOY20-80(450km).parquet

出力:
    Figure/verify_normalization_SWARM-B_450km.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ヘッドレス実行（plt.show()ブロック回避）
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ============================================================
# 設定
# ============================================================
RAW_PARQUET  = Path("integrateddata/swarm_dnsbpod_2018_DOY20-80.parquet")
NORM_PARQUET = Path("normalizeddata/swarm_dnsbpod_2018_normalized_DOY20-80(450km).parquet")
KPINDEX_CSV  = Path("data/Kpindex/SW-20180120_20180320.csv")
OUT_PNG      = Path("Figure/verify_normalization_SWARM-B_450km.png")

LAT_MIN, LAT_MAX = -60.0, 60.0
DOY_START, DOY_END = 20, 80

# Non-SSW reference period for profile check
DOY_REF_START, DOY_REF_END = 35, 40

# ============================================================
# データ読み込み
# ============================================================
def load_data():
    print("Loading normalized parquet ...")
    df_norm = pd.read_parquet(NORM_PARQUET)
    df_norm["datetime"] = pd.to_datetime(df_norm["datetime"], utc=True)
    df_norm = df_norm.dropna(subset=["datetime", "lat", "density_norm"]).copy()
    df_norm["DOY_int"] = df_norm["datetime"].dt.dayofyear
    df_norm = df_norm[
        (df_norm["lat"].abs() <= LAT_MAX) &
        (df_norm["DOY_int"] >= DOY_START) &
        (df_norm["DOY_int"] <= DOY_END)
    ].copy()
    print(f"  normalized: {len(df_norm):,} rows")

    print("Loading raw parquet ...")
    df_raw = pd.read_parquet(RAW_PARQUET)
    # datetime may be in the index (DatetimeIndex) rather than a column
    if "datetime" not in df_raw.columns:
        df_raw = df_raw.reset_index()
        if df_raw.columns[0] != "datetime":
            df_raw = df_raw.rename(columns={df_raw.columns[0]: "datetime"})
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], utc=True)
    df_raw = df_raw.dropna(subset=["datetime", "lat", "density"]).copy()
    df_raw["DOY_int"] = df_raw["datetime"].dt.dayofyear
    df_raw = df_raw[
        (df_raw["lat"].abs() <= LAT_MAX) &
        (df_raw["DOY_int"] >= DOY_START) &
        (df_raw["DOY_int"] <= DOY_END)
    ].copy()
    print(f"  raw:        {len(df_raw):,} rows")

    return df_norm, df_raw


# ============================================================
# 検証① norm_ratio ヒストグラム
# ============================================================
def plot_ratio_histogram(ax: plt.Axes, df_norm: pd.DataFrame) -> None:
    if "norm_ratio_model_ref_over_real" not in df_norm.columns:
        ax.text(0.5, 0.5, "norm_ratio column not found", transform=ax.transAxes,
                ha="center", va="center", color="red")
        ax.set_title("① norm_ratio (not available)")
        return

    ratio = df_norm["norm_ratio_model_ref_over_real"].dropna().to_numpy()
    ratio = ratio[np.isfinite(ratio)]

    p1,  p5,  p50, p95, p99 = np.percentile(ratio, [1, 5, 50, 95, 99])
    n_outlier = np.sum((ratio < 0.3) | (ratio > 10.0))

    ax.hist(ratio, bins=100, color="#2A6AE0", alpha=0.75, edgecolor="none")
    ax.axvline(p5,  color="orange", lw=1.5, linestyle="--", label=f"5th = {p5:.3f}")
    ax.axvline(p50, color="red",    lw=1.5, linestyle="-",  label=f"median = {p50:.3f}")
    ax.axvline(p95, color="orange", lw=1.5, linestyle="--", label=f"95th = {p95:.3f}")
    ax.axvline(1.0, color="gray",   lw=1.0, linestyle=":",  label="ratio = 1.0")
    ax.set_xlabel("norm_ratio (model_ref / model_real)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"① norm_ratio distribution\n"
                 f"[1st, 99th] = [{p1:.3f}, {p99:.3f}]  |  outliers (<0.3 or >10): {n_outlier}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# ============================================================
# 検証② F10.7・Ap・高度との相関
# ============================================================
def plot_correlation_bars(ax: plt.Axes, df_norm: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """正規化前後でF10.7・Ap・高度との相関係数を比較"""

    # Kp/Apデータを読み込んでマージ
    try:
        df_geo = pd.read_csv(KPINDEX_CSV, parse_dates=["DATE"])
        f107_col = next((c for c in ["F10.7_ADJ", "F10.7_OBS", "F107"] if c in df_geo.columns), None)
        if f107_col is None:
            raise KeyError(f"F10.7 column not found. Available: {list(df_geo.columns)}")
        # "F107_geo" という名前でマージ（normデータのF107列と衝突を避ける）
        df_geo = df_geo[["DATE", f107_col, "AP_AVG"]].copy()
        df_geo.rename(columns={f107_col: "F107_geo", "AP_AVG": "Ap"}, inplace=True)
        df_geo["DATE"] = pd.to_datetime(df_geo["DATE"], utc=True).dt.floor("D").dt.tz_localize(None)
    except Exception as e:
        ax.text(0.5, 0.5, f"Kp CSV load error:\n{e}", transform=ax.transAxes,
                ha="center", va="center", color="red", fontsize=8)
        return

    def merge_geo(df, density_col):
        d = df.copy()
        d["DATE"] = d["datetime"].dt.floor("D").dt.tz_localize(None)
        # normデータにF107が既にある場合はそれを再利用し、Apのみgeoからマージ
        if "F107" in d.columns:
            d["F107_geo"] = d["F107"]
            d = d.merge(df_geo[["DATE", "Ap"]], on="DATE", how="left")
        else:
            d = d.merge(df_geo, on="DATE", how="left")
        return d.dropna(subset=[density_col, "F107_geo", "Ap"])

    df_raw_m  = merge_geo(df_raw,  "density")
    df_norm_m = merge_geo(df_norm, "density_norm")

    # 高度列名を特定
    alt_col_raw = next((c for c in ["alt_km", "altitude_km", "altitude_m"] if c in df_raw_m.columns), None)
    alt_col_norm = next((c for c in ["alt_km", "altitude_km", "altitude_m"] if c in df_norm_m.columns), None)

    def safe_corr(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            return 0.0
        r, _ = stats.pearsonr(x[mask], y[mask])
        return r

    variables = ["F10.7", "Ap", "altitude"]
    r_before, r_after = [], []

    r_before.append(safe_corr(df_raw_m["F107_geo"].to_numpy(), df_raw_m["density"].to_numpy()))
    r_after.append(safe_corr(df_norm_m["F107_geo"].to_numpy(), df_norm_m["density_norm"].to_numpy()))

    r_before.append(safe_corr(df_raw_m["Ap"].to_numpy(), df_raw_m["density"].to_numpy()))
    r_after.append(safe_corr(df_norm_m["Ap"].to_numpy(), df_norm_m["density_norm"].to_numpy()))

    # 高度の相関
    if alt_col_raw and alt_col_norm:
        alt_raw  = df_raw_m[alt_col_raw].to_numpy()
        if alt_col_raw == "altitude_m":
            alt_raw = alt_raw / 1000.0
        r_before.append(safe_corr(alt_raw, df_raw_m["density"].to_numpy()))

        alt_norm = df_norm_m[alt_col_norm].to_numpy()
        if alt_col_norm == "altitude_m":
            alt_norm = alt_norm / 1000.0
        r_after.append(safe_corr(alt_norm, df_norm_m["density_norm"].to_numpy()))
    else:
        variables = ["F10.7", "Ap"]

    x = np.arange(len(variables))
    width = 0.35

    bars_before = ax.bar(x - width/2, r_before, width, label="Before norm.", color="#E05C2A", alpha=0.80)
    bars_after  = ax.bar(x + width/2, r_after,  width, label="After norm.",  color="#2A6AE0", alpha=0.80)

    # 値をバーの上に表示
    for bar, val in zip(bars_before, r_before):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * np.sign(val),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    for bar, val in zip(bars_after, r_after):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * np.sign(val),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8, color="#2A6AE0")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=10)
    ax.set_title("② Correlation with F10.7 / Ap / altitude\n(should decrease after normalization)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", alpha=0.3)


# ============================================================
# 検証③ 緯度プロファイルの前後比較（non-SSW期間）
# ============================================================
def plot_lat_profile(ax: plt.Axes, df_norm: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    LAT_BIN = 3.0
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    # non-SSW reference期間に絞る
    mask_ref = (df_raw["DOY_int"] >= DOY_REF_START) & (df_raw["DOY_int"] <= DOY_REF_END)
    mask_ref_norm = (df_norm["DOY_int"] >= DOY_REF_START) & (df_norm["DOY_int"] <= DOY_REF_END)

    df_r = df_raw[mask_ref].copy()
    df_n = df_norm[mask_ref_norm].copy()

    def lat_median(df, col):
        df["lat_bin"] = pd.cut(df["lat"], bins=lat_bins, labels=lat_centers)
        return df.groupby("lat_bin", observed=True)[col].median().to_numpy(dtype=float)

    prof_raw  = lat_median(df_r, "density")
    prof_norm = lat_median(df_n, "density_norm")

    # 正規化してプロット（各プロファイルを平均=1に正規化）
    prof_raw_rel  = prof_raw  / np.nanmean(prof_raw)
    prof_norm_rel = prof_norm / np.nanmean(prof_norm)

    ax.plot(lat_centers, prof_raw_rel,  color="#E05C2A", lw=2.0,
            marker="o", markersize=4, label="Before norm.")
    ax.plot(lat_centers, prof_norm_rel, color="#2A6AE0", lw=2.0,
            marker="s", markersize=4, label="After norm.")
    ax.axhline(1.0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Latitude (°)", fontsize=10)
    ax.set_ylabel("Relative density (mean=1)", fontsize=10)
    ax.set_title(f"③ Lat. profile in non-SSW ref. period\n"
                 f"(DOY {DOY_REF_START}–{DOY_REF_END}, should be flatter after norm.)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(LAT_MIN, LAT_MAX)
    ax.grid(alpha=0.3)


# ============================================================
# 検証④ MSISモデル vs 観測の散布図
# ============================================================
def plot_model_vs_obs(ax: plt.Axes, df_norm: pd.DataFrame) -> None:
    """
    rho_obs と rho_model_real の散布図。
    norm_ratio = model_ref / model_real なので、
    rho_model_real = rho_obs / density_norm * norm_ratio^-1 ... ではなく
    density_norm = rho_obs * (model_ref / model_real)
    → rho_model_real は保存されていないので、rho_obs vs density_norm の比で代用
    または raw+norm の両データをマージして比較
    """
    # normデータ内のratio列から model_real を逆算
    if "norm_ratio_model_ref_over_real" not in df_norm.columns:
        ax.text(0.5, 0.5, "norm_ratio column not found\nCannot reconstruct rho_model_real",
                transform=ax.transAxes, ha="center", va="center", color="red")
        ax.set_title("④ MSIS model vs obs (not available)")
        return

    # rho_obs * ratio = density_norm → rho_obs = density_norm / ratio
    # また rho_model_real = rho_obs / density_norm * norm_ratio^(-1) ... 違う
    # density_norm = rho_obs * (model_ref / model_real)
    # → model_real の絶対値は保存されていないが、
    #   rho_obs と density_norm の比が model_real / model_ref に比例している

    # RAW parquetから rho_obs を読んでマージ
    try:
        df_raw_tmp = pd.read_parquet(RAW_PARQUET, columns=["density"])
        # datetime is in the index for raw parquet
        if "datetime" not in df_raw_tmp.columns:
            df_raw_tmp = df_raw_tmp.reset_index()
            if df_raw_tmp.columns[0] != "datetime":
                df_raw_tmp = df_raw_tmp.rename(columns={df_raw_tmp.columns[0]: "datetime"})
        df_raw_tmp["datetime"] = pd.to_datetime(df_raw_tmp["datetime"], utc=True)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error loading raw:\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        return

    # datetimeをキーにマージ（exact match）
    df_merged = df_norm[["datetime", "density_norm", "norm_ratio_model_ref_over_real"]].merge(
        df_raw_tmp, on="datetime", how="inner"
    ).dropna()

    ratio = df_merged["norm_ratio_model_ref_over_real"].to_numpy()
    rho_obs  = df_merged["density"].to_numpy()
    rho_norm = df_merged["density_norm"].to_numpy()

    # rho_model_real を逆算: density_norm = rho_obs * ratio → ratio = density_norm / rho_obs
    # rho_model_real ∝ 1/ratio（相対的に）
    # より厳密には: density_norm = rho_obs * (model_ref / model_real)
    #               → model_real = rho_obs * model_ref / density_norm
    # model_ref の絶対値は不明だが、rho_obs と ratio から model_real/model_ref = 1/ratio

    # ここでは rho_obs vs density_norm の散布図で「正規化がどの程度変換したか」を見る
    # サンプリング（全点だと重いので10%）
    n = len(df_merged)
    idx = np.random.choice(n, min(n, 20000), replace=False)
    x = rho_obs[idx]
    y = rho_norm[idx]

    # 1:1ラインとの比較
    vmax = np.nanpercentile(np.concatenate([x, y]), 99)
    vmin = np.nanpercentile(np.concatenate([x, y]), 1)

    ax.scatter(x, y, s=1, alpha=0.2, color="#2A6AE0", rasterized=True)
    ax.plot([vmin, vmax], [vmin, vmax], color="red", lw=1.5, linestyle="--", label="1:1 line")

    # 相関係数
    r, _ = stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f"r = {r:.4f}", transform=ax.transAxes,
            fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel(r"$\rho_\mathrm{obs}$ (raw) [kg m$^{-3}$]", fontsize=10)
    ax.set_ylabel(r"$\rho_\mathrm{norm}$ [kg m$^{-3}$]", fontsize=10)
    ax.set_title("④ Raw obs. vs Normalized density\n"
                 "(scatter from 1:1 shows correction applied)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ============================================================
# メイン
# ============================================================
def main() -> None:
    np.random.seed(42)

    df_norm, df_raw = load_data()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Normalization Quality Check — Swarm-B (ref: 450 km, F10.7=70, Ap=4)\n"
        "DOY 20–80, 2018  |  lat: –60° to +60°",
        fontsize=14, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    print("① Plotting norm_ratio histogram ...")
    plot_ratio_histogram(ax1, df_norm)

    print("② Plotting correlation bars ...")
    plot_correlation_bars(ax2, df_norm, df_raw)

    print("③ Plotting latitude profile ...")
    plot_lat_profile(ax3, df_norm, df_raw)

    print("④ Plotting raw vs normalized scatter ...")
    plot_model_vs_obs(ax4, df_norm)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
