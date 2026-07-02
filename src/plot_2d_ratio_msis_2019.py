"""
plot_2d_ratio_msis_2019.py

目的:
    density_ratio_msis = rho_obs / rho_model_real を使い、
    LT・高度・F10.7・Apのモデル依存をまとめて除去した密度比を2Dプロット。

    LTセクターで絞らずに全データを使い、SSW前後の残差変動を見る。
    参照期間 (Sep9-12 & Sep20-23) の中央値を差し引いた residual もプロット。

    2019年 SH SSW: DOY 252-266 (Sep 9-23)

出力:
    Figure/2019/2D_ratio_msis_2019_SWARM-{A,B,C}.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path

# =========================
# 設定
# =========================
JOBS = [
    dict(
        label   = "SWARM-A",
        parquet = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png = Path("Figure/2019/2D_ratio_msis_2019_SWARM-A.png"),
    ),
    dict(
        label   = "SWARM-B",
        parquet = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png = Path("Figure/2019/2D_ratio_msis_2019_SWARM-B.png"),
    ),
    dict(
        label   = "SWARM-C",
        parquet = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png = Path("Figure/2019/2D_ratio_msis_2019_SWARM-C.png"),
    ),
]

T_START = "2019-09-09 00:00:00"   # DOY 252
T_END   = "2019-09-23 23:59:59"   # DOY 266

LAT_MIN, LAT_MAX = -60, 60
DOY_BIN  = 1.0
LAT_BIN  = 3.0
N_LEVELS = 21

# Non-SSW reference periods
DOY_REF1_MIN, DOY_REF1_MAX = 252, 255   # Sep 9-12
DOY_REF2_MIN, DOY_REF2_MAX = 263, 266   # Sep 20-23

# SSW period (approximate peak)
DOY_SSW_START, DOY_SSW_END = 256, 262   # Sep 13-19

VALUE_COL = "density_ratio_msis"

# x軸のDOY -> 日付ラベルマップ
DOY_DATE_MAP = {
    252: "9/9",  253: "9/10", 254: "9/11", 255: "9/12",
    256: "9/13", 257: "9/14", 258: "9/15", 259: "9/16",
    260: "9/17", 261: "9/18", 262: "9/19", 263: "9/20",
    264: "9/21", 265: "9/22", 266: "9/23",
}


# =========================
# utility
# =========================
def add_doy(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    out = df.copy()
    out["DOY"] = (
        dt.dt.dayofyear
        + dt.dt.hour / 24.0
        + dt.dt.minute / 1440.0
        + dt.dt.second / 86400.0
    )
    return out


def grid_median(
    df: pd.DataFrame,
    doy_bins: np.ndarray,
    lat_bins: np.ndarray,
    value_col: str,
) -> np.ndarray:
    Z = np.full((len(lat_bins) - 1, len(doy_bins) - 1), np.nan)
    if len(df) == 0:
        return Z

    doy = df["DOY"].to_numpy()
    lat = df["lat"].to_numpy()
    val = df[value_col].to_numpy()

    ok = np.isfinite(doy) & np.isfinite(lat) & np.isfinite(val)
    doy, lat, val = doy[ok], lat[ok], val[ok]

    doy_i = np.digitize(doy, doy_bins) - 1
    lat_i = np.digitize(lat, lat_bins) - 1

    ok = (
        (doy_i >= 0) & (doy_i < len(doy_bins) - 1)
        & (lat_i >= 0) & (lat_i < len(lat_bins) - 1)
    )
    doy_i, lat_i, val = doy_i[ok], lat_i[ok], val[ok]

    bucket = defaultdict(list)
    for i, j, v in zip(lat_i, doy_i, val):
        bucket[(i, j)].append(float(v))

    for (i, j), arr in bucket.items():
        Z[i, j] = float(np.median(arr))

    return Z


def compute_residual_grid(Z_full: np.ndarray, doy_bins: np.ndarray) -> np.ndarray:
    """
    非SSW参照期間 (Sep9-12 & Sep20-23) の列の中央値を reference として差し引く。
    """
    doy_centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    ref_mask = (
        ((doy_centers >= DOY_REF1_MIN) & (doy_centers <= DOY_REF1_MAX)) |
        ((doy_centers >= DOY_REF2_MIN) & (doy_centers <= DOY_REF2_MAX))
    )

    ref_cols = Z_full[:, ref_mask]
    ref_profile = np.nanmedian(ref_cols, axis=1)  # (n_lat,)
    Z_residual = Z_full - ref_profile[:, np.newaxis]
    return Z_residual


def load_and_prepare(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if "datetime" not in df.columns:
        df = df.reset_index()

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        raise KeyError("緯度列が見つかりません")
    if lat_col != "lat":
        df = df.rename(columns={lat_col: "lat"})

    df = df.dropna(subset=["datetime", "lat", VALUE_COL]).copy()

    t0 = pd.Timestamp(T_START, tz="UTC")
    t1 = pd.Timestamp(T_END, tz="UTC")
    df = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].copy()
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)].copy()
    df = add_doy(df)
    return df


def make_mesh_from_bins(
    doy_bins: np.ndarray, lat_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    doy_centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    return np.meshgrid(doy_centers, lat_centers)


def plot_job(job: dict) -> None:
    label   = job["label"]
    parquet = job["parquet"]
    out_png = job["out_png"]

    print(f"\n=== {label} ===")
    df = load_and_prepare(parquet)
    print(f"  行数: {len(df):,}")

    if len(df) == 0:
        print("  ⚠ データが空です。スキップします。")
        return

    doy_bins = np.arange(DOY_START, DOY_END + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    X, Y = make_mesh_from_bins(doy_bins, lat_bins)

    # 全LTデータでグリッド
    Z_ratio    = grid_median(df, doy_bins, lat_bins, VALUE_COL)
    Z_residual = compute_residual_grid(Z_ratio, doy_bins)

    # カラースケール
    vals_ratio = Z_ratio[np.isfinite(Z_ratio)].ravel()
    vmin_r = float(np.nanpercentile(vals_ratio, 1))
    vmax_r = float(np.nanpercentile(vals_ratio, 99))

    vals_res = Z_residual[np.isfinite(Z_residual)].ravel()
    vmax_res = float(np.nanpercentile(np.abs(vals_res), 98))
    vmin_res = -vmax_res

    levels_ratio = np.linspace(vmin_r, vmax_r, N_LEVELS + 1)
    levels_res   = np.linspace(vmin_res, vmax_res, N_LEVELS + 1)

    # x軸のtick位置とラベル
    xtick_doys = list(range(DOY_START, DOY_END + 1))
    xtick_labels = [DOY_DATE_MAP.get(d, str(d)) for d in xtick_doys]

    # =========================
    # プロット（1行 × 2パネル）
    # =========================
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f"{label}  density_ratio_msis = rho_obs / rho_MSIS_real  (2019 SH SSW: Sep 9–23)\n"
        "LT / Altitude / F10.7 / Ap dependencies removed by MSIS  (all LT, all observations)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(1, 4, figure=fig,
                           width_ratios=[1, 0.05, 1, 0.05],
                           wspace=0.12)

    def add_decorations(ax: plt.Axes, title: str) -> None:
        ax.axvspan(DOY_REF1_MIN, DOY_REF1_MAX,
                   color="lightblue", alpha=0.20, lw=0, label="Non-SSW ref")
        ax.axvspan(DOY_REF2_MIN, DOY_REF2_MAX,
                   color="lightblue", alpha=0.20, lw=0)
        ax.axvline(DOY_SSW_START, color="red", lw=1.2, ls="--", alpha=0.8, label="SSW period")
        ax.axvline(DOY_SSW_END,   color="red", lw=1.2, ls="--", alpha=0.8)
        ax.set_xlim(DOY_START, DOY_END)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlabel("Date (2019)", fontsize=10)
        ax.set_xticks(xtick_doys)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="white", lw=0.5, ls="--", alpha=0.5)  # 赤道線
        ax.grid(alpha=0.2, color="white", linewidth=0.5)
        ax.set_title(title, fontsize=11)

    # --- 左パネル: 密度比そのもの ---
    ax0 = fig.add_subplot(gs[0, 0])
    cf0 = ax0.contourf(X, Y, Z_ratio, levels=levels_ratio, cmap="plasma", extend="both")
    add_decorations(ax0, "density_ratio_msis  (rho_obs / rho_MSIS)")
    ax0.set_ylabel("Geographic Latitude (°)", fontsize=10)
    ax0.legend(fontsize=8, loc="upper left", framealpha=0.85)
    cb0 = fig.colorbar(cf0, cax=fig.add_subplot(gs[0, 1]))
    cb0.set_label("ratio (obs / MSIS)", fontsize=9)

    # --- 右パネル: reference 差し引き residual ---
    ax1 = fig.add_subplot(gs[0, 2])
    cf1 = ax1.contourf(X, Y, Z_residual, levels=levels_res, cmap="RdBu_r", extend="both")
    add_decorations(ax1, "Residual  (ratio − ref period median)")
    ax1.tick_params(axis="y", labelleft=False)
    cb1 = fig.colorbar(cf1, cax=fig.add_subplot(gs[0, 3]))
    cb1.set_label("Δratio (SSW − non-SSW ref)", fontsize=9)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  ✅ 保存完了: {out_png}")
    plt.close(fig)


# =========================
# main
# =========================
DOY_START = 252
DOY_END   = 266

def main() -> None:
    for job in JOBS:
        plot_job(job)
    print("\n✅ 全プロット完了")


if __name__ == "__main__":
    main()
