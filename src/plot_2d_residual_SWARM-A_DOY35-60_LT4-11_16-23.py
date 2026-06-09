"""
plot_2d_residual_SWARM-A_DOY35-60_LT4-11_16-23.py

目的:
    1. DOY 35-40 の正規化密度を各緯度ビンで平均 → non-SSW reference profile(lat)
    2. DOY 35-60 の 2D グリッドから reference を差し引いて residual を計算
    3. SWARM-A のLTセクター（04–11 LT / 16–23 LT）で contourf 表示
    4. SSW期間（DOY 41-60）をハイライト

出力:
    Figure/2D_residual_SWARM-A_DOY35-60_LT4-11_16-23.png
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
NORM_PARQUET = Path("normalizeddata/swarm_dnsapod_2018_normalized_DOY20-80.parquet")
OUT_PNG      = Path("Figure/2D_residual_SWARM-A_DOY35-60_LT4-11_16-23.png")

T_START = "2018-02-04 00:00:00"   # DOY 35
T_END   = "2018-03-01 23:59:59"   # DOY 60

SECTOR_MORNING = (4, 11)
SECTOR_EVENING = (16, 23)

LAT_MIN, LAT_MAX = -60, 60

DOY_BIN  = 1.0
LAT_BIN  = 3.0
N_LEVELS = 21

# Non-SSW reference period (DOY 35-40 only)
DOY_REF_MIN, DOY_REF_MAX = 35, 40

# SSW period
DOY_SSW_START, DOY_SSW_END = 41, 60

NORM_COL = "density_norm"


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


def compute_residual_grid(
    Z_full: np.ndarray,
    doy_bins: np.ndarray,
) -> np.ndarray:
    """
    DOY 35-40 の列を平均して reference profile(lat) を作り、
    Z_full から差し引く。
    """
    doy_centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    ref_mask = (doy_centers >= DOY_REF_MIN) & (doy_centers <= DOY_REF_MAX)

    ref_cols = Z_full[:, ref_mask]
    ref_profile = np.nanmean(ref_cols, axis=1)  # (n_lat,)

    Z_residual = Z_full - ref_profile[:, np.newaxis]
    return Z_residual


def daily_representative_lt_line(
    df: pd.DataFrame,
    lt_min: float,
    lt_max: float,
    stat: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    g = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)].copy()
    if len(g) == 0:
        return np.array([]), np.array([])
    g = g.set_index("datetime")
    daily = g.resample("D")["lst_h"].median().dropna() if stat == "median" else g.resample("D")["lst_h"].mean().dropna()
    if len(daily) == 0:
        return np.array([]), np.array([])
    x = daily.index.dayofyear.to_numpy() + 0.5
    y = daily.to_numpy()
    return x, y


def load_and_prepare(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if "datetime" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "datetime":
            df = df.rename(columns={df.columns[0]: "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", NORM_COL]).copy()

    keep_cols = [c for c in ["datetime", "lat", "lon", "lst_h", NORM_COL] if c in df.columns]
    df = df[keep_cols].copy()
    df = df.rename(columns={NORM_COL: "density_value"})

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


# =========================
# main
# =========================
def main() -> None:
    print("Loading normalized data (SWARM-A, DOY 35-60) ...")
    df = load_and_prepare(NORM_PARQUET)
    print(f"  rows: {len(df):,}")

    if len(df) == 0:
        raise ValueError("No data in the period.")

    doy_bins = np.arange(35, 61 + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

    # LT セクター分割
    df_m = df[(df["lst_h"] >= SECTOR_MORNING[0]) & (df["lst_h"] < SECTOR_MORNING[1])].copy()
    df_e = df[(df["lst_h"] >= SECTOR_EVENING[0]) & (df["lst_h"] < SECTOR_EVENING[1])].copy()
    print(f"  morning: {len(df_m):,}, evening: {len(df_e):,}")

    # 2D グリッド（DOY 35-60）
    Z_norm_m = grid_median(df_m, doy_bins, lat_bins, "density_value")
    Z_norm_e = grid_median(df_e, doy_bins, lat_bins, "density_value")

    # Residual グリッド
    Z_res_m = compute_residual_grid(Z_norm_m, doy_bins)
    Z_res_e = compute_residual_grid(Z_norm_e, doy_bins)

    # 対称カラースケール
    all_res = np.concatenate([
        Z_res_m[np.isfinite(Z_res_m)].ravel(),
        Z_res_e[np.isfinite(Z_res_e)].ravel(),
    ])
    vmax = float(np.nanpercentile(np.abs(all_res), 98))
    vmin = -vmax
    levels = np.linspace(vmin, vmax, N_LEVELS + 1)

    X, Y = make_mesh_from_bins(doy_bins, lat_bins)

    # =========================
    # プロット（1行 × 2列）
    # =========================
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        "Swarm-A  Residual Normalized Density (DOY 35–60, 2018)\n"
        "Reference: DOY 35–40 mean per latitude (non-SSW)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.08)

    panels = [
        (0, Z_res_m, df_m, SECTOR_MORNING, "Residual (04–11 LT)"),
        (1, Z_res_e, df_e, SECTOR_EVENING, "Residual (16–23 LT)"),
    ]

    cf_last = None
    for col, Z_res, df_sec, sector, title in panels:
        ax = fig.add_subplot(gs[0, col])

        cf = ax.contourf(
            X, Y, Z_res,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
        )
        cf_last = cf

        # SSW期間をハイライト
        ax.axvspan(DOY_SSW_START, DOY_SSW_END,
                   color="yellow", alpha=0.12, lw=0, label="SSW period")

        # Reference 帯をハイライト
        ax.axvspan(DOY_REF_MIN, DOY_REF_MAX,
                   color="lightblue", alpha=0.20, lw=0, label="Non-SSW reference")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day of Year (2018)", fontsize=10)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlim(35, 60)
        ax.set_xticks(range(35, 61, 5))
        ax.grid(alpha=0.2, color="white", linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Geographic Latitude (°)", fontsize=10)
            ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
        else:
            ax.tick_params(axis="y", labelleft=False)

        # LT 線
        lt_min, lt_max = sector
        ax_r = ax.twinx()
        ax_r.set_ylim(lt_min, lt_max)
        ax_r.set_ylabel("LT (h)", fontsize=9)
        if sector == SECTOR_MORNING:
            ax_r.set_yticks([4, 5, 6, 7, 8, 9, 10, 11])
        else:
            ax_r.set_yticks([16, 17, 18, 19, 20, 21, 22, 23])

        x_lt, y_lt = daily_representative_lt_line(df_sec, lt_min, lt_max, stat="median")
        if len(x_lt) > 0:
            ax_r.plot(x_lt, y_lt, color="k", lw=1.2)

    # カラーバー
    cb_ax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(cf_last, cax=cb_ax)
    cbar.set_label(
        "Residual density [kg m$^{-3}$]\n(obs − ref)",
        fontsize=10,
    )

    plt.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n✅ 保存完了: {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
