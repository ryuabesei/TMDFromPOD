"""
plot_2d_residual_SWARM-B.py

目的:
    1. DOY 37~42 の正規化密度を各緯度ビンで平均 → non-SSW reference profile(lat)
    2. DOY 36~52 の 2D グリッドから reference を差し引いて residual を計算
    3. SWARM-B のLTセクター（01–04 LT / 13–15 LT）で contourf 表示

出力:
    Figure/2018/2D_residual_SWARM-B.png
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
NORM_PARQUET = Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized.parquet")
OUT_PNG      = Path("Figure/2018/2D_residual_SWARM-B.png")

T_START = "2018-02-05 00:00:00"
T_END   = "2018-02-21 23:59:59"

SECTOR_NIGHT = (1, 4)    # 01–04 LT
SECTOR_DAY   = (13, 15)  # 13–15 LT

LAT_MIN, LAT_MAX = -60, 60

DOY_BIN  = 0.5
LAT_BIN  = 3.0
N_LEVELS = 21

DOY_REF_MIN = 37
DOY_REF_MAX = 42

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
    doy_centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    ref_mask = (doy_centers >= DOY_REF_MIN) & (doy_centers <= DOY_REF_MAX)
    ref_cols = Z_full[:, ref_mask]
    ref_profile = np.nanmean(ref_cols, axis=1)
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
    print("Loading normalized data (SWARM-B) ...")
    df = load_and_prepare(NORM_PARQUET)
    print(f"  rows: {len(df):,}")

    if len(df) == 0:
        raise ValueError("No data in the period.")

    doy_bins = np.arange(
        np.floor(df["DOY"].min()),
        np.ceil(df["DOY"].max()) + DOY_BIN,
        DOY_BIN,
    )
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

    df_n = df[(df["lst_h"] >= SECTOR_NIGHT[0]) & (df["lst_h"] < SECTOR_NIGHT[1])].copy()
    df_d = df[(df["lst_h"] >= SECTOR_DAY[0])   & (df["lst_h"] < SECTOR_DAY[1])].copy()
    print(f"  night (01–04 LT): {len(df_n):,}, day (13–15 LT): {len(df_d):,}")

    Z_norm_n = grid_median(df_n, doy_bins, lat_bins, "density_value")
    Z_norm_d = grid_median(df_d, doy_bins, lat_bins, "density_value")

    Z_res_n = compute_residual_grid(Z_norm_n, doy_bins)
    Z_res_d = compute_residual_grid(Z_norm_d, doy_bins)

    all_res = np.concatenate([
        Z_res_n[np.isfinite(Z_res_n)].ravel(),
        Z_res_d[np.isfinite(Z_res_d)].ravel(),
    ])
    vmax = float(np.nanpercentile(np.abs(all_res), 98))
    vmin = -vmax
    levels = np.linspace(vmin, vmax, N_LEVELS + 1)

    X, Y = make_mesh_from_bins(doy_bins, lat_bins)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        "Swarm-B  Residual Normalized Density (DOY 36–52, 2018)\n"
        f"Reference: DOY {DOY_REF_MIN}–{DOY_REF_MAX} mean per latitude",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.08)

    panels = [
        (0, Z_res_n, df_n, SECTOR_NIGHT, "Residual (01–04 LT)"),
        (1, Z_res_d, df_d, SECTOR_DAY,   "Residual (13–15 LT)"),
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

        ax.axvspan(DOY_REF_MIN, DOY_REF_MAX, color="white", alpha=0.15, lw=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day of Year (2018)", fontsize=10)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlim(doy_bins[0], doy_bins[-2] + DOY_BIN)
        ax.grid(alpha=0.2, color="white", linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Geographic Latitude (°)", fontsize=10)
        else:
            ax.tick_params(axis="y", labelleft=False)

        lt_min, lt_max = sector
        ax_r = ax.twinx()
        ax_r.set_ylim(lt_min, lt_max)
        ax_r.set_ylabel("LT (h)", fontsize=9)
        ax_r.set_yticks(range(int(lt_min), int(lt_max) + 1))

        x_lt, y_lt = daily_representative_lt_line(df_sec, lt_min, lt_max, stat="median")
        if len(x_lt) > 0:
            ax_r.plot(x_lt, y_lt, color="k", lw=1.2)

    cb_ax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(cf_last, cax=cb_ax)
    cbar.set_label("Residual density [kg m$^{-3}$]\n(obs − ref)", fontsize=10)

    plt.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n✅ 保存完了: {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
