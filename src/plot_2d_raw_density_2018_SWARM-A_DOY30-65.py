"""
plot_2d_raw_density_2018_SWARM-A_DOY30-65.py

目的:
    raw（正規化前: integrateddata/2018/swarm_dnsapod_2018_DOY20-80.parquet）の
    2018年SSW時（DOY 30–65）における SWARM-A 質量密度の2D分布（緯度 × DOY）を
    Morning (04–11 LT) と Evening (16–23 LT) の2セクターでプロットする。

添付画像スタイルの特徴:
    - 20レベルのturboカラーマップ (cmap="turbo")
    - 左軸: Geographic Latitude (°) [-60, 60]
    - 右軸: LT (h) （黒実線で各日の代表LTを表示）
    - 横軸: Day of Year 2018 (DOY 30–65)
    - 右側に共通カラーバー ("Density [kg m$^{-3}$]")
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 非対話型バックエンドを設定して表示ブロックを回避

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path


# =========================
# 設定
# =========================
RAW_PARQUET = Path("integrateddata/2018/swarm_dnsapod_2018_DOY20-80.parquet")
OUT_PNG = Path("Figure/2018/swarm_dnsapod_2018_raw_density_DOY30-65_LT4-11_16-23.png")

T_START = "2018-01-30 00:00:00"
T_END = "2018-03-06 23:59:59"

SECTOR_MORNING = (4, 11)      # 04–11 LT
SECTOR_EVENING = (16, 23)    # 16–23 LT

LAT_MIN, LAT_MAX = -60, 60

DOY_MIN, DOY_MAX = 30.0, 65.0
DOY_BIN = 0.5
LAT_BIN = 3.0
N_LEVELS = 20

RAW_COL = "density"


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


def ensure_required_columns(df: pd.DataFrame, required: list[str], parquet_path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{parquet_path} に必要列がありません: {missing}")


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

    if stat == "median":
        daily = g.resample("D")["lst_h"].median().dropna()
    elif stat == "mean":
        daily = g.resample("D")["lst_h"].mean().dropna()
    else:
        raise ValueError("stat は 'median' または 'mean' を指定してください。")

    if len(daily) == 0:
        return np.array([]), np.array([])

    x = daily.index.dayofyear.to_numpy() + 0.5
    y = daily.to_numpy()
    return x, y


def load_and_prepare(parquet_path: Path, density_col: str) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if "datetime" not in df.columns:
        df = df.reset_index()
        if "datetime" in df.columns:
            pass
        elif "index" in df.columns:
            df = df.rename(columns={"index": "datetime"})
        else:
            df = df.rename(columns={df.columns[0]: "datetime"})

    ensure_required_columns(df, ["datetime", "lat", "lst_h", density_col], parquet_path)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", density_col]).copy()

    keep_cols = ["datetime", "lat", "lon", "lst_h", density_col]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df = df.rename(columns={density_col: "density_value"})

    t0 = pd.Timestamp(T_START, tz="UTC")
    t1 = pd.Timestamp(T_END, tz="UTC")
    df = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].copy()

    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)].copy()

    df = add_doy(df)
    df = df[(df["DOY"] >= DOY_MIN) & (df["DOY"] <= DOY_MAX)].copy()
    return df


def split_sectors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_m = df[(df["lst_h"] >= SECTOR_MORNING[0]) & (df["lst_h"] < SECTOR_MORNING[1])].copy()
    df_e = df[(df["lst_h"] >= SECTOR_EVENING[0]) & (df["lst_h"] < SECTOR_EVENING[1])].copy()
    return df_m, df_e


def row_vmin_vmax(*grids: np.ndarray) -> tuple[float, float]:
    arrs = [g[np.isfinite(g)].ravel() for g in grids if np.isfinite(g).any()]
    if not arrs:
        raise ValueError("有効なグリッド値がありません。")
    vals = np.concatenate(arrs)
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    if vmin == vmax:
        eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps
    return vmin, vmax


def make_levels(vmin: float, vmax: float, n_levels: int) -> np.ndarray:
    return np.linspace(vmin, vmax, n_levels + 1)


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
    print("Loading raw data for DOY 30-65 ...")
    df_raw = load_and_prepare(RAW_PARQUET, RAW_COL)
    print(f"  raw : {len(df_raw):,} rows")

    if len(df_raw) == 0:
        raise ValueError("対象期間のデータが空です。")

    doy_bins = np.arange(DOY_MIN, DOY_MAX + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

    df_raw_m, df_raw_e = split_sectors(df_raw)
    print(f"  RAW morning={len(df_raw_m):,}, evening={len(df_raw_e):,}")

    Z_m = grid_median(df_raw_m, doy_bins, lat_bins, "density_value")
    Z_e = grid_median(df_raw_e, doy_bins, lat_bins, "density_value")

    vmin_raw, vmax_raw = row_vmin_vmax(Z_m, Z_e)
    levels_raw = make_levels(vmin_raw, vmax_raw, N_LEVELS)

    X, Y = make_mesh_from_bins(doy_bins, lat_bins)

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(
        "Swarm-A Raw Thermospheric Mass Density (2018, DOY 30–65)",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[1, 1, 0.04],
        wspace=0.08,
    )

    panels = [
        (0, Z_m, "Before norm. (04–11 LT)", SECTOR_MORNING, df_raw_m),
        (1, Z_e, "Before norm. (16–23 LT)", SECTOR_EVENING, df_raw_e),
    ]

    cf_last = None
    for col, Z_data, title, sector, df_sec in panels:
        ax = fig.add_subplot(gs[0, col])

        cf = ax.contourf(
            X, Y, Z_data,
            levels=levels_raw,
            cmap="turbo",
            extend="both",
        )
        cf_last = cf

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Day of Year 2018 (DOY 30–65)", fontsize=11)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlim(DOY_MIN, DOY_MAX)
        ax.grid(alpha=0.2, color="white", linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Geographic Latitude (°)", fontsize=11)
        else:
            ax.tick_params(axis="y", labelleft=False)

        # 代表 LT 線
        lt_min, lt_max = sector
        ax_r = ax.twinx()
        ax_r.set_ylim(lt_min, lt_max)
        ax_r.set_ylabel("LT (h)", fontsize=10)

        if sector == SECTOR_MORNING:
            ax_r.set_yticks([4, 5, 6, 7, 8, 9, 10, 11])
        else:
            ax_r.set_yticks([16, 17, 18, 19, 20, 21, 22, 23])

        x_lt, y_lt = daily_representative_lt_line(
            df_sec,
            lt_min=lt_min,
            lt_max=lt_max,
            stat="median",
        )
        if len(x_lt) > 0:
            ax_r.plot(x_lt, y_lt, color="k", lw=1.2)

        # SSWピーク（DOY 36-45付近）のハイライト
        ax.axvspan(36, 45, color="white", alpha=0.10, lw=0)

    cb_ax = fig.add_subplot(gs[0, 2])
    fig.colorbar(cf_last, cax=cb_ax).set_label(
        "Density [kg m$^{-3}$]", fontsize=11
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n✅ 保存完了: {OUT_PNG}")


if __name__ == "__main__":
    main()
