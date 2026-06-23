"""
plot_density_before_after_normalization.py

目的:
    正規化前（integrateddata/2018/swarm_dnsapod_2018.parquet）と
    正規化後（normalizeddata/2018/swarm_dnsapod_2018_normalized.parquet）の
    2D密度分布（緯度 × DoY）を左右に並べて比較する。

参考図に合わせた点:
    - 右軸に LT(h) を置く
    - 黒実線で「各日の衛星トラックの代表LT」を表示
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
RAW_PARQUET = Path("integrateddata/2018/swarm_dnsbpod_2018.parquet")
NORM_PARQUET = Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized(450km).parquet")
OUT_PNG = Path("figures/swarm_dnsbpod_2018_before_after_normalization(450km).png")

T_START = "2018-02-05 00:00:00"
T_END = "2018-02-20 23:59:59"

SECTOR_MORNING = (1, 3)      # 07–09 LT
SECTOR_EVENING = (13, 15)    # 18–21 LT

LAT_MIN, LAT_MAX = -60, 60

DOY_BIN = 0.5
LAT_BIN = 3.0
N_LEVELS = 20

RAW_COL = "density"
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
    """
    参考文献の 'local time of the satellite tracks at corresponding days'
    に合わせ、各日の代表LTを返す。

    stat:
        "median" 推奨
        "mean"   でも可
    """
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
    print("Loading raw data ...")
    df_raw = load_and_prepare(RAW_PARQUET, RAW_COL)
    print(f"  raw  : {len(df_raw):,} rows")

    print("Loading normalized data ...")
    df_norm = load_and_prepare(NORM_PARQUET, NORM_COL)
    print(f"  norm : {len(df_norm):,} rows")

    if len(df_raw) == 0 or len(df_norm) == 0:
        raise ValueError("対象期間のデータが空です。")

    doy_all = np.concatenate([df_raw["DOY"].to_numpy(), df_norm["DOY"].to_numpy()])
    doy_bins = np.arange(np.floor(np.nanmin(doy_all)), np.ceil(np.nanmax(doy_all)) + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

    df_raw_m, df_raw_e = split_sectors(df_raw)
    df_norm_m, df_norm_e = split_sectors(df_norm)

    print(f"  RAW  morning={len(df_raw_m):,}, evening={len(df_raw_e):,}")
    print(f"  NORM morning={len(df_norm_m):,}, evening={len(df_norm_e):,}")

    Z = {
        "raw_m": grid_median(df_raw_m, doy_bins, lat_bins, "density_value"),
        "raw_e": grid_median(df_raw_e, doy_bins, lat_bins, "density_value"),
        "norm_m": grid_median(df_norm_m, doy_bins, lat_bins, "density_value"),
        "norm_e": grid_median(df_norm_e, doy_bins, lat_bins, "density_value"),
    }

    vmin_raw, vmax_raw = row_vmin_vmax(Z["raw_m"], Z["raw_e"])
    vmin_norm, vmax_norm = row_vmin_vmax(Z["norm_m"], Z["norm_e"])

    levels_raw = make_levels(vmin_raw, vmax_raw, N_LEVELS)
    levels_norm = make_levels(vmin_norm, vmax_norm, N_LEVELS)

    X, Y = make_mesh_from_bins(doy_bins, lat_bins)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Swarm-B Thermospheric Mass Density (2018)\n"
        "Before Normalization (top) vs After Normalization (bottom)",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1, 1, 0.05],
        hspace=0.30,
        wspace=0.08,
    )

    panels = [
        (0, 0, "raw_m",  levels_raw,  "Before norm. (01–03 LT)", SECTOR_MORNING),
        (0, 1, "raw_e",  levels_raw,  "Before norm. (13–15 LT)", SECTOR_EVENING),
        (1, 0, "norm_m", levels_norm, "After norm. (01–03 LT)",  SECTOR_MORNING),
        (1, 1, "norm_e", levels_norm, "After norm. (13–15 LT)",  SECTOR_EVENING),
    ]

    df_lookup = {
        "raw_m": df_raw_m,
        "raw_e": df_raw_e,
        "norm_m": df_norm_m,
        "norm_e": df_norm_e,
    }

    cf_lookup = {}

    for row, col, key, levels, title, sector in panels:
        ax = fig.add_subplot(gs[row, col])

        cf = ax.contourf(
            X, Y, Z[key],
            levels=levels,
            cmap="turbo",
            extend="both",
        )
        cf_lookup[(row, col)] = cf

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Day of Year (2018)", fontsize=10)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlim(doy_bins[0], doy_bins[-2] + DOY_BIN)
        ax.grid(alpha=0.2, color="white", linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Geographic Latitude (°)", fontsize=10)
        else:
            ax.tick_params(axis="y", labelleft=False)

        # 参考文献に合わせた LT 線
        lt_min, lt_max = sector
        ax_r = ax.twinx()
        ax_r.set_ylim(lt_min, lt_max)
        ax_r.set_ylabel("LT (h)", fontsize=9)

        if sector == SECTOR_MORNING:
            ax_r.set_yticks([1, 2, 3])
        else:
            ax_r.set_yticks([13, 14, 15])

        x_lt, y_lt = daily_representative_lt_line(
            df_lookup[key],
            lt_min=lt_min,
            lt_max=lt_max,
            stat="median",   # 参考図寄りなら median 推奨
        )
        if len(x_lt) > 0:
            ax_r.plot(x_lt, y_lt, color="k", lw=1.0)

        ax.axvspan(36, 45, color="white", alpha=0.10, lw=0)

    cb_ax_raw = fig.add_subplot(gs[0, 2])
    cb_ax_norm = fig.add_subplot(gs[1, 2])

    fig.colorbar(cf_lookup[(0, 1)], cax=cb_ax_raw).set_label(
        "Density [kg m$^{-3}$]", fontsize=10
    )
    fig.colorbar(cf_lookup[(1, 1)], cax=cb_ax_norm).set_label(
        "Normalized density [kg m$^{-3}$]\n(ref: 450 km, F10.7=70, Ap=4)",
        fontsize=10
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n✅ 保存完了: {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()