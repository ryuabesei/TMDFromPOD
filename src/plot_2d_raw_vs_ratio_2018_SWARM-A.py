"""
plot_2d_raw_vs_ratio_2018_SWARM-A.py

Purpose:
    2-row x 2-column 2D map (Latitude vs DOY) for SWARM-A, 2018 NH SSW (DOY 30-65):
      Top    : Raw thermospheric mass density  [kg m^-3]  (cmap="turbo")
      Bottom : rho_ratio  (rho_obs / rho_MSIS_real)       (cmap="turbo")
      Left   : Morning sector  (04-11 LT)
      Right  : Evening sector  (16-23 LT)

Output:
    Figure/2018/2D_raw_vs_ratio_2018_SWARM-A_DOY30-65.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path

# ============================================================
# Paths
# ============================================================
RAW_PARQUET  = Path("integrateddata/2018/swarm_dnsapod_2018_DOY20-80.parquet")
NORM_PARQUET = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
OUT_PNG      = Path("Figure/2018/2D_raw_vs_ratio_2018_SWARM-A_DOY30-65.png")

# ============================================================
# Settings
# ============================================================
T_START = "2018-01-30 00:00:00"
T_END   = "2018-03-06 23:59:59"

LAT_MIN, LAT_MAX = -60, 60
DOY_MIN, DOY_MAX = 30.0, 65.0
DOY_BIN = 0.5
LAT_BIN = 3.0
N_LEVELS = 20

# LT sectors
SEC_MORNING = (4,  11)   # 04-11 LT
SEC_EVENING = (16, 23)   # 16-23 LT

# Non-SSW reference periods
DOY_REF1      = (30, 40)
DOY_REF2      = (61, 65)
DOY_SSW_START = 41
DOY_SSW_END   = 60
DOY_SSW_PEAK  = 43   # Feb 12, 2018

RAW_COL   = "density"
RATIO_COL = "density_ratio_msis"


# ============================================================
# Utilities
# ============================================================
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

    bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
    for i, j, v in zip(lat_i, doy_i, val):
        bucket[(i, j)].append(float(v))
    for (i, j), arr in bucket.items():
        Z[i, j] = float(np.median(arr))

    return Z


def daily_lt_line(
    df: pd.DataFrame,
    lt_min: float,
    lt_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Daily median LT within the specified sector."""
    if "lst_h" not in df.columns or len(df) == 0:
        return np.array([]), np.array([])
    g = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)].copy()
    if len(g) == 0:
        return np.array([]), np.array([])
    g = g.set_index("datetime")
    daily = g.resample("D")["lst_h"].median().dropna()
    if len(daily) == 0:
        return np.array([]), np.array([])
    x = daily.index.dayofyear.to_numpy() + 0.5
    y = daily.to_numpy()
    return x, y


def row_vmin_vmax(*grids: np.ndarray, pct_lo: int = 1, pct_hi: int = 99) -> tuple[float, float]:
    vals = np.concatenate([g[np.isfinite(g)].ravel() for g in grids if np.isfinite(g).any()])
    vmin = float(np.nanpercentile(vals, pct_lo))
    vmax = float(np.nanpercentile(vals, pct_hi))
    if vmin == vmax:
        eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps; vmax += eps
    return vmin, vmax


def load_df(path: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "datetime":
            df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    # normalise lat column name
    for cname in ["lat", "latitude", "geod_lat"]:
        if cname in df.columns and cname != "lat":
            df = df.rename(columns={cname: "lat"})
            break

    required = ["datetime", "lat", "lst_h", value_col]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    t0 = pd.Timestamp(T_START, tz="UTC")
    t1 = pd.Timestamp(T_END,   tz="UTC")
    df = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)]
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)]
    df = add_doy(df)
    df = df[(df["DOY"] >= DOY_MIN) & (df["DOY"] <= DOY_MAX)]
    return df


def split_lt(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into Morning (04-11 LT) and Evening (16-23 LT) sectors."""
    df_m = df[(df["lst_h"] >= SEC_MORNING[0]) & (df["lst_h"] < SEC_MORNING[1])].copy()
    df_e = df[(df["lst_h"] >= SEC_EVENING[0]) & (df["lst_h"] < SEC_EVENING[1])].copy()
    return df_m, df_e


def decorate_ax(
    ax: plt.Axes,
    title: str,
    show_ylabel: bool = False,
    show_xlabel: bool = False,
) -> None:
    ax.axvspan(*DOY_REF1, color="lightblue", alpha=0.20, lw=0, label=f"Non-SSW ref (DOY {DOY_REF1[0]}-{DOY_REF1[1]})")
    ax.axvspan(*DOY_REF2, color="lightblue", alpha=0.20, lw=0, label=f"Non-SSW ref (DOY {DOY_REF2[0]}-{DOY_REF2[1]})")
    ax.axvspan(DOY_SSW_START, DOY_SSW_END, color="lightyellow", alpha=0.30, lw=0, label=f"SSW period (DOY {DOY_SSW_START}-{DOY_SSW_END})")
    ax.axvline(DOY_SSW_START, color="red", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(DOY_SSW_END,   color="red", lw=1.2, ls="--", alpha=0.7)
    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xticks(range(int(DOY_MIN), int(DOY_MAX) + 1, 5))
    ax.grid(alpha=0.2, color="white", linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    if show_ylabel:
        ax.set_ylabel("Geographic Latitude (°)", fontsize=11, fontweight="bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    if show_xlabel:
        ax.set_xlabel(f"Day of Year 2018 (DOY {int(DOY_MIN)}–{int(DOY_MAX)})", fontsize=11, fontweight="bold")


def overlay_lt(ax: plt.Axes, df_sec: pd.DataFrame, lt_min: float, lt_max: float) -> None:
    """Overlay representative LT line on a twin right axis."""
    x_lt, y_lt = daily_lt_line(df_sec, lt_min, lt_max)
    ax_r = ax.twinx()
    ax_r.set_ylim(lt_min, lt_max)
    ax_r.set_ylabel("LT (h)", fontsize=10)
    ax_r.set_yticks(range(int(lt_min), int(lt_max) + 1))
    if len(x_lt) > 0:
        ax_r.plot(x_lt, y_lt, color="black", lw=1.5, ls="-")


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("Loading raw density data ...")
    df_raw = load_df(RAW_PARQUET, RAW_COL)
    print(f"  raw  : {len(df_raw):,} rows")

    print("Loading normalized (ratio) data ...")
    df_norm = load_df(NORM_PARQUET, RATIO_COL)
    print(f"  norm : {len(df_norm):,} rows")

    # Split by LT sector
    df_raw_m,  df_raw_e  = split_lt(df_raw)
    df_norm_m, df_norm_e = split_lt(df_norm)
    print(f"  RAW  morning={len(df_raw_m):,}  evening={len(df_raw_e):,}")
    print(f"  NORM morning={len(df_norm_m):,}  evening={len(df_norm_e):,}")

    doy_bins = np.arange(DOY_MIN, DOY_MAX + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

    # Grids
    Z_raw_m   = grid_median(df_raw_m,  doy_bins, lat_bins, RAW_COL)
    Z_raw_e   = grid_median(df_raw_e,  doy_bins, lat_bins, RAW_COL)
    Z_ratio_m = grid_median(df_norm_m, doy_bins, lat_bins, RATIO_COL)
    Z_ratio_e = grid_median(df_norm_e, doy_bins, lat_bins, RATIO_COL)

    # Colour limits (shared within each row)
    vmin_raw,   vmax_raw   = row_vmin_vmax(Z_raw_m,   Z_raw_e)
    vmin_ratio, vmax_ratio = row_vmin_vmax(Z_ratio_m, Z_ratio_e)
    levels_raw   = np.linspace(vmin_raw,   vmax_raw,   N_LEVELS + 1)
    levels_ratio = np.linspace(vmin_ratio, vmax_ratio, N_LEVELS + 1)

    # Mesh centres
    doy_c = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    lat_c = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    X, Y  = np.meshgrid(doy_c, lat_c)

    # ── Figure: 2 rows x (2 cols + colorbar) ─────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "SWARM-A  2018 NH SSW  (DOY 30–65)\n"
        "Top: Raw Thermospheric Mass Density  |  Bottom: $\\rho_{\\rm ratio}$  ($\\rho_{\\rm obs}$ / $\\rho_{\\rm MSIS,real}$)\n"
        f"Non-SSW ref: DOY {DOY_REF1[0]}–{DOY_REF1[1]} & DOY {DOY_REF2[0]}–{DOY_REF2[1]}   "
        f"SSW period: DOY {DOY_SSW_START}–{DOY_SSW_END}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1, 1, 0.04],
        height_ratios=[1, 1],
        wspace=0.08,
        hspace=0.18,
    )

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    cax0 = fig.add_subplot(gs[0, 2])
    cax1 = fig.add_subplot(gs[1, 2])

    # ── Row 0: Raw density ───────────────────────────────────
    cf00 = ax00.contourf(X, Y, Z_raw_m, levels=levels_raw, cmap="turbo", extend="both")
    decorate_ax(ax00, "Raw Density — Morning (04–11 LT)", show_ylabel=True)
    ax00.legend(loc="upper left", fontsize=7, framealpha=0.7)
    overlay_lt(ax00, df_raw_m, *SEC_MORNING)

    cf01 = ax01.contourf(X, Y, Z_raw_e, levels=levels_raw, cmap="turbo", extend="both")
    decorate_ax(ax01, "Raw Density — Evening (16–23 LT)")
    overlay_lt(ax01, df_raw_e, *SEC_EVENING)

    cb0 = fig.colorbar(cf01, cax=cax0)
    cb0.set_label("Density  [kg m$^{-3}$]", fontsize=10, fontweight="bold")

    # ── Row 1: rho_ratio ────────────────────────────────────
    cf10 = ax10.contourf(X, Y, Z_ratio_m, levels=levels_ratio, cmap="turbo", extend="both")
    decorate_ax(ax10, "$\\rho_{\\rm ratio}$ — Morning (04–11 LT)", show_ylabel=True, show_xlabel=True)
    overlay_lt(ax10, df_norm_m, *SEC_MORNING)

    cf11 = ax11.contourf(X, Y, Z_ratio_e, levels=levels_ratio, cmap="turbo", extend="both")
    decorate_ax(ax11, "$\\rho_{\\rm ratio}$ — Evening (16–23 LT)", show_xlabel=True)
    overlay_lt(ax11, df_norm_e, *SEC_EVENING)

    cb1 = fig.colorbar(cf11, cax=cax1)
    cb1.set_label(
        "$\\rho_{\\rm ratio}$  ($\\rho_{\\rm obs}$ / $\\rho_{\\rm MSIS}$)",
        fontsize=10, fontweight="bold",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
