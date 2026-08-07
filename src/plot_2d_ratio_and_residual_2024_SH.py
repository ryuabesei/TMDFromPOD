"""
plot_2d_ratio_and_residual_2024_SH.py

Purpose:
    Plot 2D (DOY vs Latitude) Hovmöller maps of:
      Row 0: rho_ratio  (density_ratio_msis = rho_obs / rho_MSIS_real)
      Row 1: delta_ratio (residual vs non-SSW reference)
    for SWARM-A, B, C during the 2024 Southern Hemisphere SSW period.

    SH SSW 2024 timeline:
      - Ref period 1 : 2024-06-15 to 2024-07-04  (DOY 167–186)
      - SSW window 1 : 2024-07-05 to 2024-07-09  (DOY 187–191, peak Jul 7)
      - SSW window 2 : 2024-08-03 to 2024-08-07  (DOY 216–220, peak Aug 5)
      - Ref period 2 : 2024-08-10 to 2024-08-25  (DOY 223–238)

    LT sectors per satellite:
      SWARM-A/C : Morning (04–11 LT) / Evening (16–23 LT)
      SWARM-B   : Nightside (22–05 LT, wrap) / Dayside (11–17 LT)

Output:
    Figure/2024/sh/2D_ratio_and_residual_2024_SH_SWARM-{A,B,C}.png

NOTE:
    Run AFTER normalize_swarm_2024_SH.py to ensure normalizeddata/2024_SH/ exists.
    If normalised data is not yet available, the script will exit with a clear error.
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────
YEAR = 2024

# Time window (inclusive)
T_START = "2024-06-15 00:00:00"
T_END   = "2024-08-25 23:59:59"

# 2024 is a leap year → DOY ranges
DOY_MIN, DOY_MAX = 167.0, 238.0   # Jun 15 = 167, Aug 25 = 238

# Reference & SSW windows (in DOY)
DOY_REF1 = (167, 186)                           # Jun 15 – Jul 04
DOY_SSW1 = (187, 191)                           # Jul 05 – Jul 09
DOY_SSW2 = (216, 220)                           # Aug 03 – Aug 07
DOY_REF2 = (223, 238)                           # Aug 10 – Aug 25
DOY_PEAKS = [188.5, 218.5]                      # Jul 07, Aug 05 (midpoint of window)

LAT_MIN, LAT_MAX = -60, 60
DOY_BIN  = 1.0
LAT_BIN  = 3.0
N_LEVELS = 21

VALUE_COL = "density_ratio_msis"

# Representative DOY ticks with date labels
DOY_TICKS = list(range(170, 239, 7))

def doy_to_date(doy: int, year: int = YEAR) -> str:
    """Convert integer DOY to 'MM/DD' string for tick labels."""
    try:
        d = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)
        return d.strftime("%m/%d")
    except Exception:
        return str(doy)

# ──────────────────────────────────────────────────────────────────────────────
# Satellite configurations
# ──────────────────────────────────────────────────────────────────────────────

# Will be set by resolve_parquets() at runtime
def resolve_parquets() -> list[dict]:
    """Try normalizeddata/2024_SH/ first, fall back to normalizeddata/2024/."""
    base_sh = Path("normalizeddata/2024_SH")
    base_nh = Path("normalizeddata/2024")

    configs = []
    for sat_id, prefix, sec1, sec2, sec1_wrap, sec1_title, sec2_title in [
        ("A", "swarm_dnsapod",
         (4, 11),  (16, 23), False,
         "Morning (04–11 LT)", "Evening (16–23 LT)"),
        ("B", "swarm_dnsbpod",
         (22, 5),  (11, 17), True,
         "Nightside (22–05 LT)", "Dayside (11–17 LT)"),
        ("C", "swarm_dnscpod",
         (4, 11),  (16, 23), False,
         "Morning (04–11 LT)", "Evening (16–23 LT)"),
    ]:
        # Prefer SH-specific normalised data; fall back to NH-period data
        candidates = [
            base_sh / f"{prefix}_2024_SH_normalized_with_LT.parquet",
            base_nh / f"{prefix}_2024_normalized_with_LT.parquet",
        ]
        parquet = next((p for p in candidates if p.exists()), None)
        if parquet is None:
            print(f"[WARN] No parquet found for SWARM-{sat_id}. Tried: {candidates}")
            continue
        configs.append(dict(
            label      = f"SWARM-{sat_id}",
            parquet    = parquet,
            out_png    = Path(f"Figure/2024/sh/2D_ratio_and_residual_2024_SH_SWARM-{sat_id}.png"),
            sec1       = sec1,
            sec2       = sec2,
            sec1_wrap  = sec1_wrap,
            sec1_title = sec1_title,
            sec2_title = sec2_title,
        ))
    return configs


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

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
    bucket: dict = defaultdict(list)
    for i, j, v in zip(lat_i, doy_i, val):
        bucket[(i, j)].append(float(v))
    for (i, j), arr in bucket.items():
        Z[i, j] = float(np.median(arr))
    return Z


def compute_residual_grid(Z_full: np.ndarray, doy_bins: np.ndarray) -> np.ndarray:
    """Subtract non-SSW reference profile (median over ref1 + ref2 columns)."""
    doy_centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    ref_mask = (
        ((doy_centers >= DOY_REF1[0]) & (doy_centers <= DOY_REF1[1]))
        | ((doy_centers >= DOY_REF2[0]) & (doy_centers <= DOY_REF2[1]))
    )
    ref_profile = np.nanmedian(Z_full[:, ref_mask], axis=1)   # (n_lat,)
    return Z_full - ref_profile[:, np.newaxis]


def lt_filter(df: pd.DataFrame, lt_min: float, lt_max: float, wrap: bool) -> pd.DataFrame:
    if wrap:
        return df[(df["lst_h"] >= lt_min) | (df["lst_h"] < lt_max)].copy()
    return df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)].copy()


def daily_representative_lt_line(
    df: pd.DataFrame, lt_min: float, lt_max: float, wrap: bool
) -> tuple[np.ndarray, np.ndarray]:
    g = lt_filter(df, lt_min, lt_max, wrap)
    if len(g) == 0:
        return np.array([]), np.array([])
    g = g.set_index("datetime")
    daily = g.resample("D")["lst_h"].median().dropna()
    if len(daily) == 0:
        return np.array([]), np.array([])
    x = daily.index.dayofyear.to_numpy() + 0.5
    y = daily.to_numpy()
    return x, y


def load_and_prepare(parquet_path: Path) -> pd.DataFrame:
    print(f"  Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    if "datetime" not in df.columns:
        df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        raise KeyError(f"Latitude column not found in {parquet_path}")
    if lat_col != "lat":
        df = df.rename(columns={lat_col: "lat"})

    # Resolve ratio column
    for cand in ["density_ratio_msis", "rho_ratio", "density_norm"]:
        if cand in df.columns:
            if cand != VALUE_COL:
                df[VALUE_COL] = df[cand]
            break
    else:
        raise KeyError(f"No density ratio column found in {parquet_path}. Columns: {list(df.columns)}")

    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL]).copy()
    t0 = pd.Timestamp(T_START, tz="UTC")
    t1 = pd.Timestamp(T_END,   tz="UTC")
    df = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].copy()
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)].copy()
    df = add_doy(df)
    print(f"    → {len(df):,} rows after filtering")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def decorate_ax(
    ax: plt.Axes,
    title: str,
    ylabel: bool = False,
    xlabel: bool = False,
) -> None:
    # Reference period shading
    ax.axvspan(*DOY_REF1, color="lightblue",   alpha=0.18, lw=0, label="Non-SSW ref")
    ax.axvspan(*DOY_REF2, color="lightblue",   alpha=0.18, lw=0)
    # SSW windows
    ax.axvspan(*DOY_SSW1, color="lightyellow", alpha=0.40, lw=0, label="SSW period")
    ax.axvspan(*DOY_SSW2, color="lightyellow", alpha=0.40, lw=0)
    # SSW peak lines
    for peak in DOY_PEAKS:
        ax.axvline(peak, color="red", lw=1.5, ls="--", alpha=0.75)

    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    tick_labels = [f"{d}\n({doy_to_date(d)})" for d in DOY_TICKS]
    ax.set_xticks(DOY_TICKS)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.grid(alpha=0.20, color="white", linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    if ylabel:
        ax.set_ylabel("Geographic Latitude (°)", fontsize=11, fontweight="bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    if xlabel:
        ax.set_xlabel(f"Day of Year {YEAR} (DOY 167–238)", fontsize=11, fontweight="bold")


def overlay_lt_line(
    ax: plt.Axes,
    df_sec: pd.DataFrame,
    sec: tuple[float, float],
    wrap: bool,
) -> None:
    ax_r = ax.twinx()
    lt_label = "Local Time (h)"
    if wrap:
        ax_r.set_ylabel(lt_label, fontsize=9, color="k")
        ax_r.set_ylim(0, 24)
    else:
        ax_r.set_ylabel(lt_label, fontsize=9, color="k")
        ax_r.set_ylim(sec[0], sec[1])
    x_lt, y_lt = daily_representative_lt_line(df_sec, sec[0], sec[1], wrap)
    if len(x_lt) > 0:
        ax_r.plot(x_lt, y_lt, color="black", lw=1.5, ls="-")


def plot_satellite(sat: dict) -> None:
    label      = sat["label"]
    parquet    = sat["parquet"]
    out_png    = sat["out_png"]
    sec1       = sat["sec1"]
    sec2       = sat["sec2"]
    sec1_wrap  = sat["sec1_wrap"]
    sec1_title = sat["sec1_title"]
    sec2_title = sat["sec2_title"]

    print(f"\n{'='*55}")
    print(f"  {label}  —  2024 SH SSW")
    print(f"{'='*55}")

    df = load_and_prepare(parquet)

    doy_bins = np.arange(DOY_MIN, DOY_MAX + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    X = np.meshgrid(
        0.5 * (doy_bins[:-1] + doy_bins[1:]),
        0.5 * (lat_bins[:-1] + lat_bins[1:])
    )
    Xg, Yg = X  # noqa

    df_s1 = lt_filter(df, sec1[0], sec1[1], sec1_wrap)
    df_s2 = lt_filter(df, sec2[0], sec2[1], False)

    print(f"  Sec1 ({sec1_title}): {len(df_s1):,} pts")
    print(f"  Sec2 ({sec2_title}): {len(df_s2):,} pts")

    Z_ratio_1 = grid_median(df_s1, doy_bins, lat_bins, VALUE_COL)
    Z_ratio_2 = grid_median(df_s2, doy_bins, lat_bins, VALUE_COL)
    Z_resid_1 = compute_residual_grid(Z_ratio_1, doy_bins)
    Z_resid_2 = compute_residual_grid(Z_ratio_2, doy_bins)

    # Symmetric color limits for ratio
    all_ratio = np.concatenate([
        Z_ratio_1[np.isfinite(Z_ratio_1)],
        Z_ratio_2[np.isfinite(Z_ratio_2)],
    ])
    vmin_r = float(np.nanpercentile(all_ratio, 1))
    vmax_r = float(np.nanpercentile(all_ratio, 99))
    levels_ratio = np.linspace(vmin_r, vmax_r, N_LEVELS + 1)

    all_res = np.concatenate([
        Z_resid_1[np.isfinite(Z_resid_1)],
        Z_resid_2[np.isfinite(Z_resid_2)],
    ])
    vmax_d = float(np.nanpercentile(np.abs(all_res), 98))
    levels_resid = np.linspace(-vmax_d, vmax_d, N_LEVELS + 1)

    # ── Figure layout: 2 rows × 3 cols (col2 = colorbar) ──────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1, 1, 0.04],
        height_ratios=[1, 1],
        wspace=0.08, hspace=0.22,
    )

    # Row 0 — rho_ratio
    ax00 = fig.add_subplot(gs[0, 0])
    cf00 = ax00.contourf(Xg, Yg, Z_ratio_1, levels=levels_ratio, cmap="plasma", extend="both")
    decorate_ax(ax00, f"ρ_ratio  —  {sec1_title}", ylabel=True)
    overlay_lt_line(ax00, df_s1, sec1, sec1_wrap)
    ax00.legend(loc="upper left", fontsize=8, framealpha=0.75)

    ax01 = fig.add_subplot(gs[0, 1])
    cf01 = ax01.contourf(Xg, Yg, Z_ratio_2, levels=levels_ratio, cmap="plasma", extend="both")
    decorate_ax(ax01, f"ρ_ratio  —  {sec2_title}")
    overlay_lt_line(ax01, df_s2, sec2, False)

    cb0 = fig.add_subplot(gs[0, 2])
    fig.colorbar(cf01, cax=cb0).set_label(
        "ρ_ratio  (obs / MSIS)", fontsize=10, fontweight="bold"
    )

    # Row 1 — delta_ratio
    ax10 = fig.add_subplot(gs[1, 0])
    cf10 = ax10.contourf(Xg, Yg, Z_resid_1, levels=levels_resid, cmap="RdBu_r", extend="both")
    decorate_ax(ax10, f"Δρ_ratio  —  {sec1_title}", ylabel=True, xlabel=True)
    overlay_lt_line(ax10, df_s1, sec1, sec1_wrap)

    ax11 = fig.add_subplot(gs[1, 1])
    cf11 = ax11.contourf(Xg, Yg, Z_resid_2, levels=levels_resid, cmap="RdBu_r", extend="both")
    decorate_ax(ax11, f"Δρ_ratio  —  {sec2_title}", xlabel=True)
    overlay_lt_line(ax11, df_s2, sec2, False)

    cb1 = fig.add_subplot(gs[1, 2])
    fig.colorbar(cf11, cax=cb1).set_label(
        "Δρ_ratio  (ratio − non-SSW ref)", fontsize=10, fontweight="bold"
    )

    fig.suptitle(
        f"{label}  —  2024 SH SSW  (DOY 167–238)\n"
        "Top: ρ_ratio (obs/MSIS)  |  Bottom: Δρ_ratio (Residual vs non-SSW ref)\n"
        "SSW peaks: Jul 7 & Aug 5  |  ERA5 10 hPa T rise ~15 K (Jul), ~17 K (Aug)",
        fontsize=13, fontweight="bold", y=1.00,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  ✅ Saved: {out_png}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== 2D Hovmöller plot: 2024 SH SSW ===\n")
    satellites = resolve_parquets()
    if not satellites:
        print("[ERROR] No satellite parquet files found. "
              "Run the download + normalization scripts first.")
        return
    for sat in satellites:
        plot_satellite(sat)
    print("\n✅  All 2D plots for 2024 SH SSW completed.")
    print("   Output: Figure/2024/sh/")


if __name__ == "__main__":
    main()
