"""
plot_2d_ratio_and_residual_2021.py

Purpose:
    Plot 2D (Date vs Latitude) maps of:
      Top row:    rho_ratio (density_ratio_msis = rho_obs / rho_MSIS_real)
      Bottom row: delta_ratio (residual ratio vs non-SSW reference)
    for SWARM-C and GRACE-FO during the 2021 NH SSW (2020-12-25 to 2021-02-05).

    LT sectors:
        SWARM-C:  Fixed  — Dawn (LT 4.5-10.5h) / Dusk (LT 16.5-22.5h)
        GRACE-FO: Dynamic orbital planes (Plane A / Plane B), circular-distance tracking.

    Grid resolution: 1 day × 3° latitude

Output:
    Figure/2021/2D_ratio_and_residual_2021_SWARM-C.png
    Figure/2021/2D_ratio_and_residual_2021_GRACE-FO.png
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# ============================================================
# Settings
# ============================================================
DATE_START = pd.Timestamp("2020-12-25", tz="UTC")
DATE_END   = pd.Timestamp("2021-02-05", tz="UTC")

DATE_REF1_START = pd.Timestamp("2020-12-25", tz="UTC")
DATE_REF1_END   = pd.Timestamp("2020-12-29", tz="UTC")
DATE_REF2_START = pd.Timestamp("2021-02-01", tz="UTC")
DATE_REF2_END   = pd.Timestamp("2021-02-05", tz="UTC")

DATE_SSW_START = pd.Timestamp("2020-12-30", tz="UTC")
DATE_SSW_END   = pd.Timestamp("2021-01-31", tz="UTC")
DATE_SSW_ONSET = pd.Timestamp("2021-01-05", tz="UTC")
DATE_SSW_PEAK  = pd.Timestamp("2021-01-04", tz="UTC")

LAT_MIN, LAT_MAX = -60, 60
DAY_BIN = 1.0       # bin width in fractional days from DATE_START
LAT_BIN = 3.0       # bin width in degrees
N_LEVELS = 21

VALUE_COL = "density_ratio_msis"

# ── day offset helpers ──────────────────────────────────────
T0_num = mdates.date2num(DATE_START.to_pydatetime())
T1_num = mdates.date2num(DATE_END.to_pydatetime())

def ts_to_daynum(ts: pd.Timestamp) -> float:
    return mdates.date2num(ts.to_pydatetime())

REF1_START_N = ts_to_daynum(DATE_REF1_START)
REF1_END_N   = ts_to_daynum(DATE_REF1_END)
REF2_START_N = ts_to_daynum(DATE_REF2_START)
REF2_END_N   = ts_to_daynum(DATE_REF2_END)
SSW_START_N  = ts_to_daynum(DATE_SSW_START)
SSW_END_N    = ts_to_daynum(DATE_SSW_END)
ONSET_N      = ts_to_daynum(DATE_SSW_ONSET)
PEAK_N       = ts_to_daynum(DATE_SSW_PEAK)


# ============================================================
# Orbital-plane helpers (GRACE-FO)
# ============================================================
def circ_dist(a: float, b: float, period: float = 24.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def assign_orbital_plane(df: pd.DataFrame, lat_col: str) -> pd.DataFrame:
    df = df.copy()
    df["orbital_plane"] = "A"
    for date, grp in df.groupby("date"):
        p25 = grp["lst_h"].quantile(0.25)
        p75 = grp["lst_h"].quantile(0.75)
        midpoint = (p25 + p75) / 2.0
        if abs(p75 - p25) < 3.0:
            midpoint = 12.0
        df.loc[grp.index[grp["lst_h"] >= midpoint], "orbital_plane"] = "B"

    dates_sorted = sorted(df["date"].unique())
    init_a = df[df["orbital_plane"] == "A"].groupby("date")["lst_h"].median()
    init_b = df[df["orbital_plane"] == "B"].groupby("date")["lst_h"].median()
    corrected_a_prev = init_a.get(dates_sorted[0], np.nan)

    for i in range(1, len(dates_sorted)):
        curr_date = dates_sorted[i]
        curr_a0 = init_a.get(curr_date, np.nan)
        curr_b0 = init_b.get(curr_date, np.nan)
        if np.isnan(corrected_a_prev) or np.isnan(curr_a0) or np.isnan(curr_b0):
            corrected_a_prev = curr_a0
            continue
        if circ_dist(corrected_a_prev, curr_b0) < circ_dist(corrected_a_prev, curr_a0):
            mask = df["date"] == curr_date
            df.loc[mask & (df["orbital_plane"] == "A"), "orbital_plane"] = "_tmp"
            df.loc[mask & (df["orbital_plane"] == "B"), "orbital_plane"] = "A"
            df.loc[mask & (df["orbital_plane"] == "_tmp"), "orbital_plane"] = "B"
            corrected_a_prev = curr_b0
        else:
            corrected_a_prev = curr_a0
    return df


# ============================================================
# Grid utilities
# ============================================================
def add_daynum(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["daynum"] = mdates.date2num(df["datetime"].dt.to_pydatetime())
    return out


def grid_median(df: pd.DataFrame, day_bins: np.ndarray, lat_bins: np.ndarray) -> np.ndarray:
    Z = np.full((len(lat_bins) - 1, len(day_bins) - 1), np.nan)
    if len(df) == 0:
        return Z

    day = df["daynum"].to_numpy()
    lat = df["lat"].to_numpy()
    val = df[VALUE_COL].to_numpy()

    ok = np.isfinite(day) & np.isfinite(lat) & np.isfinite(val)
    day, lat, val = day[ok], lat[ok], val[ok]

    day_i = np.digitize(day, day_bins) - 1
    lat_i = np.digitize(lat, lat_bins) - 1

    ok = (
        (day_i >= 0) & (day_i < len(day_bins) - 1)
        & (lat_i >= 0) & (lat_i < len(lat_bins) - 1)
    )
    day_i, lat_i, val = day_i[ok], lat_i[ok], val[ok]

    bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
    for i, j, v in zip(lat_i, day_i, val):
        bucket[(i, j)].append(float(v))
    for (i, j), arr in bucket.items():
        Z[i, j] = float(np.median(arr))
    return Z


def compute_residual_grid(Z_full: np.ndarray, day_bins: np.ndarray) -> np.ndarray:
    """Per-row (latitude) subtraction: subtract median of non-SSW reference columns."""
    day_centers = 0.5 * (day_bins[:-1] + day_bins[1:])
    ref_mask = (
        ((day_centers >= REF1_START_N) & (day_centers <= REF1_END_N)) |
        ((day_centers >= REF2_START_N) & (day_centers <= REF2_END_N))
    )
    ref_cols = Z_full[:, ref_mask]
    ref_profile = np.nanmedian(ref_cols, axis=1)
    return Z_full - ref_profile[:, np.newaxis]


def daily_median_lt(df_sec: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_daynum, y_lt_median) arrays for drawing an LT overlay line.
    x is the matplotlib date number for each day's noon, y is the daily median LST."""
    if len(df_sec) == 0:
        return np.array([]), np.array([])
    tmp = df_sec.copy()
    tmp = tmp.set_index("datetime")
    daily = tmp.resample("D")["lst_h"].median().dropna()
    if len(daily) == 0:
        return np.array([]), np.array([])
    x = np.array([mdates.date2num(d.to_pydatetime()) + 0.5 for d in daily.index])
    y = daily.to_numpy()
    return x, y


def overlay_lt_line(
    ax: plt.Axes,
    df_sec: pd.DataFrame,
    lt_range: tuple[float, float],
    ylabel: bool = True,
) -> None:
    """Draw a daily-median LT line on a twin y-axis (right side), black style like 2019 figure."""
    ax_r = ax.twinx()
    ax_r.set_ylim(lt_range[0], lt_range[1])
    n_ticks = 4 if (lt_range[1] - lt_range[0]) > 6 else 3
    ticks = np.linspace(lt_range[0], lt_range[1], n_ticks)
    ax_r.set_yticks(ticks)
    ax_r.set_yticklabels([f"{t:.0f}" for t in ticks], fontsize=9)
    ax_r.tick_params(axis="y", colors="black", right=True, labelright=True)
    ax_r.spines["right"].set_color("black")
    ax_r.spines["right"].set_linewidth(0.8)
    ax_r.set_ylabel("Local Time (h)", fontsize=10, color="black", labelpad=6)

    x_lt, y_lt = daily_median_lt(df_sec)
    if len(x_lt) > 0:
        # Avoid connecting lines across the 0h/24h wrap-around jump.
        # Find where the absolute difference between consecutive days is > 12 hours.
        diffs = np.abs(np.diff(y_lt))
        jump_indices = np.where(diffs > 12.0)[0]
        
        if len(jump_indices) == 0:
            ax_r.plot(x_lt, y_lt, color="black", lw=2.0, ls="-", zorder=10)
        else:
            start_idx = 0
            for idx in jump_indices:
                ax_r.plot(x_lt[start_idx:idx+1], y_lt[start_idx:idx+1], color="black", lw=2.0, ls="-", zorder=10)
                start_idx = idx + 1
            ax_r.plot(x_lt[start_idx:], y_lt[start_idx:], color="black", lw=2.0, ls="-", zorder=10)


# ============================================================
# Load data
# ============================================================
def load_data(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col != "lat":
        df = df.rename(columns={lat_col: "lat"})
    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL])
    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] <= DATE_END + pd.Timedelta(hours=23, minutes=59))]
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)]
    df["date"] = df["datetime"].dt.normalize()
    df = add_daynum(df)
    return df


# ============================================================
# Axis decoration
# ============================================================
def decorate_ax(ax: plt.Axes, title: str, ylabel: bool = False) -> None:
    ax.axvspan(REF1_START_N, REF1_END_N, color="lightblue", alpha=0.18, lw=0, label="Non-SSW ref")
    ax.axvspan(REF2_START_N, REF2_END_N, color="lightblue", alpha=0.18, lw=0)
    ax.axvline(ONSET_N, color="blue",  lw=1.3, ls="--", alpha=0.8, label="Onset (01/05)")
    ax.axvline(PEAK_N,  color="green", lw=1.3, ls="--", alpha=0.8, label="T10hPa peak (01/04)")
    ax.set_xlim(T0_num, T1_num)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.grid(alpha=0.2, color="white", linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel("Geographic Latitude (°)", fontsize=11, fontweight="bold")
    else:
        ax.tick_params(axis="y", labelleft=False)


# ============================================================
# Plot a single satellite
# ============================================================
def plot_satellite(
    label: str,
    df: pd.DataFrame,
    sec_defs: list[dict],
    out_png: Path,
) -> None:
    print(f"\n=== {label} ===  ({len(df):,} rows)")

    day_bins = np.arange(T0_num, T1_num + DAY_BIN, DAY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    X_centers = 0.5 * (day_bins[:-1] + day_bins[1:])
    Y_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    X, Y = np.meshgrid(X_centers, Y_centers)

    # compute grids
    grids = []
    for sec in sec_defs:
        df_sec = df[df["_sector"] == sec["key"]]
        Zr = grid_median(df_sec, day_bins, lat_bins)
        Zd = compute_residual_grid(Zr, day_bins)
        grids.append({"ratio": Zr, "delta": Zd, "df": df_sec, **sec})
        print(f"  {sec['title']}: {len(df_sec):,} obs, "
              f"ratio range [{np.nanmin(Zr):.3f}, {np.nanmax(Zr):.3f}]")

    # shared color scales
    all_ratio = np.concatenate([g["ratio"][np.isfinite(g["ratio"])] for g in grids])
    vmin_r = float(np.nanpercentile(all_ratio, 1))
    vmax_r = float(np.nanpercentile(all_ratio, 99))
    levels_ratio = np.linspace(vmin_r, vmax_r, N_LEVELS + 1)

    all_delta = np.concatenate([g["delta"][np.isfinite(g["delta"])] for g in grids])
    vmax_d = float(np.nanpercentile(np.abs(all_delta), 98))
    levels_delta = np.linspace(-vmax_d, vmax_d, N_LEVELS + 1)

    # layout: 2 rows × (n_sec + 1 colorbar) columns
    n = len(grids)
    fig = plt.figure(figsize=(8 * n + 1, 10))
    gs = gridspec.GridSpec(2, n + 1, figure=fig,
                           width_ratios=[1] * n + [0.04],
                           height_ratios=[1, 1],
                           wspace=0.08, hspace=0.20)

    for col, g in enumerate(grids):
        lt_range = g.get("lt_range", (0.0, 24.0))
        is_last_col = (col == n - 1)

        # Row 0: rho_ratio
        ax0 = fig.add_subplot(gs[0, col])
        cf0 = ax0.contourf(X, Y, g["ratio"], levels=levels_ratio, cmap="plasma", extend="both")
        decorate_ax(ax0, f"rho_ratio — {g['title']}", ylabel=(col == 0))
        overlay_lt_line(ax0, g["df"], lt_range, ylabel=is_last_col)
        if col == 0:
            ax0.legend(loc="upper left", fontsize=8, framealpha=0.7)

        # Row 1: delta_ratio
        ax1 = fig.add_subplot(gs[1, col])
        cf1 = ax1.contourf(X, Y, g["delta"], levels=levels_delta, cmap="RdBu_r", extend="both")
        decorate_ax(ax1, f"delta_ratio — {g['title']}", ylabel=(col == 0))
        overlay_lt_line(ax1, g["df"], lt_range, ylabel=is_last_col)
        ax1.set_xlabel("Date (2020/2021)", fontsize=10, fontweight="bold")

        # LT subtitle (per-panel)
        lt_info = g.get("lt_info", "")
        if lt_info:
            ax0.set_title(f"rho_ratio — {g['title']}\n{lt_info}", fontsize=11, fontweight="bold", pad=6)
            ax1.set_title(f"delta_ratio — {g['title']}\n{lt_info}", fontsize=11, fontweight="bold", pad=6)

    # Colorbars
    cb_ax0 = fig.add_subplot(gs[0, n])
    fig.colorbar(cf0, cax=cb_ax0).set_label("rho_ratio\n(obs / MSIS)", fontsize=10, fontweight="bold")

    cb_ax1 = fig.add_subplot(gs[1, n])
    fig.colorbar(cf1, cax=cb_ax1).set_label("delta_ratio\n(ratio − non-SSW ref)", fontsize=10, fontweight="bold")

    # Legend strip at bottom
    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, fc="lightblue", alpha=0.3, label="Non-SSW ref"),
        plt.Line2D([0], [0], color="blue",  lw=1.3, ls="--", label="Onset (01/05)"),
        plt.Line2D([0], [0], color="green", lw=1.3, ls="--", label="T10hPa peak (01/04)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=10,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        f"{label}  density_ratio_msis (2021 NH SSW)\n"
        "Top: rho_ratio (rho_obs/rho_MSIS)  |  Bottom: delta_ratio (Residual vs non-SSW ref)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    # ── SWARM-C ─────────────────────────────────────────────
    print("\nLoading SWARM-C ...")
    df_swc = load_data(Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet"))
    df_swc["_sector"] = "none"
    df_swc.loc[(df_swc["lst_h"] >= 4.5)  & (df_swc["lst_h"] < 10.5), "_sector"] = "dawn"
    df_swc.loc[(df_swc["lst_h"] >= 16.5) & (df_swc["lst_h"] < 22.5), "_sector"] = "dusk"

    swc_secs = [
        dict(key="dawn", title="Dawn (LT 4.5–10.5h)",  lt_range=(4.5,  10.5)),
        dict(key="dusk", title="Dusk (LT 16.5–22.5h)", lt_range=(16.5, 22.5)),
    ]
    plot_satellite("SWARM-C", df_swc, swc_secs,
                   Path("Figure/2021/2D_ratio_and_residual_2021_SWARM-C.png"))

    # ── GRACE-FO ─────────────────────────────────────────────
    print("\nLoading GRACE-FO ...")
    df_gfo = load_data(Path("normalizeddata/2021/grace_fo_dns_2021_normalized_with_LT_removed.parquet"))

    lat_col = "lat"
    print("  Assigning orbital planes ...")
    df_gfo = assign_orbital_plane(df_gfo, lat_col)
    df_gfo["_sector"] = df_gfo["orbital_plane"]   # "A" or "B"

    # Build per-plane LT drift info for annotation
    lt_a = df_gfo[df_gfo["orbital_plane"] == "A"].groupby("date")["lst_h"].median()
    lt_b = df_gfo[df_gfo["orbital_plane"] == "B"].groupby("date")["lst_h"].median()
    info_a = f"LT {lt_a.iloc[0]:.1f}h → {lt_a.iloc[-1]:.1f}h"
    info_b = f"LT {lt_b.iloc[0]:.1f}h → {lt_b.iloc[-1]:.1f}h"

    gfo_secs = [
        dict(key="A", title="Orbital Plane A", lt_info=info_a, lt_range=(0.0, 24.0)),
        dict(key="B", title="Orbital Plane B", lt_info=info_b, lt_range=(8.0, 15.0)),
    ]
    plot_satellite("GRACE-FO", df_gfo, gfo_secs,
                   Path("Figure/2021/2D_ratio_and_residual_2021_GRACE-FO.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
