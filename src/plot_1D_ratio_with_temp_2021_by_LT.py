"""
plot_1D_ratio_with_temp_2021_by_LT.py

Purpose:
    Plot daily-median rho_ratio (left axis) and ERA5 10 hPa temperature (right axis)
    split by LT sector for SWARM-C and GRACE-FO during the 2021 NH SSW (2020-12-25 to 2021-02-05).

    LT sectors are defined dynamically per satellite to avoid data gaps due to precession:
        SWARM-C:  Fixed windows — Dawn (LT 4.5–10.5 h) / Dusk (LT 16.5–22.5 h)
        GRACE-FO: DYNAMIC — two orbital planes always ~12 h apart, both drifting together:
                    Plane A (lower-LT side):  ~1 h (Dec 28) → ~10 h (Feb 5)  [Night → Morning]
                    Plane B (higher-LT side): ~13 h (Dec 28) → ~22 h (Feb 5)  [Afternoon → Night]
                  Each day, observations are split at the daily LT mid-gap (the valley between
                  the two orbit-plane clusters), so that Plane A and Plane B are tracked
                  consistently throughout the drift.

    Layout per satellite: 3 rows (latitude bands) x 2 columns (LT sectors)

Output:
    Figure/2021/1D_ratio_with_temp_2021_SWARM-C_by_LT.png
    Figure/2021/1D_ratio_with_temp_2021_GRACE-FO_by_LT.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

LT_SWARM = [
    dict(label="Dawn (LT 4.5–10.5 h)",  lt_min=4.5,  lt_max=10.5,  color="#1a6faf"),
    dict(label="Dusk (LT 16.5–22.5 h)", lt_min=16.5, lt_max=22.5, color="#e07b39"),
]

# GRACE-FO: dynamic mode (lt_min/lt_max are ignored at plot time; use dynamic=True)
LT_GRACE = [
    dict(label="Orbital Plane A (lower LT)",  plane="A", color="#1a6faf", dynamic=True),
    dict(label="Orbital Plane B (higher LT)", plane="B", color="#e07b39", dynamic=True),
]

SATELLITES = [
    dict(
        label      = "SWARM-C",
        parquet    = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet"),
        out_png    = Path("Figure/2021/1D_ratio_with_temp_2021_SWARM-C_by_LT.png"),
        lt_sectors = LT_SWARM,
    ),
    dict(
        label      = "GRACE-FO",
        parquet    = Path("normalizeddata/2021/grace_fo_dns_2021_normalized_with_LT_removed.parquet"),
        out_png    = Path("Figure/2021/1D_ratio_with_temp_2021_GRACE-FO_by_LT.png"),
        lt_sectors = LT_GRACE,
    ),
]

DATE_START = pd.Timestamp("2020-12-25", tz="UTC")
DATE_END   = pd.Timestamp("2021-02-05", tz="UTC")

LAT_BANDS = [
    ("High  (40-60°)", 40.0, 60.0),
    ("Mid   (20-40°)", 20.0, 40.0),
    ("Low   ( 0-20°)",  0.0, 20.0),
]

DATE_REF1_START = pd.Timestamp("2020-12-25", tz="UTC")
DATE_REF1_END   = pd.Timestamp("2020-12-29", tz="UTC")
DATE_REF2_START = pd.Timestamp("2021-02-01", tz="UTC")
DATE_REF2_END   = pd.Timestamp("2021-02-05", tz="UTC")

DATE_SSW_START = pd.Timestamp("2020-12-30", tz="UTC")
DATE_SSW_END   = pd.Timestamp("2021-01-31", tz="UTC")
DATE_SSW_ONSET = pd.Timestamp("2021-01-05", tz="UTC")
DATE_SSW_PEAK  = pd.Timestamp("2021-01-04", tz="UTC")

VALUE_COL = "density_ratio_msis"

def get_ref_median(daily: pd.Series) -> float:
    idx = daily.index
    mask = (
        ((idx >= DATE_REF1_START) & (idx <= DATE_REF1_END)) |
        ((idx >= DATE_REF2_START) & (idx <= DATE_REF2_END))
    )
    if mask.sum() == 0 or daily[mask].isna().all():
        return np.nan
    return float(daily[mask].median())


def assign_orbital_plane(df: pd.DataFrame, lat_col: str) -> pd.DataFrame:
    """Dynamically classify each observation into Orbital Plane A (lower LT) or B (higher LT).

    GRACE-FO always has two orbit planes ~12 h apart in LT. Both drift together at ~0.08 h/day.
    Each day we find the mid-gap between the two clusters (the valley in the bimodal LT histogram)
    and split observations there.

    Step 1 – Initial split (per day):
        midpoint = (25th-pct + 75th-pct) / 2
        Plane A: lst_h < midpoint  (lower LT side)
        Plane B: lst_h >= midpoint (higher LT side)

    Step 2 – Continuity fix (0/24 h wraparound):
        When Plane A crosses midnight (e.g. 0.044 h → 23.96 h the next day),
        the initial split assigns the newly-lower cluster (~12 h) as A and the
        wrapped cluster (~24 h) as B — physically swapping the labels.
        We detect this by monitoring the day-to-day jump in Plane-A median LT.
        A jump > 6 h triggers a label swap for that day onward, restoring
        consistent tracking of the same physical orbit throughout the period.
    """
    df = df.copy()
    df["orbital_plane"] = "A"   # default

    for date, grp in df.groupby("date"):
        p25 = grp["lst_h"].quantile(0.25)
        p75 = grp["lst_h"].quantile(0.75)
        midpoint = (p25 + p75) / 2.0
        # On transition days the spread may be huge; use 12 as the canonical split
        # if the two clusters are within 3 h of each other (degenerate)
        if abs(p75 - p25) < 3.0:
            midpoint = 12.0
        plane_b_mask = grp["lst_h"] >= midpoint
        df.loc[grp.index[plane_b_mask], "orbital_plane"] = "B"

    # ----------------------------------------------------------------
    # Continuity fix: track physical orbital planes across 0/24 h boundary.
    #
    # Each day has two LT clusters ~12 h apart. The algorithm above labels
    # the lower-LT cluster "A" and the higher-LT cluster "B". When Plane A
    # drifts past midnight (0h), the clusters wrap: Plane A reappears at ~24h
    # which is labeled "B" by the above step, causing a physical swap.
    #
    # Fix: for each day, compute the circular distance from the previous
    # day's corrected Plane-A median to each of the two current clusters.
    # Assign whichever cluster is closer (circularly) to be the new Plane A.
    # This guarantees continuous tracking of the same physical orbit.
    # ----------------------------------------------------------------
    def circ_dist(a: float, b: float, period: float = 24.0) -> float:
        d = abs(a - b) % period
        return min(d, period - d)

    dates_sorted = sorted(df["date"].unique())

    # Compute initial per-day cluster medians (before any swap correction)
    init_a = df[df["orbital_plane"] == "A"].groupby("date")["lst_h"].median()
    init_b = df[df["orbital_plane"] == "B"].groupby("date")["lst_h"].median()

    # Start tracking from day 0 (no correction needed on first day)
    corrected_a_prev = init_a.get(dates_sorted[0], np.nan)

    for i in range(1, len(dates_sorted)):
        curr_date = dates_sorted[i]
        curr_a0 = init_a.get(curr_date, np.nan)
        curr_b0 = init_b.get(curr_date, np.nan)

        if np.isnan(corrected_a_prev) or np.isnan(curr_a0) or np.isnan(curr_b0):
            corrected_a_prev = curr_a0
            continue

        dist_to_a = circ_dist(corrected_a_prev, curr_a0)
        dist_to_b = circ_dist(corrected_a_prev, curr_b0)

        if dist_to_b < dist_to_a:
            # Cluster B is actually the continuation of Plane A — swap labels
            mask = df["date"] == curr_date
            df.loc[mask & (df["orbital_plane"] == "A"), "orbital_plane"] = "_tmp"
            df.loc[mask & (df["orbital_plane"] == "B"), "orbital_plane"] = "A"
            df.loc[mask & (df["orbital_plane"] == "_tmp"), "orbital_plane"] = "B"
            corrected_a_prev = curr_b0
            print(f"    [wrap-fix] Plane-A → cluster B on {curr_date.date()}: "
                  f"prev_A={corrected_a_prev:.3f}h  clust_A={curr_a0:.3f}h  clust_B={curr_b0:.3f}h  "
                  f"(dist_A={dist_to_a:.2f} dist_B={dist_to_b:.2f}) → SWAPPED")
        else:
            corrected_a_prev = curr_a0

    # Diagnostics: report daily medians after continuity fix
    daily_lt_a_final = df[df["orbital_plane"] == "A"].groupby("date")["lst_h"].median().rename("LT_A_med")
    daily_lt_b_final = df[df["orbital_plane"] == "B"].groupby("date")["lst_h"].median().rename("LT_B_med")
    diag = pd.concat([daily_lt_a_final, daily_lt_b_final], axis=1)
    print("  Daily LT after continuity fix (Plane-A | Plane-B):")

    print(diag.to_string())
    return df


def load_era5_temp() -> pd.Series:
    era5_dir = Path("data/SSW2021/ERA5")
    files = sorted(list(era5_dir.glob("*.nc")))
    if len(files) == 0:
        raise FileNotFoundError(f"No NetCDF files found in {era5_dir}")
    
    datasets = [xr.open_dataset(fp) for fp in files]
    ds_all = xr.concat(datasets, dim="valid_time")
    ds_sub = ds_all.sel(latitude=slice(90.0, 60.0))
    
    weights = np.cos(np.deg2rad(ds_sub["latitude"]))
    weights.name = "weights"
    
    weighted_temp = ds_sub["t"].weighted(weights)
    temp_1d = weighted_temp.mean(dim=["latitude", "longitude"]).squeeze()
    
    df_temp = pd.Series(temp_1d.values, index=pd.to_datetime(temp_1d["valid_time"].values))
    df_daily = df_temp.resample("D").mean()
    df_daily.index = df_daily.index.tz_localize("UTC")
    df_daily = df_daily.loc[DATE_START:DATE_END]
    
    for ds in datasets:
        ds.close()
    return df_daily

def plot_satellite(sat: dict, df_temp: pd.Series) -> None:
    label      = sat["label"]
    parquet    = sat["parquet"]
    out_png    = sat["out_png"]
    lt_sectors = sat["lt_sectors"]
    use_dynamic = lt_sectors[0].get("dynamic", False)

    print(f"\n=== {label} ===")
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    df = df.dropna(subset=["datetime", lat_col, "lst_h", VALUE_COL])
    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] <= DATE_END + pd.Timedelta(hours=23, minutes=59))]
    df["date"] = df["datetime"].dt.normalize()
    print(f"  {len(df):,} rows loaded after filtering")

    # GRACE-FO: assign dynamic orbital planes before plotting
    if use_dynamic:
        print("  [GRACE-FO] Assigning orbital planes dynamically ...")
        df = assign_orbital_plane(df, lat_col)

    n_bands = len(LAT_BANDS)
    n_lt    = len(lt_sectors)
    x_min = DATE_START - pd.Timedelta(hours=12)
    x_max = DATE_END + pd.Timedelta(hours=12)

    fig, axes = plt.subplots(n_bands, n_lt, figsize=(6.8 * n_lt, 3.3 * n_bands), sharex=True, sharey="row")
    fig.subplots_adjust(hspace=0.08, wspace=0.18)

    for col_idx, lt in enumerate(lt_sectors):
        lt_label = lt["label"]
        color    = lt["color"]

        if use_dynamic:
            plane_key = lt["plane"]
            df_lt = df[df["orbital_plane"] == plane_key]
            # Build an informative title showing LT range at start & end of period
            lt_start = df_lt[df_lt["date"] == df_lt["date"].min()]["lst_h"].median()
            lt_end   = df_lt[df_lt["date"] == df_lt["date"].max()]["lst_h"].median()
            lt_label = f"{lt['label']}\n(LT {lt_start:.1f}h→{lt_end:.1f}h)"
        else:
            lt_min = lt["lt_min"]
            lt_max = lt["lt_max"]
            df_lt = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]
        print(f"  {lt_label.splitlines()[0]}: {len(df_lt):,} obs")

        for bi, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
            ax = axes[bi, col_idx]
            mask = (df_lt[lat_col].abs() >= lat_lo) & (df_lt[lat_col].abs() < lat_hi)
            sub = df_lt[mask]
            daily = sub.groupby("date")[VALUE_COL].median()
            ref = get_ref_median(daily)

            # 背景ハイライト
            ax.axvspan(DATE_REF1_START, DATE_REF1_END, color="lightblue", alpha=0.20)
            ax.axvspan(DATE_REF2_START, DATE_REF2_END, color="lightblue", alpha=0.20)
            ax.axvspan(DATE_SSW_START, DATE_SSW_END, color="lightyellow", alpha=0.35)
            
            # Onset / Peak 線
            ax.axvline(DATE_SSW_ONSET, color="blue", linewidth=1.2, linestyle="--", zorder=5)
            ax.axvline(DATE_SSW_PEAK, color="green", linewidth=1.2, linestyle="--", zorder=5)
            ax.axhline(ref, color="gray", linewidth=0.8, linestyle="--", zorder=2)

            # 左軸: rho_ratio
            ax.plot(daily.index, daily.values, color=color, linewidth=2.0, marker="o", markersize=4, zorder=4,
                    label="rho_ratio")
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(axis="y", alpha=0.3, linewidth=0.7)
            
            if col_idx == 0:
                ax.set_ylabel("rho_ratio\n(obs / MSIS)", fontsize=9, color=color)
            ax.tick_params(axis="y", labelcolor=color)

            # 右軸: ERA5 Temperature (各サブプロットに重ねる)
            ax2 = ax.twinx()
            ax2.plot(df_temp.index, df_temp.values, color="#d62728", linewidth=1.2, linestyle="-", marker="x", markersize=2.5, alpha=0.75, zorder=3,
                     label="ERA5 Temp")
            if col_idx == n_lt - 1:
                ax2.set_ylabel("Temperature [K]", fontsize=9, color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")

            # テキスト注記
            ax.text(0.01, 0.97, band_label.strip(), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="left")
            if not np.isnan(ref):
                ax.text(0.99, 0.97, f"ref = {ref:.3f}", transform=ax.transAxes, fontsize=7,
                        va="top", ha="right", color="gray")

            if bi == 0:
                ax.set_title(lt_label, fontsize=11, fontweight="bold", pad=6, color=color)
            if bi == n_bands - 1:
                ax.set_xlabel("Date (2020/2021)", fontsize=10)

    # 共通凡例の配置
    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, fc="lightblue",  alpha=0.3, label="Non-SSW ref"),
        plt.Rectangle((0, 0), 1, 1, fc="lightyellow", alpha=0.5, label="SSW period"),
        plt.Line2D([0], [0], color="blue", lw=1.2, ls="--", label="Onset (01/05)"),
        plt.Line2D([0], [0], color="green", lw=1.2, ls="--", label="T10hPa Peak (01/04)"),
        plt.Line2D([0], [0], color="#d62728", lw=1.2, marker="x", label="ERA5 Temp (10 hPa, 60N-90N)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"{label}  density_ratio_msis & Stratospheric Temp (2021 NH SSW) by LT\n(Left Axis: rho_ratio | Right Axis: ERA5 T10hPa 60°N–90°N)",
                 fontsize=12, fontweight="bold", y=1.01)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)

def main() -> None:
    print("Loading ERA5 Temp...")
    df_temp = load_era5_temp()
    for sat in SATELLITES:
        plot_satellite(sat, df_temp)
    print("\nDone.")

if __name__ == "__main__":
    main()
