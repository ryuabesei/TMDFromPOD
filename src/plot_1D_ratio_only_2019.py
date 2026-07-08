"""
plot_1D_ratio_only_2019.py

Purpose:
    Plot daily-median density_ratio_msis (= rho_obs / rho_MSIS_real) as a 1D
    time series for the 2019 SH SSW period (2019-08-20 to 2019-09-23) [all LT, no delta].
    Results are shown for three latitude bands:
        High:   |lat| 40-60 deg
        Mid:    |lat| 20-40 deg
        Low:    |lat|  0-20 deg
    
    ERA5 T10hPa peak date (09/19) is highlighted.

Output:
    Figure/2019/1D_ratio_only_msis_2019_SWARM-{A,B,C}.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

JOBS = [
    dict(
        label   = "SWARM-A",
        parquet = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        out_png = Path("Figure/2019/1D_ratio_only_msis_2019_SWARM-A.png"),
        color   = ("#1f77b4", "#ff7f0e", "#2ca02c"),
    ),
    dict(
        label   = "SWARM-B",
        parquet = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        out_png = Path("Figure/2019/1D_ratio_only_msis_2019_SWARM-B.png"),
        color   = ("#1f77b4", "#ff7f0e", "#2ca02c"),
    ),
    dict(
        label   = "SWARM-C",
        parquet = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        out_png = Path("Figure/2019/1D_ratio_only_msis_2019_SWARM-C.png"),
        color   = ("#1f77b4", "#ff7f0e", "#2ca02c"),
    ),
]

DATE_START = pd.Timestamp("2019-08-20", tz="UTC")
DATE_END   = pd.Timestamp("2019-09-23", tz="UTC")

LAT_BANDS = [
    ("High  (40-60 deg)",  40.0, 60.0),
    ("Mid   (20-40 deg)",  20.0, 40.0),
    ("Low   ( 0-20 deg)",   0.0, 20.0),
]

DATE_REF1_START = pd.Timestamp("2019-08-20", tz="UTC")
DATE_REF1_END   = pd.Timestamp("2019-08-26", tz="UTC")
DATE_REF2_START = pd.Timestamp("2019-09-20", tz="UTC")
DATE_REF2_END   = pd.Timestamp("2019-09-23", tz="UTC")

DATE_SSW_START = pd.Timestamp("2019-08-27", tz="UTC")
DATE_SSW_END   = pd.Timestamp("2019-09-19", tz="UTC")
DATE_SSW_PEAK  = pd.Timestamp("2019-09-19", tz="UTC")

VALUE_COL = "density_ratio_msis"

def load_swarm_daily(parquet: Path) -> dict[str, pd.Series]:
    print(f"  Loading {parquet.name} ...")
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        raise KeyError("Latitude column not found")
    if lat_col != "lat":
        df = df.rename(columns={lat_col: "lat"})
    df = df.dropna(subset=["datetime", "lat", VALUE_COL])
    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] <= DATE_END)]
    df["date"] = df["datetime"].dt.normalize()

    result = {}
    for band_label, lat_lo, lat_hi in LAT_BANDS:
        mask = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
        daily = df[mask].groupby("date")[VALUE_COL].median()
        result[band_label] = daily
    return result

def plot_job(job: dict) -> None:
    label   = job["label"]
    parquet = job["parquet"]
    out_png = job["out_png"]
    colors  = job["color"]

    print(f"\n=== {label} ===")
    daily_bands = load_swarm_daily(parquet)

    fig, axes = plt.subplots(len(LAT_BANDS), 1, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(hspace=0.12)
    x_min = DATE_START - pd.Timedelta(hours=12)
    x_max = DATE_END + pd.Timedelta(hours=12)

    for ax, (band_label, _, _), color in zip(axes, LAT_BANDS, colors):
        daily = daily_bands[band_label]

        ax.axvspan(DATE_REF1_START, DATE_REF1_END, color="lightblue", alpha=0.25, label="Non-SSW ref")
        ax.axvspan(DATE_REF2_START, DATE_REF2_END, color="lightblue", alpha=0.25)
        ax.axvspan(DATE_SSW_START, DATE_SSW_END, color="lightyellow", alpha=0.40, label="SSW period")
        ax.axvline(DATE_SSW_PEAK, color="red", linewidth=1.5, linestyle="--", zorder=5, label="ERA5 T10hPa peak (09/19)")

        ax.plot(daily.index, daily.values, color=color, linewidth=2.0, marker="o", markersize=4, zorder=4,
                label=f"rho_ratio ({band_label.strip()})")

        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("rho_ratio\n(rho_obs / rho_MSIS)", fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        ax.text(0.01, 0.97, band_label.strip(), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

        h1, l1 = ax.get_legend_handles_labels()
        ax.legend(h1, l1, fontsize=7, loc="upper right", framealpha=0.85, ncol=2)

    axes[-1].set_xlabel("Date (2019)", fontsize=11)
    fig.suptitle(f"{label}  density_ratio_msis (2019 SH SSW)  [all LT]",
                 fontsize=12, fontweight="bold", y=0.96)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)

def main() -> None:
    for job in JOBS:
        plot_job(job)
    print("\nDone.")

if __name__ == "__main__":
    main()
