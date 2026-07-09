"""
plot_1D_ratio_only_2021.py

Purpose:
    Plot daily-median rho_ratio (rho_obs / rho_MSIS_real) as a 1D time series
    for the 2021 NH SSW period (2020-12-25 to 2021-02-05) for SWARM-C and GRACE-FO.
    Results are shown for three latitude bands:
        High:   |lat| 40-60 deg
        Mid:    |lat| 20-40 deg
        Low:    |lat|  0-20 deg

    Reference definition based on stratospheric temperature:
        Non-SSW ref: 2020-12-25 to 2020-12-29  AND  2021-02-01 to 2021-02-05
        SSW period:  2020-12-30 to 2021-01-31
        Peak date:   2021-01-04 (from T10hPa data)
        Onset date:  2021-01-05 (official zonal wind reverse)

Output:
    Figure/2021/1D_ratio_only_msis_2021_SWARM-C.png
    Figure/2021/1D_ratio_only_msis_2021_GRACE-FO.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

JOBS = [
    dict(
        label   = "SWARM-C",
        parquet = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet"),
        out_png = Path("Figure/2021/1D_ratio_only_msis_2021_SWARM-C.png"),
        color   = ("#4b0082", "#ba55d3", "#696969"),
    ),
    dict(
        label   = "GRACE-FO",
        parquet = Path("normalizeddata/2021/grace_fo_dns_2021_normalized_with_LT_removed.parquet"),
        out_png = Path("Figure/2021/1D_ratio_only_msis_2021_GRACE-FO.png"),
        color   = ("#4b0082", "#ba55d3", "#696969"),
    ),
]

DATE_START = pd.Timestamp("2020-12-25", tz="UTC")
DATE_END   = pd.Timestamp("2021-02-05", tz="UTC")

LAT_BANDS = [
    ("High  (40-60 deg)",  40.0, 60.0),
    ("Mid   (20-40 deg)",  20.0, 40.0),
    ("Low   ( 0-20 deg)",   0.0, 20.0),
]

DATE_REF1_START = pd.Timestamp("2020-12-25", tz="UTC")
DATE_REF1_END   = pd.Timestamp("2020-12-29", tz="UTC")
DATE_REF2_START = pd.Timestamp("2021-02-01", tz="UTC")
DATE_REF2_END   = pd.Timestamp("2021-02-05", tz="UTC")

DATE_SSW_START  = pd.Timestamp("2020-12-30", tz="UTC")
DATE_SSW_END    = pd.Timestamp("2021-01-31", tz="UTC")
DATE_SSW_ONSET  = pd.Timestamp("2021-01-05", tz="UTC")
DATE_SSW_PEAK   = pd.Timestamp("2021-01-04", tz="UTC")

VALUE_COL = "density_ratio_msis"

def load_data_daily(parquet: Path) -> dict[str, pd.Series]:
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        raise KeyError("Latitude column not found")
    df = df.dropna(subset=["datetime", lat_col, VALUE_COL])
    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] <= DATE_END + pd.Timedelta(hours=23, minutes=59))]
    df["date"] = df["datetime"].dt.normalize()

    result = {}
    for band_label, lat_lo, lat_hi in LAT_BANDS:
        mask = (df[lat_col].abs() >= lat_lo) & (df[lat_col].abs() < lat_hi)
        daily = df[mask].groupby("date")[VALUE_COL].median()
        result[band_label] = daily
    return result

def get_ref_median(daily: pd.Series) -> float:
    idx = daily.index
    mask = (
        ((idx >= DATE_REF1_START) & (idx <= DATE_REF1_END)) |
        ((idx >= DATE_REF2_START) & (idx <= DATE_REF2_END))
    )
    return float(daily[mask].median())

def plot_job(job: dict) -> None:
    label   = job["label"]
    parquet = job["parquet"]
    out_png = job["out_png"]
    colors  = job["color"]

    print(f"\n=== {label} ===")
    daily_bands = load_data_daily(parquet)

    fig, axes = plt.subplots(len(LAT_BANDS), 1, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(hspace=0.12)
    x_min = DATE_START - pd.Timedelta(hours=12)
    x_max = DATE_END + pd.Timedelta(hours=12)

    for ax, (band_label, _, _), color in zip(axes, LAT_BANDS, colors):
        daily = daily_bands[band_label]
        ref = get_ref_median(daily)

        # 参照期間とSSW期間のハイライト
        ax.axvspan(DATE_REF1_START, DATE_REF1_END, color="lightblue", alpha=0.25, label="Non-SSW ref")
        ax.axvspan(DATE_REF2_START, DATE_REF2_END, color="lightblue", alpha=0.25)
        ax.axvspan(DATE_SSW_START, DATE_SSW_END, color="lightyellow", alpha=0.40, label="SSW period")
        
        # Onset / Peak 線
        ax.axvline(DATE_SSW_ONSET, color="blue", linewidth=1.2, linestyle="--", zorder=5, label="Onset (01/05)")
        ax.axvline(DATE_SSW_PEAK, color="green", linewidth=1.2, linestyle="--", zorder=5, label="T10hPa Peak (01/04)")
        ax.axhline(ref, color="gray", linewidth=0.8, linestyle="--", zorder=2)

        # プロット
        ax.plot(daily.index, daily.values, color=color, linewidth=2.0, marker="o", markersize=4, zorder=4,
                label=f"rho_ratio ({band_label.strip()})")

        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("rho_ratio\n(obs / MSIS)", fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        ax.text(0.01, 0.97, band_label.strip(), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")
        if not np.isnan(ref):
            ax.text(0.99, 0.97, f"ref = {ref:.3f}", transform=ax.transAxes, fontsize=8,
                    va="top", ha="right", color="gray")

        h1, l1 = ax.get_legend_handles_labels()
        ax.legend(h1, l1, fontsize=7, loc="upper right", framealpha=0.85, ncol=2)

    axes[-1].set_xlabel("Date (2020/2021)", fontsize=11)
    fig.suptitle(f"{label}  density_ratio_msis (2021 NH SSW)  [all LT]",
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
