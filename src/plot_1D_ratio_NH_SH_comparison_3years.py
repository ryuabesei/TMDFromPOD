"""
plot_1D_ratio_NH_SH_comparison_3years.py

Purpose:
    Compare Northern Hemisphere (NH, lat > 0) vs Southern Hemisphere (SH, lat < 0)
    thermospheric density ratio residuals (delta ratio) for three major SSW events:
      - 2018 NH SSW (left column)
      - 2019 SH SSW (center column)
      - 2021 NH SSW (right column)

    Satellite: SWARM-C (available across all three years)
    Layout: 3 rows (latitude bands) x 3 columns (years)
    Latitude Bands:
      - High:  40-60°
      - Mid:   20-40°
      - Low:    0-20°

Output:
    Figure/comparison/1D_ratio_NH_SH_comparison_3years.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# ============================================================
# Paths & Settings
# ============================================================
PARQUET_2018 = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet")
PARQUET_2019 = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet")
PARQUET_2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")

OUT_PNG = Path("Figure/comparison/1D_ratio_NH_SH_comparison_3years.png")

VALUE_COL = "density_ratio_msis"

LAT_BANDS = [
    ("High  (40-60°)", 40.0, 60.0),
    ("Mid   (20-40°)", 20.0, 40.0),
    ("Low   ( 0-20°)",  0.0, 20.0),
]

# 2018 settings
DOY_START_2018, DOY_END_2018 = 30, 65
DOY_REF1_2018 = (30, 40)
DOY_REF2_2018 = (61, 65)
DOY_PEAK_2018 = 43  # Feb 12

# 2019 settings
DATE_START_2019 = pd.Timestamp("2019-08-20", tz="UTC")
DATE_END_2019   = pd.Timestamp("2019-09-23", tz="UTC")
DATE_REF1_2019  = (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC"))
DATE_REF2_2019  = (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC"))
DATE_PEAK_2019  = pd.Timestamp("2019-09-19", tz="UTC")

# 2021 settings
DATE_START_2021 = pd.Timestamp("2020-12-25", tz="UTC")
DATE_END_2021   = pd.Timestamp("2021-02-05", tz="UTC")
DATE_REF1_2021  = (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29", tz="UTC"))
DATE_REF2_2021  = (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05", tz="UTC"))
DATE_PEAK_2021  = pd.Timestamp("2021-01-04", tz="UTC")

# ============================================================
# Helper Functions to Load and Process Data
# ============================================================
def load_daily_NH_SH_2018(parquet: Path) -> dict[str, dict[str, pd.Series]]:
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", VALUE_COL])
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START_2018) & (df["DOY_int"] <= DOY_END_2018)]

    result = {}
    for band_label, lat_lo, lat_hi in LAT_BANDS:
        mask_abs = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
        sub = df[mask_abs]
        nh = sub[sub["lat"] > 0].groupby("DOY_int")[VALUE_COL].median()
        sh = sub[sub["lat"] < 0].groupby("DOY_int")[VALUE_COL].median()
        result[band_label] = {"NH": nh, "SH": sh}
    return result

def load_daily_NH_SH_date(parquet: Path, date_start: pd.Timestamp, date_end: pd.Timestamp) -> dict[str, dict[str, pd.Series]]:
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        raise KeyError(f"Latitude column not found in {parquet.name}")
    df = df.dropna(subset=["datetime", lat_col, VALUE_COL])
    df = df[(df["datetime"] >= date_start) & (df["datetime"] <= date_end + pd.Timedelta(hours=23, minutes=59))]
    df["date"] = df["datetime"].dt.normalize()

    result = {}
    for band_label, lat_lo, lat_hi in LAT_BANDS:
        mask_abs = (df[lat_col].abs() >= lat_lo) & (df[lat_col].abs() < lat_hi)
        sub = df[mask_abs]
        nh = sub[sub[lat_col] > 0].groupby("date")[VALUE_COL].median()
        sh = sub[sub[lat_col] < 0].groupby("date")[VALUE_COL].median()
        result[band_label] = {"NH": nh, "SH": sh}
    return result

def compute_residual_2018(daily: pd.Series) -> pd.Series:
    doy = daily.index
    mask = (
        ((doy >= DOY_REF1_2018[0]) & (doy <= DOY_REF1_2018[1])) |
        ((doy >= DOY_REF2_2018[0]) & (doy <= DOY_REF2_2018[1]))
    )
    if mask.sum() == 0 or daily[mask].isna().all():
        return daily * np.nan
    ref = float(daily[mask].median())
    return daily - ref

def compute_residual_date(daily: pd.Series, ref1: tuple[pd.Timestamp, pd.Timestamp], ref2: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
    idx = daily.index
    mask = (
        ((idx >= ref1[0]) & (idx <= ref1[1])) |
        ((idx >= ref2[0]) & (idx <= ref2[1]))
    )
    if mask.sum() == 0 or daily[mask].isna().all():
        return daily * np.nan
    ref = float(daily[mask].median())
    return daily - ref

# ============================================================
# Main Script
# ============================================================
def main() -> None:
    print("Loading data for all three years...")
    data_2018 = load_daily_NH_SH_2018(PARQUET_2018)
    data_2019 = load_daily_NH_SH_date(PARQUET_2019, DATE_START_2019, DATE_END_2019)
    data_2021 = load_daily_NH_SH_date(PARQUET_2021, DATE_START_2021, DATE_END_2021)

    print("Setting up figures...")
    n_bands = len(LAT_BANDS)
    fig = plt.figure(figsize=(18, 3.8 * n_bands))
    fig.suptitle(
        "SWARM-C Thermospheric Density Response: NH vs SH Comparison\n"
        "2018 NH SSW (left)  |  2019 SH SSW (center)  |  2021 NH SSW (right)\n"
        "Solid Blue: Northern Hemisphere (lat > 0)  |  Solid Red: Southern Hemisphere (lat < 0)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(n_bands, 3, figure=fig, hspace=0.22, wspace=0.12)

    # We will share y-axis per row (latitude band) to make responses comparable
    axes_row = []

    for row, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
        ax_2018 = fig.add_subplot(gs[row, 0])
        ax_2019 = fig.add_subplot(gs[row, 1])
        ax_2021 = fig.add_subplot(gs[row, 2])
        
        axes_row.append([ax_2018, ax_2019, ax_2021])

        # ----------------------------------------------------
        # 1. 2018 NH SSW (Left Column)
        # ----------------------------------------------------
        nh_2018 = data_2018[band_label]["NH"]
        sh_2018 = data_2018[band_label]["SH"]
        res_nh_2018 = compute_residual_2018(nh_2018)
        res_sh_2018 = compute_residual_2018(sh_2018)

        ax_2018.plot(res_nh_2018.index, res_nh_2018, color="#1f77b4", linewidth=2.0, label="NH")
        ax_2018.plot(res_sh_2018.index, res_sh_2018, color="#d62728", linewidth=2.0, label="SH")
        
        # Reference windows
        ax_2018.axvspan(DOY_REF1_2018[0], DOY_REF1_2018[1], color="gray", alpha=0.15)
        ax_2018.axvspan(DOY_REF2_2018[0], DOY_REF2_2018[1], color="gray", alpha=0.15)
        # SSW Peak
        ax_2018.axvline(DOY_PEAK_2018, color="black", linestyle="--", alpha=0.7)
        ax_2018.text(DOY_PEAK_2018 + 0.5, 0.9, "SSW Peak (DOY 43)", transform=ax_2018.get_xaxis_transform(),
                     fontsize=9, color="black", alpha=0.8)

        ax_2018.set_xlim(DOY_START_2018 - 0.5, DOY_END_2018 + 0.5)
        ax_2018.grid(True, linestyle=":", alpha=0.6)
        if row == 0:
            ax_2018.set_title("2018 NH SSW (SWARM-C)", fontsize=12, fontweight="bold")
        if row == n_bands - 1:
            ax_2018.set_xlabel("Day of Year (2018)", fontsize=10)
        ax_2018.set_ylabel(f"{band_label}\n" + r"$\Delta$ Density Ratio", fontsize=10)

        # ----------------------------------------------------
        # 2. 2019 SH SSW (Center Column)
        # ----------------------------------------------------
        nh_2019 = data_2019[band_label]["NH"]
        sh_2019 = data_2019[band_label]["SH"]
        res_nh_2019 = compute_residual_date(nh_2019, DATE_REF1_2019, DATE_REF2_2019)
        res_sh_2019 = compute_residual_date(sh_2019, DATE_REF1_2019, DATE_REF2_2019)

        ax_2019.plot(res_nh_2019.index, res_nh_2019, color="#1f77b4", linewidth=2.0, label="NH")
        ax_2019.plot(res_sh_2019.index, res_sh_2019, color="#d62728", linewidth=2.0, label="SH")

        # Reference windows
        ax_2019.axvspan(DATE_REF1_2019[0], DATE_REF1_2019[1], color="gray", alpha=0.15)
        ax_2019.axvspan(DATE_REF2_2019[0], DATE_REF2_2019[1], color="gray", alpha=0.15)
        # SSW Peak
        ax_2019.axvline(DATE_PEAK_2019, color="black", linestyle="--", alpha=0.7)
        ax_2019.text(DATE_PEAK_2019 + pd.Timedelta(hours=12), 0.9, "SSW Peak (09/19)", transform=ax_2019.get_xaxis_transform(),
                     fontsize=9, color="black", alpha=0.8)

        ax_2019.set_xlim(DATE_START_2019 - pd.Timedelta(hours=12), DATE_END_2019 + pd.Timedelta(hours=12))
        ax_2019.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_2019.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax_2019.grid(True, linestyle=":", alpha=0.6)
        if row == 0:
            ax_2019.set_title("2019 SH SSW (SWARM-C)", fontsize=12, fontweight="bold")
        if row == n_bands - 1:
            ax_2019.set_xlabel("Date (2019)", fontsize=10)

        # ----------------------------------------------------
        # 3. 2021 NH SSW (Right Column)
        # ----------------------------------------------------
        nh_2021 = data_2021[band_label]["NH"]
        sh_2021 = data_2021[band_label]["SH"]
        res_nh_2021 = compute_residual_date(nh_2021, DATE_REF1_2021, DATE_REF2_2021)
        res_sh_2021 = compute_residual_date(sh_2021, DATE_REF1_2021, DATE_REF2_2021)

        ax_2021.plot(res_nh_2021.index, res_nh_2021, color="#1f77b4", linewidth=2.0, label="NH")
        ax_2021.plot(res_sh_2021.index, res_sh_2021, color="#d62728", linewidth=2.0, label="SH")

        # Reference windows
        ax_2021.axvspan(DATE_REF1_2021[0], DATE_REF1_2021[1], color="gray", alpha=0.15)
        ax_2021.axvspan(DATE_REF2_2021[0], DATE_REF2_2021[1], color="gray", alpha=0.15)
        # SSW Peak
        ax_2021.axvline(DATE_PEAK_2021, color="black", linestyle="--", alpha=0.7)
        ax_2021.text(DATE_PEAK_2021 + pd.Timedelta(hours=12), 0.9, "SSW Peak (01/04)", transform=ax_2021.get_xaxis_transform(),
                     fontsize=9, color="black", alpha=0.8)

        ax_2021.set_xlim(DATE_START_2021 - pd.Timedelta(hours=12), DATE_END_2021 + pd.Timedelta(hours=12))
        ax_2021.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_2021.xaxis.set_major_locator(mdates.DayLocator(interval=6))
        ax_2021.grid(True, linestyle=":", alpha=0.6)
        if row == 0:
            ax_2021.set_title("2021 NH SSW (SWARM-C)", fontsize=12, fontweight="bold")
        if row == n_bands - 1:
            ax_2021.set_xlabel("Date (2020-21)", fontsize=10)

    # Share y-axis per row manually to ensure proper scaling comparison
    for row_axes in axes_row:
        # find the common y-limits across 2018, 2019, 2021 panels
        ymin = min(ax.get_ylim()[0] for ax in row_axes)
        ymax = max(ax.get_ylim()[1] for ax in row_axes)
        # Set symmetric or at least unified limits
        for ax in row_axes:
            ax.set_ylim(ymin, ymax)

    # Add legend to the top-right subplot
    axes_row[0][2].legend(loc="upper right", framealpha=0.9)

    # Create output directory if it doesn't exist
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving figure to {OUT_PNG}...")
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()
