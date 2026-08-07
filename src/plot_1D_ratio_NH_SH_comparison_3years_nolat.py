"""
plot_1D_ratio_NH_SH_comparison_3years_nolat.py

Purpose:
    Compare Northern Hemisphere (NH, lat > 0) vs Southern Hemisphere (SH, lat < 0)
    thermospheric density ratio residuals (delta ratio) for three major SSW events:
      - 2018 NH SSW (left column)
      - 2019 SH SSW (center column)
      - 2021 NH SSW (right column)

    No latitude band separation — all latitudes (0–60°) are combined into a single panel.

    Satellite: SWARM-C (available across all three years)
    Layout: 1 row x 3 columns (years)

Output:
    Figure/comparison/1D_ratio_NH_SH_comparison_3years_nolat.png
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

OUT_PNG = Path("Figure/comparison/1D_ratio_NH_SH_comparison_3years_nolat.png")

VALUE_COL = "density_ratio_msis"

# Latitude range to include (absolute value)
LAT_ABS_MIN = 0.0
LAT_ABS_MAX = 60.0

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
# Helper Functions
# ============================================================
def load_daily_NH_SH_2018(parquet: Path) -> dict[str, pd.Series]:
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", VALUE_COL])
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START_2018) & (df["DOY_int"] <= DOY_END_2018)]

    mask_abs = (df["lat"].abs() >= LAT_ABS_MIN) & (df["lat"].abs() < LAT_ABS_MAX)
    sub = df[mask_abs]
    nh = sub[sub["lat"] > 0].groupby("DOY_int")[VALUE_COL].median()
    sh = sub[sub["lat"] < 0].groupby("DOY_int")[VALUE_COL].median()
    return {"NH": nh, "SH": sh}


def load_daily_NH_SH_date(parquet: Path, date_start: pd.Timestamp, date_end: pd.Timestamp) -> dict[str, pd.Series]:
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        raise KeyError(f"Latitude column not found in {parquet.name}")
    df = df.dropna(subset=["datetime", lat_col, VALUE_COL])
    df = df[(df["datetime"] >= date_start) & (df["datetime"] <= date_end + pd.Timedelta(hours=23, minutes=59))]
    df["date"] = df["datetime"].dt.normalize()

    mask_abs = (df[lat_col].abs() >= LAT_ABS_MIN) & (df[lat_col].abs() < LAT_ABS_MAX)
    sub = df[mask_abs]
    nh = sub[sub[lat_col] > 0].groupby("date")[VALUE_COL].median()
    sh = sub[sub[lat_col] < 0].groupby("date")[VALUE_COL].median()
    return {"NH": nh, "SH": sh}


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

    print("Setting up figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.suptitle(
        "SWARM-C Thermospheric Density Response: NH vs SH Comparison (All Latitudes 0–60°)\n"
        "2018 NH SSW (left)  |  2019 SH SSW (center)  |  2021 NH SSW (right)\n"
        "Blue: Northern Hemisphere (lat > 0)  |  Red: Southern Hemisphere (lat < 0)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    ax_2018, ax_2019, ax_2021 = axes

    # ----------------------------------------------------------
    # 1. 2018 NH SSW
    # ----------------------------------------------------------
    nh_2018 = data_2018["NH"]
    sh_2018 = data_2018["SH"]
    res_nh_2018 = compute_residual_2018(nh_2018)
    res_sh_2018 = compute_residual_2018(sh_2018)

    ax_2018.plot(res_nh_2018.index, res_nh_2018, color="#1f77b4", linewidth=2.0, marker="o", markersize=4, label="NH")
    ax_2018.plot(res_sh_2018.index, res_sh_2018, color="#d62728", linewidth=2.0, marker="o", markersize=4, label="SH")

    ax_2018.axvspan(DOY_REF1_2018[0], DOY_REF1_2018[1], color="gray", alpha=0.15, label="Ref period")
    ax_2018.axvspan(DOY_REF2_2018[0], DOY_REF2_2018[1], color="gray", alpha=0.15)
    ax_2018.axvline(DOY_PEAK_2018, color="black", linestyle="--", alpha=0.7)
    ax_2018.text(DOY_PEAK_2018 + 0.5, 0.92, f"SSW Peak\n(DOY {DOY_PEAK_2018})",
                 transform=ax_2018.get_xaxis_transform(), fontsize=8.5, color="black", alpha=0.85)
    ax_2018.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

    ax_2018.set_xlim(DOY_START_2018 - 0.5, DOY_END_2018 + 0.5)
    ax_2018.set_title("2018 NH SSW (SWARM-C)", fontsize=12, fontweight="bold")
    ax_2018.set_xlabel("Day of Year (2018)", fontsize=10)
    ax_2018.set_ylabel(r"$\Delta$ Density Ratio  (rho_obs / rho_MSIS)", fontsize=10)
    ax_2018.grid(True, linestyle=":", alpha=0.6)
    ax_2018.legend(loc="upper right", framealpha=0.9)

    # ----------------------------------------------------------
    # 2. 2019 SH SSW
    # ----------------------------------------------------------
    nh_2019 = data_2019["NH"]
    sh_2019 = data_2019["SH"]
    res_nh_2019 = compute_residual_date(nh_2019, DATE_REF1_2019, DATE_REF2_2019)
    res_sh_2019 = compute_residual_date(sh_2019, DATE_REF1_2019, DATE_REF2_2019)

    ax_2019.plot(res_nh_2019.index, res_nh_2019, color="#1f77b4", linewidth=2.0, marker="o", markersize=4, label="NH")
    ax_2019.plot(res_sh_2019.index, res_sh_2019, color="#d62728", linewidth=2.0, marker="o", markersize=4, label="SH")

    ax_2019.axvspan(DATE_REF1_2019[0], DATE_REF1_2019[1], color="gray", alpha=0.15, label="Ref period")
    ax_2019.axvspan(DATE_REF2_2019[0], DATE_REF2_2019[1], color="gray", alpha=0.15)
    ax_2019.axvline(DATE_PEAK_2019, color="black", linestyle="--", alpha=0.7)
    ax_2019.text(DATE_PEAK_2019 + pd.Timedelta(hours=12), 0.92, "SSW Peak\n(09/19)",
                 transform=ax_2019.get_xaxis_transform(), fontsize=8.5, color="black", alpha=0.85)
    ax_2019.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

    ax_2019.set_xlim(DATE_START_2019 - pd.Timedelta(hours=12), DATE_END_2019 + pd.Timedelta(hours=12))
    ax_2019.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax_2019.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax_2019.set_title("2019 SH SSW (SWARM-C)", fontsize=12, fontweight="bold")
    ax_2019.set_xlabel("Date (2019)", fontsize=10)
    ax_2019.grid(True, linestyle=":", alpha=0.6)
    ax_2019.legend(loc="upper right", framealpha=0.9)

    # ----------------------------------------------------------
    # 3. 2021 NH SSW
    # ----------------------------------------------------------
    nh_2021 = data_2021["NH"]
    sh_2021 = data_2021["SH"]
    res_nh_2021 = compute_residual_date(nh_2021, DATE_REF1_2021, DATE_REF2_2021)
    res_sh_2021 = compute_residual_date(sh_2021, DATE_REF1_2021, DATE_REF2_2021)

    ax_2021.plot(res_nh_2021.index, res_nh_2021, color="#1f77b4", linewidth=2.0, marker="o", markersize=4, label="NH")
    ax_2021.plot(res_sh_2021.index, res_sh_2021, color="#d62728", linewidth=2.0, marker="o", markersize=4, label="SH")

    ax_2021.axvspan(DATE_REF1_2021[0], DATE_REF1_2021[1], color="gray", alpha=0.15, label="Ref period")
    ax_2021.axvspan(DATE_REF2_2021[0], DATE_REF2_2021[1], color="gray", alpha=0.15)
    ax_2021.axvline(DATE_PEAK_2021, color="black", linestyle="--", alpha=0.7)
    ax_2021.text(DATE_PEAK_2021 + pd.Timedelta(hours=12), 0.92, "SSW Peak\n(01/04)",
                 transform=ax_2021.get_xaxis_transform(), fontsize=8.5, color="black", alpha=0.85)
    ax_2021.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

    ax_2021.set_xlim(DATE_START_2021 - pd.Timedelta(hours=12), DATE_END_2021 + pd.Timedelta(hours=12))
    ax_2021.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax_2021.xaxis.set_major_locator(mdates.DayLocator(interval=6))
    ax_2021.set_title("2021 NH SSW (SWARM-C)", fontsize=12, fontweight="bold")
    ax_2021.set_xlabel("Date (2020-21)", fontsize=10)
    ax_2021.grid(True, linestyle=":", alpha=0.6)
    ax_2021.legend(loc="upper right", framealpha=0.9)

    # Unified y-axis across all panels
    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    print(f"Saving figure to {OUT_PNG}...")
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print("Done!")


if __name__ == "__main__":
    main()
