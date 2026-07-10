"""
plot_1D_ratio_with_temp_2019_by_LT.py

Purpose:
    Plot daily-median density_ratio_msis (= rho_obs / rho_MSIS_real) on the left axis,
    and overlay ERA5 10 hPa temperature (60-90°S average) on the right axis
    split by LT sector for SWARM-A, B, C during the 2019 SH SSW (2019-08-20 to 2019-09-23).

    LT sectors per satellite:
        SWARM-A/C: Dawn (LT 2.5–8.5 h) / Dusk (LT 14.5–20.5 h)
        SWARM-B:   Midnight (LT 0–4 h) / Noon (LT 12–16 h)

    Layout per satellite: 3 rows (latitude bands) x 2 columns (LT sectors)
    ERA5 T10hPa peak date (09/19) is highlighted.

Output:
    Figure/2019/1D_ratio_with_temp_2019_SWARM-{A,B,C}_by_LT.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

LT_DAWN_DUSK = [
    dict(label="Dawn  (LT 2.5–8.5 h)",   lt_min=2.5,  lt_max=8.5,  color="#1a6faf"),
    dict(label="Dusk  (LT 14.5–20.5 h)",  lt_min=14.5, lt_max=20.5, color="#e07b39"),
]
LT_MIDNIGHT_NOON = [
    dict(label="Midnight (LT 0–4 h)", lt_min=0,  lt_max=4,  color="#6a0dad"),
    dict(label="Noon     (LT 12–16 h)", lt_min=12, lt_max=16, color="#c0392b"),
]

SATELLITES = [
    dict(
        label      = "SWARM-A",
        parquet    = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        out_png    = Path("Figure/2019/1D_ratio_with_temp_2019_SWARM-A_by_LT.png"),
        lt_sectors = LT_DAWN_DUSK,
    ),
    dict(
        label      = "SWARM-B",
        parquet    = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        out_png    = Path("Figure/2019/1D_ratio_with_temp_2019_SWARM-B_by_LT.png"),
        lt_sectors = LT_MIDNIGHT_NOON,
    ),
    dict(
        label      = "SWARM-C",
        parquet    = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        out_png    = Path("Figure/2019/1D_ratio_with_temp_2019_SWARM-C_by_LT.png"),
        lt_sectors = LT_DAWN_DUSK,
    ),
]

DATE_START = pd.Timestamp("2019-08-20", tz="UTC")
DATE_END   = pd.Timestamp("2019-09-23", tz="UTC")

LAT_BANDS = [
    ("High  (40-60°)", 40.0, 60.0),
    ("Mid   (20-40°)", 20.0, 40.0),
    ("Low   ( 0-20°)",  0.0, 20.0),
]

DATE_REF1_START = pd.Timestamp("2019-08-20", tz="UTC")
DATE_REF1_END   = pd.Timestamp("2019-08-26", tz="UTC")
DATE_REF2_START = pd.Timestamp("2019-09-20", tz="UTC")
DATE_REF2_END   = pd.Timestamp("2019-09-23", tz="UTC")

DATE_SSW_START = pd.Timestamp("2019-08-27", tz="UTC")
DATE_SSW_END   = pd.Timestamp("2019-09-19", tz="UTC")
DATE_SSW_PEAK  = pd.Timestamp("2019-09-19", tz="UTC")

VALUE_COL = "density_ratio_msis"

def load_era5_temp() -> pd.Series:
    era5_dir = Path("data/SSW2019/ERA5")
    files = sorted(list(era5_dir.glob("*.nc")))
    if len(files) == 0:
        raise FileNotFoundError(f"No NetCDF files found in {era5_dir}")
    
    series_list = []
    for fp in files:
        print(f"  Processing temperature from {fp.name}...")
        try:
            with xr.open_dataset(fp) as ds:
                ds_sub = ds.sel(latitude=slice(-60.0, -90.0))
                weights = np.cos(np.deg2rad(ds_sub["latitude"]))
                weights.name = "weights"
                weighted_temp = ds_sub["t"].weighted(weights)
                temp_1d = weighted_temp.mean(dim=["latitude", "longitude"]).squeeze()
                
                times = temp_1d["valid_time"].values
                vals = temp_1d.values
                if temp_1d.ndim == 0:
                    s = pd.Series([vals], index=pd.to_datetime([times]))
                else:
                    s = pd.Series(vals, index=pd.to_datetime(times))
                series_list.append(s)
        except Exception as e:
            print(f"  Error processing {fp.name}: {e}")
            
    if not series_list:
        raise ValueError("No temperature data could be loaded.")
        
    df_temp = pd.concat(series_list).sort_index()
    df_temp = df_temp.groupby(df_temp.index).first()
    
    df_daily = df_temp.resample("D").mean()
    df_daily.index = df_daily.index.tz_localize("UTC")
    df_daily = df_daily.loc[DATE_START:DATE_END]
    return df_daily

def get_ref_median(daily: pd.Series) -> float:
    idx = daily.index
    mask = (
        ((idx >= DATE_REF1_START) & (idx <= DATE_REF1_END)) |
        ((idx >= DATE_REF2_START) & (idx <= DATE_REF2_END))
    )
    if mask.sum() == 0 or daily[mask].isna().all():
        return np.nan
    return float(daily[mask].median())

def plot_satellite(sat: dict, df_temp: pd.Series) -> None:
    label      = sat["label"]
    parquet    = sat["parquet"]
    out_png    = sat["out_png"]
    lt_sectors = sat["lt_sectors"]

    print(f"\n=== {label} ===")
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL])
    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] <= DATE_END)]
    df["date"] = df["datetime"].dt.normalize()
    print(f"  {len(df):,} rows after date filter")

    n_bands = len(LAT_BANDS)
    n_lt    = len(lt_sectors)
    x_min = DATE_START - pd.Timedelta(hours=12)
    x_max = DATE_END + pd.Timedelta(hours=12)

    fig, axes = plt.subplots(n_bands, n_lt, figsize=(6.8 * n_lt, 3.3 * n_bands), sharex=True, sharey="row")
    fig.subplots_adjust(hspace=0.08, wspace=0.18)

    for col_idx, lt in enumerate(lt_sectors):
        lt_label = lt["label"]
        lt_min   = lt["lt_min"]
        lt_max   = lt["lt_max"]
        color    = lt["color"]

        df_lt = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]
        print(f"  {lt_label}: {len(df_lt):,} obs")

        for bi, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
            ax = axes[bi, col_idx]
            mask = (df_lt["lat"].abs() >= lat_lo) & (df_lt["lat"].abs() < lat_hi)
            sub = df_lt[mask]
            daily = sub.groupby("date")[VALUE_COL].median()
            ref = get_ref_median(daily)

            # Left Y-axis: rho_ratio
            ax.axvspan(DATE_REF1_START, DATE_REF1_END, color="lightblue", alpha=0.20)
            ax.axvspan(DATE_REF2_START, DATE_REF2_END, color="lightblue", alpha=0.20)
            ax.axvspan(DATE_SSW_START, DATE_SSW_END, color="lightyellow", alpha=0.35)
            ax.axvline(DATE_SSW_PEAK, color="green", linewidth=1.2, linestyle="--", zorder=5)
            if not np.isnan(ref):
                ax.axhline(ref, color="gray", linewidth=0.8, linestyle="--", zorder=2)

            ax.plot(daily.index, daily.values, color=color, linewidth=2.0, marker="o", markersize=4, zorder=4,
                    label="rho_ratio")

            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(axis="y", alpha=0.3, linewidth=0.7)
            
            if col_idx == 0:
                ax.set_ylabel("rho_ratio\n(obs / MSIS)", fontsize=9, color=color)
            ax.tick_params(axis="y", labelcolor=color)

            # Right Y-axis: ERA5 Stratospheric Temperature
            ax2 = ax.twinx()
            ax2.plot(df_temp.index, df_temp.values, color="#d62728", linewidth=1.2, linestyle="-", marker="x", markersize=2.5, alpha=0.75, zorder=3,
                     label="ERA5 Temp")
            if col_idx == n_lt - 1:
                ax2.set_ylabel("Stratospheric Temp (60-90°S) [K]", fontsize=9, color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")

            ax.text(0.01, 0.97, band_label.strip(), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="left")
            if not np.isnan(ref):
                ax.text(0.99, 0.97, f"ref = {ref:.3f}", transform=ax.transAxes, fontsize=7,
                        va="top", ha="right", color="gray")

            if bi == 0:
                ax.set_title(lt_label, fontsize=11, fontweight="bold", pad=6, color=color)

            if bi == n_bands - 1:
                ax.set_xlabel("Date (2019)", fontsize=10)

    lt_names = "   |   ".join(lt["label"] for lt in lt_sectors)
    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, fc="lightblue",  alpha=0.3, label="Non-SSW ref"),
        plt.Rectangle((0, 0), 1, 1, fc="lightyellow", alpha=0.5, label="SSW period"),
        plt.Line2D([0], [0], color="green", lw=1.2, ls="--", label="ERA5 T10hPa peak"),
        plt.Line2D([0], [0], color="gray", lw=0.8, ls="--", label="Ref Median"),
        plt.Line2D([0], [0], color="#d62728", lw=1.2, marker="x", ms=4, label="ERA5 Temp (60-90°S)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"{label}  density_ratio_msis with Stratospheric Temp (2019 SH SSW)\n{lt_names}",
                 fontsize=11, fontweight="bold", y=1.01)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)

def main() -> None:
    print("Loading ERA5 stratospheric temperature (60-90°S)...")
    df_temp = load_era5_temp()
    
    for sat in SATELLITES:
        plot_satellite(sat, df_temp)
    print("\nDone.")

if __name__ == "__main__":
    main()
