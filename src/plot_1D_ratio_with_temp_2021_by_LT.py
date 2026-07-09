"""
plot_1D_ratio_with_temp_2021_by_LT.py

Purpose:
    Plot daily-median rho_ratio (left axis) and ERA5 10 hPa temperature (right axis)
    split by LT sector for SWARM-C and GRACE-FO during the 2021 NH SSW (2020-12-25 to 2021-02-05).

    LT sectors are defined dynamically per satellite to avoid data gaps due to precession:
        SWARM-C:  Dawn (LT 4.5–10.5 h) / Dusk (LT 16.5–22.5 h)
        GRACE-FO: Night/Dawn (LT 0.0–10.5 h) / Day/Dusk (LT 12.0–22.5 h)

    Layout per satellite: 3 rows (latitude bands) x 2 columns (LT sectors)

Output:
    Figure/2021/1D_ratio_with_temp_2021_SWARM-C_by_LT.png
    Figure/2021/1D_ratio_with_temp_2021_GRACE-FO.png
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

LT_GRACE = [
    dict(label="Night/Dawn (LT 0.0–10.5 h)", lt_min=0.0,  lt_max=10.5,  color="#1a6faf"),
    dict(label="Day/Dusk (LT 12.0–22.5 h)",   lt_min=12.0, lt_max=22.5, color="#e07b39"),
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

    print(f"\n=== {label} ===")
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    df = df.dropna(subset=["datetime", lat_col, "lst_h", VALUE_COL])
    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] <= DATE_END + pd.Timedelta(hours=23, minutes=59))]
    df["date"] = df["datetime"].dt.normalize()
    print(f"  {len(df):,} rows loaded after filtering")

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
