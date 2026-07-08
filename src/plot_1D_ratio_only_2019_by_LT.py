"""
plot_1D_ratio_only_2019_by_LT.py

Purpose:
    Plot daily-median density_ratio_msis (= rho_obs / rho_MSIS_real) split by LT sector
    for SWARM-A, B, C during the 2019 SH SSW (2019-08-20 to 2019-09-23) [ratio only].

    LT sectors per satellite:
        SWARM-A/C: Dawn (LT 4–8 h) / Dusk (LT 16–20 h)
        SWARM-B:   Midnight (LT 0–4 h) / Noon (LT 12–16 h)

    Layout per satellite: 3 rows (latitude bands) x 2 columns (LT sectors)
    ERA5 T10hPa peak date (09/19) is highlighted.

Output:
    Figure/2019/1D_ratio_only_msis_2019_SWARM-{A,B,C}_by_LT.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
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
        out_png    = Path("Figure/2019/1D_ratio_only_msis_2019_SWARM-A_by_LT.png"),
        lt_sectors = LT_DAWN_DUSK,
    ),
    dict(
        label      = "SWARM-B",
        parquet    = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        out_png    = Path("Figure/2019/1D_ratio_only_msis_2019_SWARM-B_by_LT.png"),
        lt_sectors = LT_MIDNIGHT_NOON,
    ),
    dict(
        label      = "SWARM-C",
        parquet    = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        out_png    = Path("Figure/2019/1D_ratio_only_msis_2019_SWARM-C_by_LT.png"),
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

def plot_satellite(sat: dict) -> None:
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

    fig, axes = plt.subplots(n_bands, n_lt, figsize=(6.5 * n_lt, 3.2 * n_bands), sharex=True, sharey="row")
    fig.subplots_adjust(hspace=0.08, wspace=0.06)

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

            ax.axvspan(DATE_REF1_START, DATE_REF1_END, color="lightblue", alpha=0.25, label="Non-SSW ref")
            ax.axvspan(DATE_REF2_START, DATE_REF2_END, color="lightblue", alpha=0.25)
            ax.axvspan(DATE_SSW_START, DATE_SSW_END, color="lightyellow", alpha=0.40, label="SSW period")
            ax.axvline(DATE_SSW_PEAK, color="red", linewidth=1.5, linestyle="--", zorder=5, label="ERA5 T10hPa peak")

            ax.plot(daily.index, daily.values, color=color, linewidth=2.0, marker="o", markersize=4, zorder=4,
                    label=f"rho_ratio ({band_label.strip()})")

            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(axis="y", alpha=0.3, linewidth=0.7)
            ax.tick_params(axis="y", labelleft=True)
            
            if col_idx == 0:
                ax.set_ylabel("rho_ratio\n(rho_obs / rho_MSIS)", fontsize=9)

            ax.text(0.01, 0.97, band_label.strip(), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="left")

            if bi == 0:
                ax.set_title(lt_label, fontsize=11, fontweight="bold", pad=6, color=color)

            if bi == n_bands - 1:
                ax.set_xlabel("Date (2019)", fontsize=10)

    lt_names = "   |   ".join(lt["label"] for lt in lt_sectors)
    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, fc="lightblue",  alpha=0.4, label="Non-SSW ref"),
        plt.Rectangle((0, 0), 1, 1, fc="lightyellow", alpha=0.6, label="SSW period"),
        plt.Line2D([0], [0], color="red", lw=1.5, ls="--", label="ERA5 T10hPa peak"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{label}  density_ratio_msis (2019 SH SSW)\n{lt_names}",
                 fontsize=11, fontweight="bold", y=1.01)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)

def main() -> None:
    for sat in SATELLITES:
        plot_satellite(sat)
    print("\nDone.")

if __name__ == "__main__":
    main()
