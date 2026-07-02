"""
1D_ratio_msis_2019.py

Purpose:
    Plot daily-mean density_ratio_msis (= rho_obs / rho_MSIS_real) as a 1D
    time series for DOY 252-266 (2019-09-09 to 2019-09-23), 2019 Southern Hemisphere SSW.

    Unlike the LT-sector version, ALL LT observations are included.
    Results are shown for three latitude bands:
        High:   |lat| 40-60 deg
        Mid:    |lat| 20-40 deg
        Low:    |lat|  0-20 deg

    Residual = daily_ratio - median(non-SSW reference: DOY 252-255 & 263-266)

    Note: 2019 SSW is a Southern Hemisphere SSW (austral winter).
    DOY 252 = Sep 9, DOY 266 = Sep 23.
    SSW peak period: approx DOY 256-262 (Sep 13-19).

Output:
    Figure/2019/1D_ratio_msis_2019_SWARM-{A,B,C}.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Settings
# ============================================================
JOBS = [
    dict(
        label   = "SWARM-A",
        parquet = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png = Path("Figure/2019/1D_ratio_msis_2019_SWARM-A.png"),
        color   = ("#1f77b4", "#ff7f0e", "#2ca02c"),  # High / Mid / Low
    ),
    dict(
        label   = "SWARM-B",
        parquet = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png = Path("Figure/2019/1D_ratio_msis_2019_SWARM-B.png"),
        color   = ("#1f77b4", "#ff7f0e", "#2ca02c"),
    ),
    dict(
        label   = "SWARM-C",
        parquet = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png = Path("Figure/2019/1D_ratio_msis_2019_SWARM-C.png"),
        color   = ("#1f77b4", "#ff7f0e", "#2ca02c"),
    ),
]

DOY_START, DOY_END = 252, 266   # Sep 9 - Sep 23

# Latitude bands (|lat|)
LAT_BANDS = [
    ("High  (40-60 deg)",  40.0, 60.0),
    ("Mid   (20-40 deg)",  20.0, 40.0),
    ("Low   ( 0-20 deg)",   0.0, 20.0),
]

# Non-SSW reference periods (first 4 days and last 4 days)
DOY_REF1 = (252, 255)   # Sep 9-12 (before SSW peak)
DOY_REF2 = (263, 266)   # Sep 20-23 (after SSW peak)

# SSW period (approximate peak)
DOY_SSW_START, DOY_SSW_END = 256, 262   # Sep 13-19

VALUE_COL = "density_ratio_msis"

# x-axis labels: DOY -> date string
DOY_DATE_MAP = {
    252: "Sep 9",
    253: "Sep 10",
    254: "Sep 11",
    255: "Sep 12",
    256: "Sep 13",
    257: "Sep 14",
    258: "Sep 15",
    259: "Sep 16",
    260: "Sep 17",
    261: "Sep 18",
    262: "Sep 19",
    263: "Sep 20",
    264: "Sep 21",
    265: "Sep 22",
    266: "Sep 23",
}

# ============================================================
# Load Swarm: daily mean per latitude band
# ============================================================
def load_swarm_daily(parquet: Path) -> dict[str, pd.Series]:
    """Returns {band_label: daily_mean_Series} indexed by DOY (integer)."""
    print(f"  Loading {parquet.name} ...")
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        raise KeyError("Latitude column not found")
    if lat_col != "lat":
        df = df.rename(columns={lat_col: "lat"})

    df = df.dropna(subset=["datetime", "lat", VALUE_COL])

    dt = df["datetime"]
    df["DOY_int"] = dt.dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]

    result = {}
    for band_label, lat_lo, lat_hi in LAT_BANDS:
        mask = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
        sub = df[mask]
        daily = sub.groupby("DOY_int")[VALUE_COL].median()
        result[band_label] = daily
        print(f"    {band_label}: {len(daily)} days")

    return result


# ============================================================
# Compute residual vs non-SSW reference
# ============================================================
def compute_residual(daily: pd.Series) -> tuple[float, pd.Series]:
    """Reference = median of non-SSW periods. Returns (ref, residual)."""
    doy = daily.index
    mask = (
        ((doy >= DOY_REF1[0]) & (doy <= DOY_REF1[1])) |
        ((doy >= DOY_REF2[0]) & (doy <= DOY_REF2[1]))
    )
    ref = float(daily[mask].median())
    return ref, daily - ref


# ============================================================
# Plot: 3-panel (one per latitude band)
# ============================================================
def plot_job(job: dict) -> None:
    label   = job["label"]
    parquet = job["parquet"]
    out_png = job["out_png"]
    colors  = job["color"]

    print(f"\n=== {label} ===")
    daily_bands = load_swarm_daily(parquet)

    fig, axes = plt.subplots(len(LAT_BANDS), 1,
                             figsize=(12, 3.5 * len(LAT_BANDS)),
                             sharex=True)
    fig.subplots_adjust(hspace=0.10)

    x_min, x_max = DOY_START - 0.5, DOY_END + 0.5

    for ax, (band_label, _, _), color in zip(axes, LAT_BANDS, colors):
        daily = daily_bands[band_label]
        ref, residual = compute_residual(daily)

        doy  = residual.index.to_numpy(dtype=float)
        vals = residual.to_numpy(dtype=float)

        # --- Shading ---
        ax.axvspan(DOY_REF1[0], DOY_REF1[1],
                   color="lightblue", alpha=0.20, label="Non-SSW ref")
        ax.axvspan(DOY_REF2[0], DOY_REF2[1],
                   color="lightblue", alpha=0.20)
        ax.axvspan(DOY_SSW_START, DOY_SSW_END,
                   color="lightyellow", alpha=0.40, label="SSW period")

        # --- Zero line ---
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)

        # --- Residual line ---
        ax.plot(doy, vals,
                color=color, linewidth=2.0,
                marker="o", markersize=5,
                zorder=4, label=f"Residual ratio ({band_label.strip()})")

        # --- Decoration ---
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("delta ratio\n(ratio - ref)", fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        ax.set_xticks(range(DOY_START, DOY_END + 1))

        # Date labels on x axis
        tick_labels = [DOY_DATE_MAP.get(d, str(d)) for d in range(DOY_START, DOY_END + 1)]
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

        # Label inside panel
        ax.text(0.01, 0.97, band_label.strip(),
                transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")

        # Reference annotation
        ax.text(0.99, 0.97,
                f"ref ratio = {ref:.3f}",
                transform=ax.transAxes, fontsize=9,
                va="top", ha="right", color="gray")

        # Legend
        h1, l1 = ax.get_legend_handles_labels()
        ax.legend(h1, l1,
                  fontsize=8, loc="upper right",
                  framealpha=0.85, ncol=2)

    axes[-1].set_xlabel("Date (2019)", fontsize=12)

    fig.suptitle(
        f"{label}  density_ratio_msis residual (DOY 252-266, 2019 SH SSW)  [all LT]\n"
        "Residual = daily median(rho_obs/rho_MSIS) - median(non-SSW ref: Sep9-12 & Sep20-23)",
        fontsize=12, fontweight="bold", y=1.005,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    for job in JOBS:
        plot_job(job)
    print("\nDone.")


if __name__ == "__main__":
    main()
