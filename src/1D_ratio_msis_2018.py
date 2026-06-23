"""
1D_ratio_msis_2018.py

Purpose:
    Plot daily-mean density_ratio_msis (= rho_obs / rho_MSIS_real) as a 1D
    time series for DOY 20-80, 2018.

    Unlike the LT-sector version, ALL LT observations are included.
    Results are shown for three latitude bands:
        High:   |lat| 40-60 deg
        Mid:    |lat| 20-40 deg
        Low:    |lat|  0-20 deg

    Residual = daily_ratio - mean(non-SSW reference: DOY 20-40 & 61-80)

    COSMIC T(10 hPa) is overlaid for comparison.

Output:
    Figure/2018/1D_ratio_msis_2018_SWARM-{A,B,C}.png
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
        parquet = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        out_png = Path("Figure/2018/1D_ratio_msis_2018_SWARM-A_DOY30-65.png"),
        color   = ("#1f77b4", "#ff7f0e", "#2ca02c"),  # High / Mid / Low
    ),
    dict(
        label   = "SWARM-B",
        parquet = Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        out_png = Path("Figure/2018/1D_ratio_msis_2018_SWARM-B_DOY30-65.png"),
        color   = ("#1f77b4", "#ff7f0e", "#2ca02c"),
    ),
    dict(
        label   = "SWARM-C",
        parquet = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        out_png = Path("Figure/2018/1D_ratio_msis_2018_SWARM-C_DOY30-65.png"),
        color   = ("#1f77b4", "#ff7f0e", "#2ca02c"),
    ),
]

COSMIC_CSV    = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
COSMIC_LAT_LABEL = "60-90 deg N"

DOY_START, DOY_END = 30, 65

# Latitude bands (|lat|)
LAT_BANDS = [
    ("High  (40-60 deg)",  40.0, 60.0),
    ("Mid   (20-40 deg)",  20.0, 40.0),
    ("Low   ( 0-20 deg)",   0.0, 20.0),
]

# Non-SSW reference
DOY_REF1 = (30, 40)
DOY_REF2 = (61, 65)

# SSW period
DOY_SSW_START, DOY_SSW_END = 41, 60

VALUE_COL = "density_ratio_msis"


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
# Load COSMIC T(10 hPa)
# ============================================================
def load_cosmic_T10(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        print(f"  [INFO] COSMIC CSV not found: {csv_path} — skipping overlay")
        return pd.Series(dtype=float)
    print(f"  Loading COSMIC T(10 hPa) ...")
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "DOY", "T10_K"])
    df = df[(df["DOY"] >= DOY_START) & (df["DOY"] <= DOY_END)]
    s = df.set_index(df["DOY"].astype(int))["T10_K"]
    print(f"    {len(s)} days (DOY {s.index.min()}-{s.index.max()})")
    return s


# ============================================================
# Plot: 3-panel (one per latitude band)
# ============================================================
def plot_job(job: dict, cosmic_T10: pd.Series) -> None:
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
                marker="o", markersize=4,
                zorder=4, label=f"Residual ratio ({band_label.strip()})")

        # --- Right axis: COSMIC T(10 hPa) ---
        if len(cosmic_T10) > 0:
            ax2 = ax.twinx()
            ax2.plot(cosmic_T10.index.to_numpy(dtype=float),
                     cosmic_T10.values,
                     color="hotpink", linewidth=1.8, linestyle="-",
                     marker="s", markersize=3, alpha=0.85,
                     label=f"COSMIC T (10 hPa, {COSMIC_LAT_LABEL})", zorder=3)
            T_vals  = cosmic_T10.values
            T_margin = max((T_vals.max() - T_vals.min()) * 0.3, 5.0)
            ax2.set_ylim(T_vals.min() - T_margin, T_vals.max() + T_margin)
            ax2.set_ylabel("T (10 hPa) [K]", fontsize=10, color="hotpink")
            ax2.tick_params(axis="y", labelcolor="hotpink")
            h2, l2 = ax2.get_legend_handles_labels()
        else:
            h2, l2 = [], []

        # --- Decoration ---
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("delta ratio\n(ratio - ref)", fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        ax.set_xticks(range(DOY_START, DOY_END + 1, 5))
        ax.set_xlim(x_min, x_max)

        # Label inside panel
        ax.text(0.01, 0.97, band_label.strip(),
                transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")

        # Reference annotation
        ax.text(0.99, 0.97,
                f"ref ratio = {ref:.3f}",
                transform=ax.transAxes, fontsize=9,
                va="top", ha="right", color="gray")

        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2,
                  fontsize=8, loc="upper right",
                  framealpha=0.85, ncol=2)

    axes[-1].set_xlabel("Day of Year (2018)", fontsize=12)

    fig.suptitle(
        f"{label}  density_ratio_msis residual (DOY 30-65, 2018)  [all LT]\n"
        "Residual = daily median(rho_obs/rho_MSIS) - mean(non-SSW ref: DOY 30-40 & 61-65)",
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
    cosmic_T10 = load_cosmic_T10(COSMIC_CSV)
    for job in JOBS:
        plot_job(job, cosmic_T10)
    print("\nDone.")


if __name__ == "__main__":
    main()
