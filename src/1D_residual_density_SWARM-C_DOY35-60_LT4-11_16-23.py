"""
1D_residual_density_SWARM-C_DOY35-60_LT4-11_16-23.py

Purpose:
    Compute and plot thermospheric density residuals (1D time series)
    from Swarm-C normalized density data, with COSMIC T(10 hPa) overlay.
    Analysis period narrowed to DOY 35-60 to reduce LT-width variation.

Steps:
    1. Load Swarm-C normalized density (from DOY20-80 parquet, filter to DOY 35-60)
    2. Filter by lat (-60 to 60) and LT sector (Morning: 04-11 LT / Evening: 16-23 LT)
    3. Compute daily mean density per sector
    4. Define reference as mean over non-SSW period (DOY 35-40 only)
    5. Residual = daily_density - reference
    6. Load COSMIC T(10 hPa) daily mean from pre-computed CSV
    7. Plot 2-panel figure (Morning / Evening) with SSW shading + COSMIC overlay

Output:
    Figure/1D_residual_SWARM-C_DOY35-60_LT4-11_16-23.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Settings
# ============================================================
NORM_PARQUET  = Path("normalizeddata/swarm_dnscpod_2018_normalized_DOY20-80.parquet")
COSMIC_CSV    = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
OUT_PNG       = Path("Figure/1D_residual_SWARM-C_DOY35-60_LT4-11_16-23.png")

DOY_START, DOY_END = 35, 60          # analysis range (narrowed from 20-80)
LAT_MIN, LAT_MAX   = -60.0, 60.0    # latitude filter

SECTOR_MORNING = (4, 11)
SECTOR_EVENING = (16, 23)

# SSW period
DOY_SSW_START, DOY_SSW_END = 41, 60

# Non-SSW reference period (DOY 35-40 only; no post-SSW data in this window)
DOY_NONSSW_BEFORE = (35, 40)

# COSMIC: latitude range label (for annotation only)
COSMIC_LAT_LABEL = "60–90°N"


# ============================================================
# Step 1-3: Load Swarm, compute daily mean per LT sector
# ============================================================
def load_swarm_daily(parquet: Path) -> dict[str, pd.Series]:
    """Returns dict of {sector_label: daily_mean_Series indexed by DOY (integer)}"""
    print("Loading Swarm-C normalized density ...")
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.dropna(subset=["datetime", "lat", "lst_h", "density_norm"])

    # DOY (fractional)
    dt = df["datetime"]
    df["DOY"] = (dt.dt.dayofyear
                 + dt.dt.hour / 24.0
                 + dt.dt.minute / 1440.0)

    # Latitude filter
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)]

    # DOY integer filter (narrowed to DOY 35-60)
    df["DOY_int"] = dt.dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]

    result = {}
    for label, (lt_min, lt_max) in [
        ("morning", SECTOR_MORNING),
        ("evening", SECTOR_EVENING),
    ]:
        sec = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]
        daily = sec.groupby("DOY_int")["density_norm"].mean().rename(label)
        result[label] = daily
        print(f"  {label}: {len(daily)} days with data")

    return result


# ============================================================
# Step 4-5: Reference and residual
# ============================================================
def compute_residual(daily: pd.Series) -> tuple[float, pd.Series]:
    """
    Reference = mean of non-SSW days (DOY 35-40 only).
    Returns (reference_value, residual_series).
    """
    doy = daily.index
    nonssw_mask = (
        (doy >= DOY_NONSSW_BEFORE[0]) & (doy <= DOY_NONSSW_BEFORE[1])
    )
    ref = float(daily[nonssw_mask].mean())
    residual = daily - ref
    return ref, residual


# ============================================================
# Step 6: COSMIC T(10 hPa) — load from pre-computed CSV
# ============================================================
def load_cosmic_T10(csv_path: Path) -> pd.Series:
    """
    Load pre-computed daily T(10 hPa) [K] from CSV.
    Returns Series indexed by DOY (integer).
    """
    print(f"Loading COSMIC T(10 hPa) from {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    # Filter to analysis DOY range
    df = df[(df["DOY"] >= DOY_START) & (df["DOY"] <= DOY_END)]
    series = df["T10_K"]
    series.index = df["DOY"].astype(int)  # index = DOY integer
    series = series.sort_index()
    print(f"  COSMIC: {len(series)} days  (DOY {series.index.min()}–{series.index.max()})")
    return series


# ============================================================
# Step 7: Publication-quality 2-panel plot
# ============================================================
def plot_residuals(
    res_morning: pd.Series,
    res_evening:  pd.Series,
    ref_morning:  float,
    ref_evening:  float,
    cosmic_T10:   pd.Series,
) -> None:

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    sectors = [
        (axes[0], res_morning, ref_morning, "Morning (04–11 LT)", "#2A6AE0"),
        (axes[1], res_evening, ref_evening, "Evening (16–23 LT)", "#E05C2A"),
    ]

    # Shared x limits
    x_min, x_max = DOY_START - 0.5, DOY_END + 0.5

    for ax, res, ref, label, color in sectors:
        doy  = res.index.to_numpy(dtype=float)
        vals = res.to_numpy(dtype=float)

        # --- SSW period shading ---
        ax.axvspan(DOY_SSW_START, DOY_SSW_END,
                   color="lightyellow", alpha=0.60, label="SSW period",
                   zorder=1)

        # --- Reference period shading ---
        ax.axvspan(DOY_NONSSW_BEFORE[0], DOY_NONSSW_BEFORE[1],
                   color="lightblue", alpha=0.30, label="Non-SSW reference",
                   zorder=1)

        # --- Zero line ---
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)

        # --- Residual line ---
        ax.plot(doy, vals,
                color=color, linewidth=2.0, marker="o", markersize=4,
                zorder=4, label=f"Residual ({label})")

        # --- Right axis: COSMIC T(10 hPa) ---
        if len(cosmic_T10) > 0:
            ax2 = ax.twinx()
            ax2.plot(cosmic_T10.index.to_numpy(dtype=float),
                     cosmic_T10.values,
                     color="hotpink", linewidth=1.8, linestyle="-",
                     marker="s", markersize=3, alpha=0.85,
                     label=f"COSMIC T (10 hPa, {COSMIC_LAT_LABEL})", zorder=3)
            T_vals = cosmic_T10.values
            T_margin = max((T_vals.max() - T_vals.min()) * 0.3, 5.0)
            ax2.set_ylim(T_vals.min() - T_margin, T_vals.max() + T_margin)
            ax2.set_ylabel("T (10 hPa) [K]", fontsize=11, color="hotpink")
            ax2.tick_params(axis="y", labelcolor="hotpink")
            # Merge legends
            h2, l2 = ax2.get_legend_handles_labels()
        else:
            h2, l2 = [], []

        # Labels
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("Residual density [kg m$^{-3}$]", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)

        # Tick marks at every 5 DOY
        ax.set_xticks(range(DOY_START, DOY_END + 1, 5))

        # Legend (combined)
        h1, l1 = ax.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left",
                  framealpha=0.85, ncol=2)

        # Subtitle inside panel
        ax.text(0.01, 0.97, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

        # Reference annotation
        ax.text(0.99, 0.97,
                f"ref = {ref:.3e} kg m$^{{-3}}$",
                transform=ax.transAxes, fontsize=9,
                va="top", ha="right", color="gray")

    axes[1].set_xlabel("Day of Year (2018)", fontsize=12)

    fig.suptitle(
        "Swarm-C Residual Normalized Density (DOY 35–60, 2018)\n"
        "Reference: mean over non-SSW period (DOY 35–40)  [LT: 04–11 / 16–23]",
        fontsize=13, fontweight="bold", y=0.995,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {OUT_PNG}")
    plt.show()


# ============================================================
# Main
# ============================================================
def main() -> None:
    # Swarm daily means
    daily = load_swarm_daily(NORM_PARQUET)
    ref_m, res_m = compute_residual(daily["morning"])
    ref_e, res_e = compute_residual(daily["evening"])
    print(f"  Morning reference: {ref_m:.4e} kg/m³")
    print(f"  Evening reference: {ref_e:.4e} kg/m³")

    # COSMIC T(10 hPa)
    cosmic_T10 = load_cosmic_T10(COSMIC_CSV)

    # Plot
    plot_residuals(res_m, res_e, ref_m, ref_e, cosmic_T10)


if __name__ == "__main__":
    main()
