"""
compare_LT_removal_effect.py

Purpose:
    Compare the 1D daily-mean residuals of:
    1) density_ratio_msis (LT dependency removed by dividing by MSIS_real)
    2) density_norm (LT dependency NOT removed, since rho_model_ref still contains LT diurnal variation)

    To make them directly comparable, we plot:
    - For density_ratio_msis: ratio - ref_ratio (dimensionless)
    - For density_norm: (density_norm - ref_norm) / ref_norm (relative change, dimensionless)

    Results are shown for SWARM-A, B, C for DOY 30-65, 2018.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JOBS = [
    dict(
        label   = "SWARM-A",
        parquet = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_DOY20-80.parquet"),
        out_png = Path("Figure/2018/compare_LT_removal_SWARM-A_DOY30-65.png"),
    ),
    dict(
        label   = "SWARM-B",
        parquet = Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized_DOY20-80.parquet"),
        out_png = Path("Figure/2018/compare_LT_removal_SWARM-B_DOY30-65.png"),
    ),
    dict(
        label   = "SWARM-C",
        parquet = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_DOY20-80.parquet"),
        out_png = Path("Figure/2018/compare_LT_removal_SWARM-C_DOY30-65.png"),
    ),
]

DOY_START, DOY_END = 30, 65

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

def compute_residual_ratio(daily: pd.Series) -> tuple[float, pd.Series]:
    doy = daily.index
    mask = (
        ((doy >= DOY_REF1[0]) & (doy <= DOY_REF1[1])) |
        ((doy >= DOY_REF2[0]) & (doy <= DOY_REF2[1]))
    )
    ref = float(daily[mask].median())
    return ref, daily - ref

def compute_residual_norm_relative(daily: pd.Series) -> tuple[float, pd.Series]:
    doy = daily.index
    mask = (
        ((doy >= DOY_REF1[0]) & (doy <= DOY_REF1[1])) |
        ((doy >= DOY_REF2[0]) & (doy <= DOY_REF2[1]))
    )
    ref = float(daily[mask].median())
    relative_residual = (daily - ref) / ref
    return ref, relative_residual

def plot_comparison(job: dict) -> None:
    label   = job["label"]
    parquet = job["parquet"]
    out_png = job["out_png"]

    print(f"Loading {parquet.name} ...")
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]

    fig, axes = plt.subplots(len(LAT_BANDS), 1, figsize=(12, 4 * len(LAT_BANDS)), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    x_min, x_max = DOY_START - 0.5, DOY_END + 0.5

    for ax, (band_label, lat_lo, lat_hi) in zip(axes, LAT_BANDS):
        mask = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
        sub = df[mask]

        # 1) LT Removed (density_ratio_msis)
        daily_ratio = sub.groupby("DOY_int")["density_ratio_msis"].median()
        ref_ratio, res_ratio = compute_residual_ratio(daily_ratio)

        # 2) LT NOT Removed (density_norm)
        daily_norm = sub.groupby("DOY_int")["density_norm"].median()
        ref_norm, res_norm_rel = compute_residual_norm_relative(daily_norm)

        # Plot shadings
        ax.axvspan(DOY_REF1[0], DOY_REF1[1], color="lightblue", alpha=0.15, label="Non-SSW ref")
        ax.axvspan(DOY_REF2[0], DOY_REF2[1], color="lightblue", alpha=0.15)
        ax.axvspan(DOY_SSW_START, DOY_SSW_END, color="lightyellow", alpha=0.35, label="SSW period")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        # Plot curves
        ax.plot(res_ratio.index, res_ratio.values, 
                color="blue", linewidth=2.0, marker="o", markersize=4, zorder=4,
                label="LT Removed (Ratio Residual)")
        
        ax.plot(res_norm_rel.index, res_norm_rel.values, 
                color="red", linewidth=1.8, linestyle="--", marker="s", markersize=4, zorder=3,
                label="LT NOT Removed (Relative Density Residual)")

        # Decorate
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("Relative Variation (dimensionless)", fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        ax.set_xticks(range(DOY_START, DOY_END + 1, 5))
        ax.legend(fontsize=9, loc="upper right")

        # Info inside panel
        ax.text(0.01, 0.97, f"{band_label.strip()}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")
        
        ax.text(0.01, 0.03, f"ref ratio: {ref_ratio:.3f}\nref norm: {ref_norm:.3e} kg/m³",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="left", color="gray")

    axes[-1].set_xlabel("Day of Year (2018)", fontsize=12)
    fig.suptitle(
        f"{label}: Influence of LT (Local Time) Removal on 1D Residuals (DOY 30-65)\n"
        "Solid Blue: LT Removed (rho_obs/rho_MSIS - ref) | Dashed Red: LT NOT Removed (relative change of density_norm)",
        fontsize=12, fontweight="bold", y=0.98
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)

def main() -> None:
    for job in JOBS:
        plot_comparison(job)
    print("Comparison plotting done.")

if __name__ == "__main__":
    main()
