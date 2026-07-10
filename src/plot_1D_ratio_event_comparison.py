"""
plot_1D_ratio_event_comparison.py

Purpose:
    Compare thermospheric density variations between:
        - 2018 NH SSW (Northern Hemisphere Stratospheric Sudden Warming)
        - 2019 SH SSW (Southern Hemisphere Stratospheric Sudden Warming)

    X-axis: Event-relative days (SSW peak = Day 0)
    Y-axis: Δρ_ratio = daily median(density_ratio_msis) − median(non-SSW reference)

    Layout per satellite: 3 rows (latitude bands) × 2 columns (LT sectors)
    Each panel overlays:
        Blue  line: 2018 NH SSW
        Orange line: 2019 SH SSW

Event settings:
    2018 NH SSW:
        SSW peak: DOY 43 (2018-02-12)
        Plot range: DOY 30–65 → Day −13 to +22
        Reference: DOY 30–40 (pre) & DOY 61–65 (post)

    2019 SH SSW:
        SSW peak: 2019-09-19
        Plot range: 2019-08-20 to 2019-09-23 → Day −30 to +4
        Reference: 2019-08-20–26 (pre) & 2019-09-20–23 (post)

Output:
    Figure/comparison/1D_ratio_event_comparison_SWARM-{A,B,C}.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Event settings
# ============================================================

# 2018 NH SSW
DOY_PEAK_2018    = 43                    # Feb 12
DOY_START_2018   = 30
DOY_END_2018     = 65
DOY_REF1_2018    = (30, 40)              # pre-SSW reference
DOY_REF2_2018    = (61, 65)             # post-SSW reference
DOY_SSW_S_2018   = 41                   # SSW onset (DOY)
DOY_SSW_E_2018   = 60                   # SSW end (DOY)

# 2019 SH SSW
DATE_PEAK_2019   = pd.Timestamp("2019-09-19", tz="UTC")
DATE_START_2019  = pd.Timestamp("2019-08-20", tz="UTC")
DATE_END_2019    = pd.Timestamp("2019-09-23", tz="UTC")
DATE_REF1S_2019  = pd.Timestamp("2019-08-20", tz="UTC")
DATE_REF1E_2019  = pd.Timestamp("2019-08-26", tz="UTC")
DATE_REF2S_2019  = pd.Timestamp("2019-09-20", tz="UTC")
DATE_REF2E_2019  = pd.Timestamp("2019-09-23", tz="UTC")
DATE_SSW_S_2019  = pd.Timestamp("2019-08-27", tz="UTC")
DATE_SSW_E_2019  = pd.Timestamp("2019-09-19", tz="UTC")

# Relative day ranges (for x-axis)
# 2018: Day -13 (+22)   2019: Day -30 to +4
XMIN = -30
XMAX = +22

# ============================================================
# Satellites
# ============================================================
LT_DAWN_DUSK = [
    dict(label="Dawn  (LT 2.5–8.5 h)",  lt_min=2.5,  lt_max=8.5,  color_2018="#1a6faf", color_2019="#e07b39"),
    dict(label="Dusk  (LT 14.5–20.5 h)", lt_min=14.5, lt_max=20.5, color_2018="#1a6faf", color_2019="#e07b39"),
]
LT_MIDNIGHT_NOON = [
    dict(label="Midnight (LT 0–4 h)",   lt_min=0,  lt_max=4,  color_2018="#1a6faf", color_2019="#e07b39"),
    dict(label="Noon     (LT 12–16 h)", lt_min=12, lt_max=16, color_2018="#1a6faf", color_2019="#e07b39"),
]

SATELLITES = [
    dict(
        label      = "SWARM-A",
        p2018      = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        p2019      = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        lt_sectors = LT_DAWN_DUSK,
        out_png    = Path("Figure/comparison/1D_ratio_event_comparison_SWARM-A.png"),
    ),
    dict(
        label      = "SWARM-B",
        p2018      = Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        p2019      = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        lt_sectors = LT_MIDNIGHT_NOON,
        out_png    = Path("Figure/comparison/1D_ratio_event_comparison_SWARM-B.png"),
    ),
    dict(
        label      = "SWARM-C",
        p2018      = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        p2019      = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        lt_sectors = LT_DAWN_DUSK,
        out_png    = Path("Figure/comparison/1D_ratio_event_comparison_SWARM-C.png"),
    ),
]

LAT_BANDS = [
    ("High  (40–60°)", 40.0, 60.0),
    ("Mid   (20–40°)", 20.0, 40.0),
    ("Low   ( 0–20°)",  0.0, 20.0),
]

VALUE_COL = "density_ratio_msis"


# ============================================================
# Data loading helpers
# ============================================================

def load_2018(parquet: Path, lt_min: float, lt_max: float) -> pd.DataFrame:
    """Load 2018 data, filter by DOY and LT, return with 'rel_day' column."""
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL])
    df["DOY"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY"] >= DOY_START_2018) & (df["DOY"] <= DOY_END_2018)]
    df = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]
    df["rel_day"] = df["DOY"] - DOY_PEAK_2018
    return df


def load_2019(parquet: Path, lt_min: float, lt_max: float) -> pd.DataFrame:
    """Load 2019 data, filter by date and LT, return with 'rel_day' column."""
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL])
    df = df[(df["datetime"] >= DATE_START_2019) & (df["datetime"] <= DATE_END_2019)]
    df = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]
    df["date"] = df["datetime"].dt.normalize()
    df["rel_day"] = (df["date"] - DATE_PEAK_2019).dt.days
    return df


# ============================================================
# Residual calculation
# ============================================================

def compute_residual_2018(daily: pd.Series) -> pd.Series:
    """
    daily.index = rel_day (integer)
    Reference: pre-SSW (rel_day -13 to -3) and post-SSW (rel_day +18 to +22)
    """
    ref_mask = (
        ((daily.index >= DOY_REF1_2018[0] - DOY_PEAK_2018) & (daily.index <= DOY_REF1_2018[1] - DOY_PEAK_2018)) |
        ((daily.index >= DOY_REF2_2018[0] - DOY_PEAK_2018) & (daily.index <= DOY_REF2_2018[1] - DOY_PEAK_2018))
    )
    if ref_mask.sum() == 0 or daily[ref_mask].isna().all():
        return daily * np.nan
    ref = float(daily[ref_mask].median())
    return daily - ref


def compute_residual_2019(daily: pd.Series) -> pd.Series:
    """
    daily.index = rel_day (integer, days since 2019-09-19)
    Reference: pre-SSW (Day -30 to -24) and post-SSW (Day +1 to +4)
    """
    pre_start  = (DATE_REF1S_2019 - DATE_PEAK_2019).days   # -30
    pre_end    = (DATE_REF1E_2019 - DATE_PEAK_2019).days   # -24
    post_start = (DATE_REF2S_2019 - DATE_PEAK_2019).days   # +1
    post_end   = (DATE_REF2E_2019 - DATE_PEAK_2019).days   # +4
    ref_mask = (
        ((daily.index >= pre_start)  & (daily.index <= pre_end)) |
        ((daily.index >= post_start) & (daily.index <= post_end))
    )
    if ref_mask.sum() == 0 or daily[ref_mask].isna().all():
        return daily * np.nan
    ref = float(daily[ref_mask].median())
    return daily - ref


# ============================================================
# Plot one satellite
# ============================================================

def plot_satellite(sat: dict) -> None:
    label      = sat["label"]
    p2018      = sat["p2018"]
    p2019      = sat["p2019"]
    lt_sectors = sat["lt_sectors"]
    out_png    = sat["out_png"]

    print(f"\n=== {label} ===")

    n_bands = len(LAT_BANDS)
    n_lt    = len(lt_sectors)

    fig, axes = plt.subplots(
        n_bands, n_lt,
        figsize=(7.0 * n_lt, 3.8 * n_bands),
        sharex=True, sharey="row"
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    # Relative day boundaries for shading
    # 2018 SSW period in rel_day
    ssw_s_2018 = DOY_SSW_S_2018 - DOY_PEAK_2018   # -2
    ssw_e_2018 = DOY_SSW_E_2018 - DOY_PEAK_2018   # +17
    ref1_s_2018 = DOY_REF1_2018[0] - DOY_PEAK_2018  # -13
    ref1_e_2018 = DOY_REF1_2018[1] - DOY_PEAK_2018  # -3
    ref2_s_2018 = DOY_REF2_2018[0] - DOY_PEAK_2018  # +18
    ref2_e_2018 = DOY_REF2_2018[1] - DOY_PEAK_2018  # +22

    # 2019 SSW period in rel_day
    ssw_s_2019  = (DATE_SSW_S_2019  - DATE_PEAK_2019).days   # -23
    ssw_e_2019  = (DATE_SSW_E_2019  - DATE_PEAK_2019).days   # 0
    pre_s_2019  = (DATE_REF1S_2019  - DATE_PEAK_2019).days   # -30
    pre_e_2019  = (DATE_REF1E_2019  - DATE_PEAK_2019).days   # -24
    post_s_2019 = (DATE_REF2S_2019  - DATE_PEAK_2019).days   # +1
    post_e_2019 = (DATE_REF2E_2019  - DATE_PEAK_2019).days   # +4

    for col_idx, lt in enumerate(lt_sectors):
        lt_label   = lt["label"]
        lt_min     = lt["lt_min"]
        lt_max     = lt["lt_max"]
        c_2018     = lt["color_2018"]
        c_2019     = lt["color_2019"]

        print(f"  LT sector: {lt_label}")

        # Load and filter data
        df18 = load_2018(p2018, lt_min, lt_max)
        df19 = load_2019(p2019, lt_min, lt_max)
        print(f"    2018: {len(df18):,} obs | 2019: {len(df19):,} obs")

        for row_idx, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
            ax = axes[row_idx, col_idx]

            # --- Filter by |lat| ---
            mask18 = (df18["lat"].abs() >= lat_lo) & (df18["lat"].abs() < lat_hi)
            mask19 = (df19["lat"].abs() >= lat_lo) & (df19["lat"].abs() < lat_hi)

            sub18 = df18[mask18]
            sub19 = df19[mask19]

            daily18 = sub18.groupby("rel_day")[VALUE_COL].median()
            daily19 = sub19.groupby("rel_day")[VALUE_COL].median()

            resid18 = compute_residual_2018(daily18)
            resid19 = compute_residual_2019(daily19)

            print(f"    {band_label}: 2018 {len(daily18)} days | 2019 {len(daily19)} days")

            # --------------------------------------------------------
            # Background shading
            # --------------------------------------------------------
            # 2018 reference windows (light blue, hatch)
            ax.axvspan(ref1_s_2018, ref1_e_2018, color="#4da6ff", alpha=0.12, zorder=1)
            ax.axvspan(ref2_s_2018, ref2_e_2018, color="#4da6ff", alpha=0.12, zorder=1)
            # 2019 reference windows (light green, lighter)
            ax.axvspan(pre_s_2019,  pre_e_2019,  color="#4db86b", alpha=0.10, zorder=1)
            ax.axvspan(post_s_2019, post_e_2019, color="#4db86b", alpha=0.10, zorder=1)

            # 2018 SSW onset period (yellow)
            ax.axvspan(ssw_s_2018, ssw_e_2018, color="gold", alpha=0.18, zorder=1)
            # 2019 SSW onset period (salmon)
            ax.axvspan(ssw_s_2019, ssw_e_2019, color="salmon", alpha=0.12, zorder=1)

            # SSW peak line (Day 0)
            ax.axvline(0, color="black", linewidth=1.2, linestyle="--", zorder=5, alpha=0.7)

            # Zero reference
            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", zorder=2)

            # --------------------------------------------------------
            # Plot residuals
            # --------------------------------------------------------
            ax.plot(
                resid18.index.to_numpy(dtype=float),
                resid18.values,
                color=c_2018, linewidth=2.2, marker="o", markersize=4,
                zorder=4, label="2018 NH SSW",
            )
            ax.plot(
                resid19.index.to_numpy(dtype=float),
                resid19.values,
                color=c_2019, linewidth=2.2, marker="s", markersize=4,
                zorder=4, label="2019 SH SSW", linestyle="--",
            )

            # --------------------------------------------------------
            # Decoration
            # --------------------------------------------------------
            ax.set_xlim(XMIN, XMAX)
            ax.grid(axis="y", alpha=0.3, linewidth=0.7)
            ax.tick_params(axis="y", labelleft=True)

            ax.text(0.01, 0.97, band_label.strip(),
                    transform=ax.transAxes, fontsize=10, fontweight="bold",
                    va="top", ha="left")

            if col_idx == 0:
                ax.set_ylabel("Δρ_ratio\n(ratio − ref)", fontsize=10)

            if row_idx == 0:
                ax.set_title(lt_label, fontsize=11, fontweight="bold", pad=8)

            if row_idx == n_bands - 1:
                ax.set_xlabel("Days relative to SSW peak (Day 0)", fontsize=10)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, loc="upper left", framealpha=0.85)

    # --------------------------------------------------------
    # Global legend
    # --------------------------------------------------------
    legend_handles = [
        mpatches.Patch(color="#4da6ff", alpha=0.4,  label="2018 reference window"),
        mpatches.Patch(color="gold",    alpha=0.45, label="2018 SSW period (DOY 41–60)"),
        mpatches.Patch(color="#4db86b", alpha=0.35, label="2019 reference window"),
        mpatches.Patch(color="salmon",  alpha=0.35, label="2019 SSW period (Aug 27 – Sep 19)"),
        plt.Line2D([0], [0], color="black", lw=1.2, ls="--", label="SSW peak (Day 0)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=5, fontsize=8.5,
        framealpha=0.90, bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        f"{label}  —  Thermospheric Density Response: NH SSW vs SH SSW\n"
        f"Blue solid: 2018 NH SSW (DOY 30–65, peak=DOY 43)   "
        f"Orange dashed: 2019 SH SSW (Aug 20 – Sep 23, peak=Sep 19)\n"
        f"Y-axis: Δρ_ratio = daily median(ρ_obs/ρ_MSIS) − median(non-SSW reference)\n"
        f"X-axis: Days relative to SSW stratospheric temperature peak (Day 0)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    for sat in SATELLITES:
        plot_satellite(sat)
    print("\nAll done.")


if __name__ == "__main__":
    main()
