"""
plot_1D_ratio_with_temp_2018_by_LT_nolat.py

Purpose:
    2018 NH SSW (SWARM-C) について、緯度帯分けなし（全緯度 0–60° 一括）で
    LTセクター別（Morning / Evening）に

        - rho_ratio (density_ratio_msis)  … 左軸
        - COSMIC T (10 hPa, 60–90°N)     … 右軸

    を重ね描きする。

Layout:
    1行 × 2列
      左列: Morning (LT 6–12 h)
      右列: Evening (LT 18–24 h)

Output:
    Figure/2018/1D_ratio_temp_2018_by_LT_nolat.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# ============================================================
# Paths
# ============================================================
PARQUET = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet")
COSMIC_CSV = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
OUT_PNG = Path("Figure/2018/1D_ratio_temp_2018_by_LT_nolat.png")

VALUE_COL = "density_ratio_msis"

# ============================================================
# Settings
# ============================================================
DOY_START, DOY_END = 30, 65

# Reference & SSW periods (DOY)
DOY_REF1 = (30, 40)
DOY_SSW  = (40, 61)     # SSW onset to recovery
DOY_REF2 = (61, 65)
DOY_PEAK = 43           # 2018-02-12

# Latitude range (absolute)
LAT_ABS_MIN, LAT_ABS_MAX = 0.0, 60.0

# LT sectors
LT_SECTORS = [
    dict(label="Morning (LT 6–12 h)",  lt_min=6,  lt_max=12, color="#1f77b4"),
    dict(label="Evening (LT 18–24 h)", lt_min=18, lt_max=24, color="#d62728"),
]

# ============================================================
# Helpers
# ============================================================
def doy_to_date(doy: int, year: int = 2018) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-01-01", tz="UTC") + pd.Timedelta(days=doy - 1)


def load_swarm(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", VALUE_COL, "lst_h"])
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]
    # Latitude filter
    df = df[(df["lat"].abs() >= LAT_ABS_MIN) & (df["lat"].abs() < LAT_ABS_MAX)]
    df["date"] = df["datetime"].dt.normalize()
    return df


def load_cosmic(csv: Path) -> pd.Series:
    df = pd.read_csv(csv)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "T10_K"])
    df = df[(df["DOY"] >= DOY_START) & (df["DOY"] <= DOY_END)]
    s = df.set_index("datetime")["T10_K"].sort_index()
    return s


def compute_daily_ratio(df_lt: pd.DataFrame) -> pd.Series:
    return df_lt.groupby("date")[VALUE_COL].median()


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("Loading SWARM-C data...")
    df = load_swarm(PARQUET)

    print("Loading COSMIC temperature...")
    temp = load_cosmic(COSMIC_CSV)

    # Convert DOY axes to dates for alignment
    x_start = doy_to_date(DOY_START)
    x_end   = doy_to_date(DOY_END)

    ref1_start = doy_to_date(DOY_REF1[0])
    ref1_end   = doy_to_date(DOY_REF1[1])
    ssw_start  = doy_to_date(DOY_SSW[0])
    ssw_end    = doy_to_date(DOY_SSW[1])
    ref2_start = doy_to_date(DOY_REF2[0])
    ref2_end   = doy_to_date(DOY_REF2[1])
    peak_date  = doy_to_date(DOY_PEAK)

    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "SWARM-C  2018 NH SSW  —  density ratio & COSMIC T (10 hPa, 60–90°N)\n"
        "All latitudes 0–60°  |  Morning (LT 6–12 h)  vs  Evening (LT 18–24 h)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    rho_axes = []
    temp_axes = []

    for ci, lt in enumerate(LT_SECTORS):
        ax_r = axes[ci]
        ax_t = ax_r.twinx()

        rho_axes.append(ax_r)
        temp_axes.append(ax_t)

        # ── Background shading ──────────────────────────────
        ax_r.axvspan(ref1_start, ref1_end, color="lightblue", alpha=0.35, lw=0,
                     label="Non-SSW ref")
        ax_r.axvspan(ref2_start, ref2_end, color="lightblue", alpha=0.35, lw=0)
        ax_r.axvspan(ssw_start,  ssw_end,  color="lightyellow", alpha=0.45, lw=0,
                     label="SSW period")
        ax_r.axvline(peak_date, color="tomato", lw=2.0, ls="--", zorder=6,
                     label=f"SSW peak (DOY {DOY_PEAK})")
        ax_r.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.5)

        # ── Rho ratio ────────────────────────────────────────
        lo_lt, hi_lt = lt["lt_min"], lt["lt_max"]
        df_lt = df[(df["lst_h"] >= lo_lt) & (df["lst_h"] < hi_lt)].copy()
        daily_ratio = compute_daily_ratio(df_lt)

        ax_r.plot(daily_ratio.index, daily_ratio.values,
                  color=lt["color"], lw=2.2, marker="o", markersize=4.5,
                  label="ρ_ratio (obs/MSIS)", zorder=5)
        ax_r.set_ylabel("ρ_ratio  (rho_obs / rho_MSIS)", fontsize=11,
                         fontweight="bold")
        ax_r.set_xlabel("Date (2018)", fontsize=10)
        ax_r.grid(True, linestyle=":", alpha=0.55)
        ax_r.set_xlim(x_start - pd.Timedelta(hours=12),
                      x_end   + pd.Timedelta(hours=12))

        # ── COSMIC temperature ───────────────────────────────
        ax_t.plot(temp.index, temp.values,
                  color="hotpink", lw=2.2, marker="s", markersize=3.5,
                  ls="-", zorder=4,
                  label="COSMIC T (10 hPa, 60–90°N)")
        ax_t.set_ylabel("T  [K]  (10 hPa, 60–90°N)", fontsize=10,
                         fontweight="bold", color="hotpink")
        ax_t.tick_params(axis="y", labelcolor="hotpink")
        ax_t.spines["right"].set_edgecolor("hotpink")

        # ── X-axis formatting ────────────────────────────────
        ax_r.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax_r.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # ── Title ────────────────────────────────────────────
        ax_r.set_title(lt["label"], fontsize=12, fontweight="bold", pad=8)

        # ── Panel legend ─────────────────────────────────────
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        lines_t, labels_t = ax_t.get_legend_handles_labels()
        ax_r.legend(lines_r + lines_t, labels_r + labels_t,
                    fontsize=9, loc="upper left", framealpha=0.88)

    # ── Unify rho y-axis across panels ──────────────────────
    ymin = min(ax.get_ylim()[0] for ax in rho_axes)
    ymax = max(ax.get_ylim()[1] for ax in rho_axes)
    for ax in rho_axes:
        ax.set_ylim(ymin, ymax)

    # ── Unify temperature y-axis across panels ───────────────
    tmin = min(ax.get_ylim()[0] for ax in temp_axes)
    tmax = max(ax.get_ylim()[1] for ax in temp_axes)
    for ax in temp_axes:
        ax.set_ylim(tmin, tmax)

    # ── Figure-level legend (background shading) ─────────────
    legend_elems = [
        mpatches.Patch(facecolor="lightblue",   alpha=0.5, label="Non-SSW ref period"),
        mpatches.Patch(facecolor="lightyellow", alpha=0.7, label="SSW period"),
        plt.Line2D([0], [0], color="tomato", lw=1.8, ls="--", label="SSW peak (DOY 43 / Feb 12)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=10,
               framealpha=0.88, bbox_to_anchor=(0.5, -0.04))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
