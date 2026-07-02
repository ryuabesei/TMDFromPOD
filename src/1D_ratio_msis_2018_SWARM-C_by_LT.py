"""
1D_ratio_msis_2018_SWARM-C_by_LT.py

Purpose:
    Plot daily-median density_ratio_msis split by LT sector for SWARM-C.

    SWARM-C observes two LT sectors during DOY 30-65:
        Morning: LT  6 - 12 h
        Evening: LT 18 - 24 h
    (Nearly same as SWARM-A, but slightly shifted ~1h)

    Residual = daily_ratio - median(non-SSW reference: DOY 30-40 & 61-65)
    COSMIC T(10 hPa) overlaid for comparison.

Output:
    Figure/2018/1D_ratio_msis_2018_SWARM-C_DOY30-65_by_LT.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARQUET    = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet")
COSMIC_CSV = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
OUT_PNG    = Path("Figure/2018/1D_ratio_msis_2018_SWARM-C_DOY30-65_by_LT.png")

DOY_START, DOY_END = 30, 65

# SWARM-C の LT セクター（SWARM-A と類似）
LT_SECTORS = [
    ("Morning (LT 6-12h)",  6,  12, "#1f77b4"),   # 青
    ("Evening (LT 18-24h)", 18, 24, "#d62728"),   # 赤
]

LAT_BANDS = [
    ("High  (40-60°)", 40.0, 60.0),
    ("Mid   (20-40°)", 20.0, 40.0),
    ("Low   ( 0-20°)",  0.0, 20.0),
]

DOY_REF1 = (30, 40)
DOY_REF2 = (61, 65)
DOY_SSW_START, DOY_SSW_END = 41, 60

VALUE_COL = "density_ratio_msis"
COSMIC_LAT_LABEL = "60–90°N"


def load_cosmic_T10(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "DOY", "T10_K"])
    df = df[(df["DOY"] >= DOY_START) & (df["DOY"] <= DOY_END)]
    return df.set_index(df["DOY"].astype(int))["T10_K"]


def compute_residual(daily: pd.Series) -> tuple[float, pd.Series]:
    doy = daily.index
    mask = (
        ((doy >= DOY_REF1[0]) & (doy <= DOY_REF1[1])) |
        ((doy >= DOY_REF2[0]) & (doy <= DOY_REF2[1]))
    )
    if mask.sum() == 0 or daily[mask].isna().all():
        return np.nan, daily * np.nan
    ref = float(daily[mask].median())
    return ref, daily - ref


def main() -> None:
    print("Loading parquet ...")
    df = pd.read_parquet(PARQUET)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL])
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]
    print(f"  {len(df):,} rows")

    cosmic_T10 = load_cosmic_T10(COSMIC_CSV)

    n_rows = len(LAT_BANDS)
    n_cols = len(LT_SECTORS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey="row")
    fig.subplots_adjust(hspace=0.08, wspace=0.06)
    x_min, x_max = DOY_START - 0.5, DOY_END + 0.5

    for col_idx, (lt_label, lt_min, lt_max, lt_color) in enumerate(LT_SECTORS):
        df_lt = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]
        print(f"\n  {lt_label}: {len(df_lt):,} obs")

        for row_idx, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
            ax = axes[row_idx, col_idx]
            mask = (df_lt["lat"].abs() >= lat_lo) & (df_lt["lat"].abs() < lat_hi)
            sub = df_lt[mask]
            daily = sub.groupby("DOY_int")[VALUE_COL].median()
            print(f"    {band_label}: {len(daily)} days")

            ref, residual = compute_residual(daily)

            ax.axvspan(*DOY_REF1, color="lightblue", alpha=0.20, label="Non-SSW ref")
            ax.axvspan(*DOY_REF2, color="lightblue", alpha=0.20)
            ax.axvspan(DOY_SSW_START, DOY_SSW_END, color="lightyellow", alpha=0.40, label="SSW period")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)
            ax.plot(residual.index, residual.values,
                    color=lt_color, linewidth=2.0, marker="o", markersize=4,
                    zorder=4, label=f"Δratio ({band_label.strip()})")

            if len(cosmic_T10) > 0:
                ax2 = ax.twinx()
                ax2.plot(cosmic_T10.index.to_numpy(dtype=float), cosmic_T10.values,
                         color="hotpink", linewidth=1.6, linestyle="-",
                         marker="s", markersize=3, alpha=0.85,
                         label=f"COSMIC T (10hPa, {COSMIC_LAT_LABEL})", zorder=3)
                T_vals = cosmic_T10.values
                T_margin = max((T_vals.max() - T_vals.min()) * 0.3, 5.0)
                ax2.set_ylim(T_vals.min() - T_margin, T_vals.max() + T_margin)
                ax2.tick_params(axis="y", labelcolor="hotpink")
                if col_idx == n_cols - 1:
                    ax2.set_ylabel("T (10 hPa) [K]", fontsize=9, color="hotpink")
                else:
                    ax2.set_yticklabels([])
                h2, l2 = ax2.get_legend_handles_labels()
            else:
                h2, l2 = [], []

            ax.set_xlim(x_min, x_max)
            ax.set_xticks(range(DOY_START, DOY_END + 1, 5))
            ax.grid(axis="y", alpha=0.3, linewidth=0.7)
            ax.tick_params(axis="y", labelleft=True)
            if col_idx == 0:
                ax.set_ylabel("Δratio (ratio − ref)", fontsize=10)
            ax.text(0.01, 0.97, band_label.strip(), transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="top", ha="left")
            if not np.isnan(ref):
                ax.text(0.99, 0.97, f"ref ratio = {ref:.3f}",
                        transform=ax.transAxes, fontsize=8, va="top", ha="right", color="gray")
            if row_idx == 0:
                h1, l1 = ax.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right",
                          framealpha=0.85, ncol=2)
                ax.set_title(lt_label, fontsize=12, fontweight="bold", pad=6)

        axes[-1, col_idx].set_xlabel("Day of Year (2018)", fontsize=11)

    fig.suptitle(
        "SWARM-C  density_ratio_msis residual  (DOY 30-65, 2018)\n"
        "Residual = daily median(rho_obs/rho_MSIS) − median(non-SSW ref: DOY 30-40 & 61-65)\n"
        "Left: Morning (LT 6-12h)   |   Right: Evening (LT 18-24h)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {OUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
