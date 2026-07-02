"""
1D_ratio_msis_2019_by_LT.py

Purpose:
    Plot daily-median density_ratio_msis split by LT sector for
    SWARM-A, B, C during the 2019 SH SSW (DOY 252-266, Sep 9-23).

    All three satellites in 2019 observe:
        Nightside: LT  0 -  6 h  (midnight / pre-dawn)
        Dayside:   LT 12 - 18 h  (noon / post-noon)

    For each LT sector, results are shown for three latitude bands:
        High:   |lat| 40-60 deg
        Mid:    |lat| 20-40 deg
        Low:    |lat|  0-20 deg

    Residual = daily_ratio - median(non-SSW ref: DOY 252-255 & 263-266)

Output:
    Figure/2019/1D_ratio_msis_2019_SWARM-{A,B,C}_by_LT.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 衛星設定
# ============================================================
SATELLITES = [
    dict(
        label   = "SWARM-A",
        parquet = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png = Path("Figure/2019/1D_ratio_msis_2019_SWARM-A_by_LT.png"),
    ),
    dict(
        label   = "SWARM-B",
        parquet = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png = Path("Figure/2019/1D_ratio_msis_2019_SWARM-B_by_LT.png"),
    ),
    dict(
        label   = "SWARM-C",
        parquet = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png = Path("Figure/2019/1D_ratio_msis_2019_SWARM-C_by_LT.png"),
    ),
]

# ============================================================
# 共通設定
# ============================================================
DOY_START, DOY_END = 252, 266   # Sep 9 - Sep 23

# 2019年：3衛星すべてが深夜側・昼間側
LT_SECTORS = [
    ("Nightside (LT 0-6h)",  0,  6,  "#1a6faf"),   # 濃い青
    ("Dayside  (LT 12-18h)", 12, 18, "#e07b39"),   # オレンジ
]

LAT_BANDS = [
    ("High  (40-60°)", 40.0, 60.0),
    ("Mid   (20-40°)", 20.0, 40.0),
    ("Low   ( 0-20°)",  0.0, 20.0),
]

DOY_REF1 = (252, 255)   # Sep 9-12
DOY_REF2 = (263, 266)   # Sep 20-23
DOY_SSW_START, DOY_SSW_END = 256, 262   # Sep 13-19

VALUE_COL = "density_ratio_msis"

# x軸 DOY -> 日付ラベル
DOY_DATE_MAP = {
    252:"9/9",  253:"9/10", 254:"9/11", 255:"9/12",
    256:"9/13", 257:"9/14", 258:"9/15", 259:"9/16",
    260:"9/17", 261:"9/18", 262:"9/19", 263:"9/20",
    264:"9/21", 265:"9/22", 266:"9/23",
}


# ============================================================
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


def plot_satellite(sat: dict) -> None:
    label   = sat["label"]
    parquet = sat["parquet"]
    out_png = sat["out_png"]

    print(f"\n=== {label} ===")
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL])
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]
    print(f"  {len(df):,} rows")

    n_rows = len(LAT_BANDS)
    n_cols = len(LT_SECTORS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey="row")
    fig.subplots_adjust(hspace=0.08, wspace=0.06)

    xtick_doys   = list(range(DOY_START, DOY_END + 1))
    xtick_labels = [DOY_DATE_MAP.get(d, str(d)) for d in xtick_doys]
    x_min, x_max = DOY_START - 0.5, DOY_END + 0.5

    for col_idx, (lt_label, lt_min, lt_max, lt_color) in enumerate(LT_SECTORS):
        df_lt = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]
        print(f"\n  {lt_label}: {len(df_lt):,} obs")

        for row_idx, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
            ax = axes[row_idx, col_idx]
            mask = (df_lt["lat"].abs() >= lat_lo) & (df_lt["lat"].abs() < lat_hi)
            sub  = df_lt[mask]
            daily = sub.groupby("DOY_int")[VALUE_COL].median()
            print(f"    {band_label}: {len(daily)} days")

            ref, residual = compute_residual(daily)

            # shading
            ax.axvspan(*DOY_REF1, color="lightblue", alpha=0.20, label="Non-SSW ref")
            ax.axvspan(*DOY_REF2, color="lightblue", alpha=0.20)
            ax.axvspan(DOY_SSW_START, DOY_SSW_END, color="lightyellow", alpha=0.40,
                       label="SSW period (est.)")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)

            # residual line
            ax.plot(residual.index, residual.values,
                    color=lt_color, linewidth=2.0, marker="o", markersize=5,
                    zorder=4, label=f"Δratio ({band_label.strip()})")

            # decoration
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(xtick_doys)
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3, linewidth=0.7)

            if col_idx == 0:
                ax.set_ylabel("Δratio (ratio − ref)", fontsize=10)

            ax.text(0.01, 0.97, band_label.strip(), transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="top", ha="left")

            if not np.isnan(ref):
                ax.text(0.99, 0.97, f"ref ratio = {ref:.3f}",
                        transform=ax.transAxes, fontsize=8,
                        va="top", ha="right", color="gray")

            if row_idx == 0:
                h1, l1 = ax.get_legend_handles_labels()
                ax.legend(h1, l1, fontsize=7, loc="upper right",
                          framealpha=0.85, ncol=2)
                ax.set_title(lt_label, fontsize=12, fontweight="bold", pad=6)

        axes[-1, col_idx].set_xlabel("Date (2019)", fontsize=11)

    fig.suptitle(
        f"{label}  density_ratio_msis residual  (2019 SH SSW: Sep 9–23)\n"
        "Residual = daily median(rho_obs/rho_MSIS) − median(non-SSW ref: Sep9-12 & Sep20-23)\n"
        "Left: Nightside (LT 0-6h)   |   Right: Dayside (LT 12-18h)",
        fontsize=12, fontweight="bold", y=1.01,
    )

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
