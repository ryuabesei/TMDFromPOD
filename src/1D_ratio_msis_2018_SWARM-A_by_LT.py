"""
1D_ratio_msis_2018_SWARM-A_by_LT.py

Purpose:
    Plot daily-median density_ratio_msis (= rho_obs / rho_MSIS_real) as a 1D
    time series for DOY 30-65, 2018, split by LT sector.

    SWARM-A observes two LT sectors during DOY 30-65:
        Morning: LT  6 - 12 h  (centred ~8-9 LT)
        Evening: LT 18 - 24 h  (centred ~19-20 LT)

    For each LT sector, results are shown for three latitude bands:
        High:   |lat| 40-60 deg
        Mid:    |lat| 20-40 deg
        Low:    |lat|  0-20 deg

    Residual = daily_ratio - median(non-SSW reference: DOY 30-40 & 61-65)

    COSMIC T(10 hPa) is overlaid for comparison.

Output:
    Figure/2018/1D_ratio_msis_2018_SWARM-A_DOY30-65_by_LT.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Settings
# ============================================================
PARQUET   = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
COSMIC_CSV = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
OUT_PNG   = Path("Figure/2018/1D_ratio_msis_2018_SWARM-A_DOY30-65_by_LT.png")

DOY_START, DOY_END = 30, 65

# LT sectors for SWARM-A (DOY 30-65)
LT_SECTORS = [
    ("Morning (LT 6-12h)",  6,  12, "#1f77b4"),   # 青系
    ("Evening (LT 18-24h)", 18, 24, "#d62728"),   # 赤系
]

LAT_BANDS = [
    ("High  (40-60°)", 40.0, 60.0),
    ("Mid   (20-40°)", 20.0, 40.0),
    ("Low   ( 0-20°)",  0.0, 20.0),
]

# Non-SSW reference
DOY_REF1 = (30, 40)
DOY_REF2 = (61, 65)

# SSW period
DOY_SSW_START, DOY_SSW_END = 41, 60

VALUE_COL = "density_ratio_msis"

COSMIC_LAT_LABEL = "60–90°N"


# ============================================================
# Load COSMIC T(10 hPa)
# ============================================================
def load_cosmic_T10(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        print(f"  [INFO] COSMIC CSV not found: {csv_path} — skipping overlay")
        return pd.Series(dtype=float)
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "DOY", "T10_K"])
    df = df[(df["DOY"] >= DOY_START) & (df["DOY"] <= DOY_END)]
    s = df.set_index(df["DOY"].astype(int))["T10_K"]
    print(f"    COSMIC: {len(s)} days (DOY {s.index.min()}-{s.index.max()})")
    return s


# ============================================================
# Compute residual
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


# ============================================================
# Main plot
# ============================================================
def main() -> None:
    print("Loading parquet ...")
    df = pd.read_parquet(PARQUET)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL])
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]
    print(f"  {len(df):,} rows after DOY filter")

    print("Loading COSMIC T(10 hPa) ...")
    cosmic_T10 = load_cosmic_T10(COSMIC_CSV)

    n_rows = len(LAT_BANDS)
    n_cols = len(LT_SECTORS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 3.5 * n_rows),
        sharex=True, sharey="row"
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.06)

    x_min, x_max = DOY_START - 0.5, DOY_END + 0.5

    for col_idx, (lt_label, lt_min, lt_max, lt_color) in enumerate(LT_SECTORS):
        df_lt = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]
        n_lt = len(df_lt)
        print(f"\n  {lt_label}: {n_lt:,} obs")

        for row_idx, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
            ax = axes[row_idx, col_idx]

            mask = (df_lt["lat"].abs() >= lat_lo) & (df_lt["lat"].abs() < lat_hi)
            sub = df_lt[mask]
            daily = sub.groupby("DOY_int")[VALUE_COL].median()
            print(f"    {band_label}: {len(daily)} days")

            ref, residual = compute_residual(daily)

            # --- Shading ---
            ax.axvspan(*DOY_REF1, color="lightblue", alpha=0.20, label="Non-SSW ref")
            ax.axvspan(*DOY_REF2, color="lightblue", alpha=0.20)
            ax.axvspan(DOY_SSW_START, DOY_SSW_END, color="lightyellow", alpha=0.40, label="SSW period")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)

            # --- Residual line ---
            ax.plot(residual.index, residual.values,
                    color=lt_color, linewidth=2.0, marker="o", markersize=4,
                    zorder=4, label=f"Δratio ({band_label.strip()})")

            # --- COSMIC T(10 hPa) on right axis ---
            if len(cosmic_T10) > 0:
                ax2 = ax.twinx()
                ax2.plot(cosmic_T10.index.to_numpy(dtype=float),
                         cosmic_T10.values,
                         color="hotpink", linewidth=1.6, linestyle="-",
                         marker="s", markersize=3, alpha=0.85,
                         label=f"COSMIC T (10hPa, {COSMIC_LAT_LABEL})", zorder=3)
                T_vals = cosmic_T10.values
                T_margin = max((T_vals.max() - T_vals.min()) * 0.3, 5.0)
                ax2.set_ylim(T_vals.min() - T_margin, T_vals.max() + T_margin)
                ax2.tick_params(axis="y", labelcolor="hotpink")
                # 右軸ラベルは各行の右端カラムのみ
                if col_idx == n_cols - 1:
                    ax2.set_ylabel("T (10 hPa) [K]", fontsize=9, color="hotpink")
                else:
                    ax2.set_yticklabels([])
                h2, l2 = ax2.get_legend_handles_labels()
            else:
                h2, l2 = [], []

            # --- Decoration ---
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(range(DOY_START, DOY_END + 1, 5))
            ax.grid(axis="y", alpha=0.3, linewidth=0.7)

            # 左軸ラベルは左端カラムのみ
            if col_idx == 0:
                ax.set_ylabel("Δratio (ratio − ref)", fontsize=10)

            # 行ラベル（左端）
            ax.text(0.01, 0.97, band_label.strip(),
                    transform=ax.transAxes, fontsize=10, fontweight="bold",
                    va="top", ha="left")

            # ref値の注記
            if not np.isnan(ref):
                ax.text(0.99, 0.97, f"ref ratio = {ref:.3f}",
                        transform=ax.transAxes, fontsize=8,
                        va="top", ha="right", color="gray")

            # 凡例（1行目のみ）
            if row_idx == 0:
                h1, l1 = ax.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right",
                          framealpha=0.85, ncol=2)
                ax.set_title(lt_label, fontsize=12, fontweight="bold", pad=6)

        # x軸ラベルは最下行
        axes[-1, col_idx].set_xlabel("Day of Year (2018)", fontsize=11)

    fig.suptitle(
        "SWARM-A  density_ratio_msis residual  (DOY 30-65, 2018)\n"
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
                      resid.to_numpy(dtype=float),
                      color=color, linewidth=2.0, marker="o", markersize=4, zorder=4)
            _add_cosmic(ax_d, cosmic_T10, show_ylabel=show_cosmic_ylabel)

            if col_idx == 0:
                ax_d.set_ylabel(f"{band_label.strip()}\nΔratio\n(ratio−ref)", fontsize=8)

            if not np.isnan(ref):
                ax_d.text(0.99, 0.97, f"ref={ref:.3f}",
                          transform=ax_d.transAxes, fontsize=7,
                          va="top", ha="right", color="gray")

            ax_d.text(0.01, 0.97, band_label.strip(),
                      transform=ax_d.transAxes, fontsize=9,
                      fontweight="bold", va="top", ha="left")

            if bi == n_bands - 1:
                ax_d.set_xlabel("Day of Year (2018)", fontsize=10)

    # 凡例
    lt_names = "   |   ".join(lt["label"] for lt in lt_sectors)
    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, fc="lightblue",  alpha=0.4,
                       label="Non-SSW ref (DOY 30-40, 61-65)"),
        plt.Rectangle((0, 0), 1, 1, fc="lightyellow", alpha=0.6,
                       label="SSW period (DOY 41-60)"),
        plt.Line2D([0], [0], color="hotpink", lw=1.5, marker="s", ms=3,
                   label=f"COSMIC T (10 hPa, {COSMIC_LAT_LABEL})"),
    ]
    fig.legend(handles=legend_elems,
               loc="lower center", ncol=3, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"{label}  2018 NH SSW  (DOY 30–65)\n"
        f"{lt_names}\n"
        "Top rows: ρ_ratio = daily median(ρ_obs/ρ_MSIS_real)   "
        "Bottom rows: Δratio = ratio − median(non-SSW ref)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("Loading COSMIC T(10 hPa) ...")
    cosmic_T10 = load_cosmic_T10(COSMIC_CSV)
    for sat in SATELLITES:
        plot_satellite(sat, cosmic_T10)
    print("\nDone.")


if __name__ == "__main__":
    main()
