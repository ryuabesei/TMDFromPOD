"""
1D_ratio_msis_2018_by_LT_nolat_all.py

Purpose:
    Plot daily-median rho_ratio (= rho_obs / rho_MSIS_real) for SWARM-A, B, C,
    split by LT sector only — NO latitude band separation (|lat| 0-60° all combined).

    SWARM-A/C: Morning (LT 6-12h) / Evening (LT 18-24h)
    SWARM-B  : Nightside (LT 0-6h) / Dayside (LT 12-18h)

    COSMIC T(10 hPa, 60-90 N) overlaid on right axis.

Output:
    Figure/2018/1D_ratio_only_msis_2018_SWARM-A_DOY30-65_by_LT_nolat.png
    Figure/2018/1D_ratio_only_msis_2018_SWARM-B_DOY30-65_by_LT_nolat.png
    Figure/2018/1D_ratio_only_msis_2018_SWARM-C_DOY30-65_by_LT_nolat.png
"""

from __future__ import annotations
from pathlib import Path
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 衛星ごとの設定
# ============================================================
SATELLITES = [
    dict(
        label      = "SWARM-A",
        parquet    = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        out_png    = Path("Figure/2018/1D_ratio_only_msis_2018_SWARM-A_DOY30-65_by_LT_nolat.png"),
        sectors    = [
            ("Morning (LT 6-12h)",  6,  12, "#1f77b4"),
            ("Evening (LT 18-24h)", 18, 24, "#d62728"),
        ],
    ),
    dict(
        label      = "SWARM-B",
        parquet    = Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        out_png    = Path("Figure/2018/1D_ratio_only_msis_2018_SWARM-B_DOY30-65_by_LT_nolat.png"),
        sectors    = [
            ("Nightside (LT 0-6h)",  0,  6,  "#1f77b4"),
            ("Dayside  (LT 12-18h)", 12, 18, "#e07b00"),
        ],
    ),
    dict(
        label      = "SWARM-C",
        parquet    = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        out_png    = Path("Figure/2018/1D_ratio_only_msis_2018_SWARM-C_DOY30-65_by_LT_nolat.png"),
        sectors    = [
            ("Morning (LT 6-12h)",  6,  12, "#1f77b4"),
            ("Evening (LT 18-24h)", 18, 24, "#d62728"),
        ],
    ),
]

COSMIC_CSV = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")

DOY_START, DOY_END     = 30, 65
LAT_ABS_MAX            = 60.0
DOY_REF1               = (30, 40)
DOY_REF2               = (61, 65)
DOY_SSW_START, DOY_SSW_END = 41, 60
VALUE_COL              = "density_ratio_msis"
COSMIC_LAT_LABEL       = "60–90°N"


# ============================================================
# Utilities
# ============================================================
def load_cosmic_T10(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        print(f"  [INFO] COSMIC CSV not found: {csv_path}")
        return pd.Series(dtype=float)
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "DOY", "T10_K"])
    df = df[(df["DOY"] >= DOY_START) & (df["DOY"] <= DOY_END)]
    return df.set_index(df["DOY"].astype(int))["T10_K"]


def plot_satellite(sat: dict, cosmic_T10: pd.Series) -> None:
    label   = sat["label"]
    parquet = sat["parquet"]
    out_png = sat["out_png"]
    sectors = sat["sectors"]

    print(f"\n=== {label} ===")
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL])
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]
    df = df[df["lat"].abs() <= LAT_ABS_MAX].copy()
    print(f"  {len(df):,} rows (|lat| 0–{LAT_ABS_MAX:.0f}°)")

    # ── レイアウト: 1行 × 2列 ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    # タイトルと上端の余白を確保
    fig.subplots_adjust(top=0.82, hspace=0.0, wspace=0.06)

    x_min, x_max = DOY_START - 0.5, DOY_END + 0.5

    for col_idx, (lt_label, lt_min, lt_max, lt_color) in enumerate(sectors):
        ax = axes[col_idx]

        df_lt = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]
        print(f"  {lt_label}: {len(df_lt):,} obs")

        daily = df_lt.groupby("DOY_int")[VALUE_COL].median()
        print(f"    -> {len(daily)} days with data")

        # シェーディング
        ax.axvspan(*DOY_REF1, color="lightblue",  alpha=0.25, lw=0)
        ax.axvspan(*DOY_REF2, color="lightblue",  alpha=0.25, lw=0)
        ax.axvspan(DOY_SSW_START, DOY_SSW_END,
                   color="lightyellow", alpha=0.50, lw=0)

        # rho_ratio
        ax.plot(daily.index, daily.values,
                color=lt_color, linewidth=2.0,
                marker="o", markersize=4.5, zorder=4,
                label=r"$\rho_{obs}/\rho_{MSIS}$  (|lat| 0–60°)")

        # COSMIC T(10 hPa) 右軸
        if len(cosmic_T10) > 0:
            ax2 = ax.twinx()
            ax2.plot(cosmic_T10.index.to_numpy(dtype=float), cosmic_T10.values,
                     color="hotpink", linewidth=1.8, linestyle="-",
                     marker="s", markersize=3.5, alpha=0.90, zorder=3,
                     label=f"COSMIC T (10 hPa, {COSMIC_LAT_LABEL})")
            T_vals   = cosmic_T10.values
            T_margin = max((T_vals.max() - T_vals.min()) * 0.3, 5.0)
            ax2.set_ylim(T_vals.min() - T_margin, T_vals.max() + T_margin)
            ax2.tick_params(axis="y", labelcolor="hotpink")
            if col_idx == len(sectors) - 1:
                ax2.set_ylabel("T (10 hPa) [K]", fontsize=10, color="hotpink")
            else:
                ax2.set_yticklabels([])
            h2, l2 = ax2.get_legend_handles_labels()
        else:
            h2, l2 = [], []

        # 装飾
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(range(DOY_START, DOY_END + 1, 5))
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)

        if col_idx == 0:
            ax.set_ylabel(r"$\rho_{ratio}$  ($\rho_{obs}$ / $\rho_{MSIS}$)",
                          fontsize=11)
        ax.set_xlabel("Day of Year (2018)", fontsize=11)

        # パネルタイトル（色付き）
        ax.set_title(lt_label, fontsize=13, fontweight="bold",
                     pad=8, color=lt_color)

        # 凡例（各パネル内）
        h1, l1 = ax.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2,
                  fontsize=8, loc="upper right", framealpha=0.85, ncol=1)

    # 図全体タイトル（余白 top=0.82 に合わせて y 調整）
    fig.suptitle(
        f"{label}  density_ratio_msis  (DOY {DOY_START}–{DOY_END}, 2018)\n"
        r"$\rho_{ratio}$ = daily median($\rho_{obs}$ / $\rho_{MSIS}$),  "
        "|lat| 0–60°  (all latitudes combined)",
        fontsize=12, fontweight="bold",
        x=0.5, y=0.98, va="top",
    )

    # 図下の共通凡例
    legend_elems = [
        Patch(facecolor="lightblue",   alpha=0.50,
              label=f"Non-SSW ref  (DOY {DOY_REF1[0]}–{DOY_REF1[1]} & {DOY_REF2[0]}–{DOY_REF2[1]})"),
        Patch(facecolor="lightyellow", alpha=0.70,
              label=f"SSW period  (DOY {DOY_SSW_START}–{DOY_SSW_END})"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.04))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    cosmic_T10 = load_cosmic_T10(COSMIC_CSV)
    for sat in SATELLITES:
        plot_satellite(sat, cosmic_T10)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
