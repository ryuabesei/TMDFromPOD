"""
1D_ratio_msis_NH_SH_comparison.py

Purpose:
    2018 NH SSW と 2019 SH SSW について、
    Northern Hemisphere (lat > 0) と Southern Hemisphere (lat < 0) を分けて
    density_ratio_msis の残差（delta ratio）を1Dプロットで比較する。

    - 2018 NH SSW: NH 高緯度で応答が大きいはず
    - 2019 SH SSW: SH 高緯度で応答が大きいはず

構成:
    3行（緯度帯: High 40-60° / Mid 20-40° / Low 0-20°） × 2列（2018 | 2019）
    各パネル: NH (青実線) vs SH (赤破線)

出力:
    Figure/comparison/1D_ratio_NH_SH_SWARM-{A,B,C}_comparison.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 衛星ごとの設定
# ============================================================
SATELLITES = [
    dict(
        label    = "SWARM-A",
        p2018    = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        p2019    = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png  = Path("Figure/comparison/1D_ratio_NH_SH_SWARM-A_comparison.png"),
    ),
    dict(
        label    = "SWARM-B",
        p2018    = Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        p2019    = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png  = Path("Figure/comparison/1D_ratio_NH_SH_SWARM-B_comparison.png"),
    ),
    dict(
        label    = "SWARM-C",
        p2018    = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        p2019    = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW.parquet"),
        out_png  = Path("Figure/comparison/1D_ratio_NH_SH_SWARM-C_comparison.png"),
    ),
]

# ============================================================
# 2018 設定
# ============================================================
DOY_START_2018, DOY_END_2018 = 30, 65
DOY_REF1_2018 = (30, 40)
DOY_REF2_2018 = (61, 65)
DOY_SSW_2018  = (41, 60)

# ============================================================
# 2019 設定
# ============================================================
DOY_START_2019, DOY_END_2019 = 252, 266
DOY_REF1_2019 = (252, 255)
DOY_REF2_2019 = (263, 266)
DOY_SSW_2019  = (256, 262)

# 2019 DOY -> 日付ラベル
DOY_DATE_2019 = {
    252:"9/9", 253:"9/10", 254:"9/11", 255:"9/12",
    256:"9/13", 257:"9/14", 258:"9/15", 259:"9/16",
    260:"9/17", 261:"9/18", 262:"9/19", 263:"9/20",
    264:"9/21", 265:"9/22", 266:"9/23",
}

VALUE_COL = "density_ratio_msis"

# 緯度帯（絶対値）
LAT_BANDS = [
    ("High  (40-60°)", 40.0, 60.0),
    ("Mid   (20-40°)", 20.0, 40.0),
    ("Low   ( 0-20°)",  0.0, 20.0),
]


# ============================================================
# データ読み込みと日次中央値の計算（NH / SH 分離）
# ============================================================
def load_daily_NH_SH(parquet: Path, doy_start: int, doy_end: int
                      ) -> dict[str, dict[str, pd.Series]]:
    """
    Returns:
        {band_label: {"NH": Series(DOY->median), "SH": Series(DOY->median)}}
    """
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", VALUE_COL])
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= doy_start) & (df["DOY_int"] <= doy_end)]

    result = {}
    for band_label, lat_lo, lat_hi in LAT_BANDS:
        mask_abs = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
        sub = df[mask_abs]
        nh = sub[sub["lat"] > 0].groupby("DOY_int")[VALUE_COL].median()
        sh = sub[sub["lat"] < 0].groupby("DOY_int")[VALUE_COL].median()
        result[band_label] = {"NH": nh, "SH": sh}

    return result


def compute_residual(daily: pd.Series, ref1: tuple, ref2: tuple) -> tuple[float, pd.Series]:
    doy = daily.index
    mask = (
        ((doy >= ref1[0]) & (doy <= ref1[1])) |
        ((doy >= ref2[0]) & (doy <= ref2[1]))
    )
    if mask.sum() == 0 or daily[mask].isna().all():
        return 0.0, daily * np.nan
    ref = float(daily[mask].median())
    return ref, daily - ref


# ============================================================
# 1つの衛星についてプロット
# ============================================================
def plot_satellite(sat: dict) -> None:
    label   = sat["label"]
    p2018   = sat["p2018"]
    p2019   = sat["p2019"]
    out_png = sat["out_png"]

    print(f"\n=== {label} ===")
    data2018 = load_daily_NH_SH(p2018, DOY_START_2018, DOY_END_2018)
    data2019 = load_daily_NH_SH(p2019, DOY_START_2019, DOY_END_2019)

    n_bands = len(LAT_BANDS)
    fig = plt.figure(figsize=(18, 3.8 * n_bands))
    fig.suptitle(
        f"{label} — NH vs SH Thermospheric Density Response\n"
        "2018 NH SSW (left)  |  2019 SH SSW (right)\n"
        "Solid: Northern Hemisphere  |  Dashed: Southern Hemisphere",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(n_bands, 2, figure=fig, hspace=0.15, wspace=0.12)

    for row, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):

        # ---- 2018 (左列) ----
        ax_l = fig.add_subplot(gs[row, 0])
        d18 = data2018[band_label]

        for hemi, ls, color, lw in [("NH", "-",  "#1f77b4", 2.2),
                                      ("SH", "--", "#d62728", 2.0)]:
            daily = d18[hemi]
            if len(daily) == 0:
                continue
            _, res = compute_residual(daily, DOY_REF1_2018, DOY_REF2_2018)
            ax_l.plot(res.index, res.values,
                      color=color, lw=lw, ls=ls, marker="o", ms=4,
                      label=f"{hemi}")

        ax_l.axvspan(*DOY_REF1_2018, color="lightblue", alpha=0.18, lw=0, label="Non-SSW ref")
        ax_l.axvspan(*DOY_REF2_2018, color="lightblue", alpha=0.18, lw=0)
        ax_l.axvspan(*DOY_SSW_2018,  color="lightyellow", alpha=0.35, lw=0, label="SSW period")
        ax_l.axhline(0, color="gray", lw=0.8, ls=":")
        ax_l.set_xlim(DOY_START_2018 - 0.5, DOY_END_2018 + 0.5)
        ax_l.set_xticks(range(DOY_START_2018, DOY_END_2018 + 1, 5))
        ax_l.grid(axis="y", alpha=0.3)
        ax_l.set_ylabel("Δratio (ratio − ref)", fontsize=10)
        ax_l.text(0.01, 0.97, band_label.strip(), transform=ax_l.transAxes,
                  fontsize=10, fontweight="bold", va="top")
        ax_l.legend(fontsize=8, loc="upper right", framealpha=0.85)

        if row == 0:
            ax_l.set_title("2018 NH SSW  (Jan 30 – Mar 6)", fontsize=11, fontweight="bold")
        if row == n_bands - 1:
            ax_l.set_xlabel("Day of Year (2018)", fontsize=10)

        # ---- 2019 (右列) ----
        ax_r = fig.add_subplot(gs[row, 1])
        d19 = data2019[band_label]

        for hemi, ls, color, lw in [("NH", "-",  "#1f77b4", 2.2),
                                      ("SH", "--", "#d62728", 2.0)]:
            daily = d19[hemi]
            if len(daily) == 0:
                continue
            _, res = compute_residual(daily, DOY_REF1_2019, DOY_REF2_2019)
            ax_r.plot(res.index, res.values,
                      color=color, lw=lw, ls=ls, marker="o", ms=4,
                      label=f"{hemi}")

        ax_r.axvspan(*DOY_REF1_2019, color="lightblue", alpha=0.18, lw=0, label="Non-SSW ref")
        ax_r.axvspan(*DOY_REF2_2019, color="lightblue", alpha=0.18, lw=0)
        ax_r.axvspan(*DOY_SSW_2019,  color="lightyellow", alpha=0.35, lw=0, label="SSW period")
        ax_r.axhline(0, color="gray", lw=0.8, ls=":")
        ax_r.set_xlim(DOY_START_2019 - 0.5, DOY_END_2019 + 0.5)
        xticks_2019 = list(range(DOY_START_2019, DOY_END_2019 + 1))
        ax_r.set_xticks(xticks_2019)
        ax_r.set_xticklabels([DOY_DATE_2019.get(d, "") for d in xticks_2019],
                              rotation=45, ha="right", fontsize=8)
        ax_r.grid(axis="y", alpha=0.3)
        ax_r.tick_params(axis="y", labelleft=False)
        ax_r.text(0.01, 0.97, band_label.strip(), transform=ax_r.transAxes,
                  fontsize=10, fontweight="bold", va="top")
        ax_r.legend(fontsize=8, loc="upper right", framealpha=0.85)

        if row == 0:
            ax_r.set_title("2019 SH SSW  (Sep 9 – Sep 23)", fontsize=11, fontweight="bold")
        if row == n_bands - 1:
            ax_r.set_xlabel("Date (2019)", fontsize=10)

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
    print("\nDone.")


if __name__ == "__main__":
    main()
