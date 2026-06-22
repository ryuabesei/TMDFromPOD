# ==============================================================================
# 2019_plot_kp_f107.py
# 目的: DOY 252〜266（2019年9月9日〜23日）の Kp index と F10.7 を1つのグラフに
#       重ねてプロットする。
#       左軸: F10.7 adjusted (折れ線グラフ)
#       右軸: 日平均 Kp (単一色 steelblue の棒グラフ)
# ==============================================================================

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =========================
# 設定
# =========================
KPINDEX_CSV = Path("data/SSW2019/Kpindex/SW-All_2019-09-09_to_2019-09-23.csv")
OUT_PNG     = Path("Figure/2019/Kp_F107_SSW2019.png")

DOY_START     = 252
DOY_END       = 266
DOY_REF_START = 252
DOY_REF_END   = 254

# =========================
# データ読み込み
# =========================
df = pd.read_csv(KPINDEX_CSV, parse_dates=["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)
df["DOY"] = df["DATE"].dt.dayofyear
df = df[(df["DOY"] >= DOY_START) & (df["DOY"] <= DOY_END)].copy()
print(f"Loaded {len(df)} daily rows (DOY {df['DOY'].min()}–{df['DOY'].max()})")

# 日平均Kp（KP_SUM は各3時間Kp×10 の合計）
df["KP_daily_mean"] = df["KP_SUM"] / 80.0

# =========================
# プロット（1パネル）
# =========================
fig, ax_f107 = plt.subplots(figsize=(13, 5))

# --- シェーディング (基準期間 DOY 252〜254) ---
ax_f107.axvspan(DOY_REF_START, DOY_REF_END,
                color="#6BAED6", alpha=0.18, lw=0)

# --- 右軸（Kp） ---
ax_kp = ax_f107.twinx()

# 日平均Kp 棒グラフ
ax_kp.bar(df["DOY"], df["KP_daily_mean"],
          width=0.80, color="steelblue", alpha=0.55,
          edgecolor="none", zorder=2)

ax_kp.set_ylabel("Kp index", fontsize=12, color="steelblue")
ax_kp.tick_params(axis="y", labelcolor="steelblue")
ax_kp.set_ylim(0, max(df["KP_daily_mean"].max() * 2.0, 8))
ax_kp.spines["right"].set_color("steelblue")
ax_kp.grid(False)  # グリッド完全オフ

# --- 左軸（F10.7） ---
ax_f107.plot(df["DOY"], df["F10.7_ADJ"],
             color="#B2182B", lw=2.2, marker="o",
             markersize=6, zorder=6, label="F10.7 (adjusted)")

f107_vals = df["F10.7_ADJ"].dropna()
f107_min  = f107_vals.min()
f107_max  = f107_vals.max()
ax_f107.set_ylim(f107_min - 4, f107_max + 8)
ax_f107.set_ylabel("F10.7 [sfu]", fontsize=12, color="#B2182B")
ax_f107.tick_params(axis="y", labelcolor="#B2182B")
ax_f107.spines["left"].set_color("#B2182B")

# --- X軸 ---
ax_f107.set_xlabel("Day of Year 2019", fontsize=12)
ax_f107.set_xlim(DOY_START - 0.5, DOY_END + 0.5)
ax_f107.set_xticks(range(DOY_START, DOY_END + 1, 2))
ax_f107.grid(False)  # グリッド完全オフ

# 日付ラベルを X 軸下に追記
date_labels = {
    252: "Sep 9",  254: "Sep 11", 256: "Sep 13",
    258: "Sep 15", 260: "Sep 17", 262: "Sep 19",
    264: "Sep 21", 266: "Sep 23",
}
for doy_val, label in date_labels.items():
    if DOY_START <= doy_val <= DOY_END:
        ax_f107.text(doy_val, f107_min - 3.8, label,
                     ha="center", va="top", fontsize=8, color="gray")

# --- タイトル ---
ax_f107.set_title(
    "F10.7 & Kp Index — DOY 252–266, 2019 (SSW period)\n"
    "Shading: baseline reference (blue, DOY 252–254)",
    fontsize=12, fontweight="bold"
)

# --- 凡例（統合） ---
handles = [
    mpatches.Patch(color="#6BAED6", alpha=0.55, label="Baseline reference (DOY 252–254)"),
    plt.Line2D([], [], color="#B2182B", lw=2.2, marker="o",
               markersize=6, label="F10.7 adjusted"),
    mpatches.Patch(color="steelblue", alpha=0.55, label="Daily mean Kp"),
]
ax_f107.legend(handles=handles, fontsize=9.5, loc="upper left",
               framealpha=0.90, ncol=1)

plt.tight_layout()

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ 保存完了: {OUT_PNG}")
plt.close(fig)
