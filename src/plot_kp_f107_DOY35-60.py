"""
plot_kp_f107_DOY35-60.py

目的:
    DOY35〜60（2018年）の Kp index と F10.7 を1つのグラフに重ねてプロットする。
    左軸: F10.7（折れ線）
    右軸: 日平均Kp（棒グラフ）+ 3時間Kp（散布点）

出力:
    Figure/2018/Kp_F107_DOY35-60.png
"""

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
KPINDEX_CSV = Path("data/Kpindex/SW-20180120_20180320.csv")
OUT_PNG     = Path("Figure/2018/Kp_F107_DOY35-60.png")

DOY_START     = 35
DOY_END       = 60
DOY_REF_START = 35
DOY_REF_END   = 40
DOY_SSW_START = 41
DOY_SSW_END   = 60
SSW_ONSET_DOY = 40.5

# =========================
# データ読み込み
# =========================
df = pd.read_csv(KPINDEX_CSV, parse_dates=["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)
df["DOY"] = df["DATE"].dt.dayofyear
df = df[(df["DOY"] >= DOY_START) & (df["DOY"] <= DOY_END)].copy()
print(f"Loaded {len(df)} daily rows (DOY {df['DOY'].min()}–{df['DOY'].max()})")

kp_cols = [f"KP{i}" for i in range(1, 9)]

# 日平均Kp（KP_SUM は各3時間Kp×10 の合計）
df["KP_daily_mean"] = df["KP_SUM"] / 80.0

# # 各3時間Kpを長形式に
# kp_long_rows = []
# for _, row in df.iterrows():
#     for i, col in enumerate(kp_cols):
#         kp_long_rows.append({
#             "DOY_frac": row["DOY"] + (i * 3 + 1.5) / 24.0,
#             "Kp":        row[col] / 10.0,
#         })
# df_kp3h = pd.DataFrame(kp_long_rows)

# =========================
# プロット（1パネル）
# =========================
fig, ax_f107 = plt.subplots(figsize=(13, 5))

# --- シェーディング ---
ax_f107.axvspan(DOY_REF_START, DOY_REF_END,
                color="#6BAED6", alpha=0.18, lw=0)
ax_f107.axvspan(DOY_SSW_START, DOY_SSW_END,
                color="#FD8D3C", alpha=0.15, lw=0)
ax_f107.axvline(SSW_ONSET_DOY, color="red", lw=1.8,
                linestyle="--", alpha=0.75, zorder=5)

# --- 右軸（Kp） ---
ax_kp = ax_f107.twinx()

# 日平均Kp 棒グラフ
ax_kp.bar(df["DOY"], df["KP_daily_mean"],
          width=0.80, color="steelblue", alpha=0.55,
          edgecolor="none", zorder=2)

# # 3時間Kp 散布点
# ax_kp.scatter(df_kp3h["DOY_frac"], df_kp3h["Kp"],
#               s=14, color="steelblue", alpha=0.55,
#               zorder=3, edgecolors="none")


ax_kp.set_ylabel("Kp index", fontsize=12, color="steelblue")
ax_kp.tick_params(axis="y", labelcolor="steelblue")
ax_kp.set_ylim(0, max(df["KP_daily_mean"].max() * 2.0, 8))
ax_kp.spines["right"].set_color("steelblue")
ax_kp.grid(False)  # twinx側のグリッドを完全にオフ


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
ax_f107.set_xlabel("Day of Year 2018", fontsize=12)
ax_f107.set_xlim(DOY_START - 0.5, DOY_END + 0.5)
ax_f107.set_xticks(range(DOY_START, DOY_END + 1, 5))
ax_f107.grid(False)  # グリッド完全オフ

# 日付ラベルを X 軸下に追記
date_labels = {
    35: "Feb 4",  40: "Feb 9",  45: "Feb 14",
    50: "Feb 19", 55: "Feb 24", 60: "Mar 1",
}
for doy_val, label in date_labels.items():
    if DOY_START <= doy_val <= DOY_END:
        ax_f107.text(doy_val, f107_min - 3.8, label,
                     ha="center", va="top", fontsize=8, color="gray")

# --- タイトル ---
ax_f107.set_title(
    "F10.7 & Kp Index — DOY 35–60, 2018\n"
    "Shading: non-SSW ref (blue, DOY35–40) / SSW period (orange, DOY41–60)",
    fontsize=12, fontweight="bold"
)

# --- 凡例（統合） ---
handles = [
    mpatches.Patch(color="#6BAED6", alpha=0.55, label="Non-SSW ref (DOY35–40)"),
    mpatches.Patch(color="#FD8D3C", alpha=0.55, label="SSW period (DOY41–60)"),
    plt.Line2D([], [], color="red", lw=1.8, linestyle="--", label="SSW onset (DOY40.5)"),
    plt.Line2D([], [], color="#B2182B", lw=2.2, marker="o",
               markersize=6, label="F10.7 adjusted"),
    mpatches.Patch(color="steelblue", alpha=0.55, label="Daily mean Kp"),
    # plt.Line2D([], [], marker="o", color="steelblue", markersize=5,
    #            linewidth=0, alpha=0.65, label="3-hr Kp"),
]
ax_f107.legend(handles=handles, fontsize=8.5, loc="upper left",
               framealpha=0.90, ncol=2)

plt.tight_layout()

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ 保存完了: {OUT_PNG}")
