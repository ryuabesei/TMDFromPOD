# kp_f107_barplot_daily.py
# 目的：
# - 2018/02/05〜02/21 の日平均Kp（棒グラフ）とF10.7（折れ線）を同時表示

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

CSV_PATH  = Path("data/Kpindex/SW-All.csv")
F107_PATH = Path("data/F10.7/F10_F30_obs_abs.txt")
START_DATE = "2018-02-05"
END_DATE   = "2018-02-21"

# =========================
# Kp 読み込み
# =========================
df = pd.read_csv(CSV_PATH)
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.set_index("DATE").sort_index()

dfp = df.loc[START_DATE:END_DATE]
kp_cols = [f"KP{i}" for i in range(1, 9)]
kp_daily_mean = dfp[kp_cols].mean(axis=1)
doy = kp_daily_mean.index.dayofyear

# =========================
# F10.7 読み込み
# =========================
# ヘッダー行（#で始まる行）をスキップし、空白区切りで読み込む
f107_df = pd.read_csv(
    F107_PATH,
    comment="#",
    sep=r"\s+",
    header=None,
    names=[
        "year", "month", "day", "cont_day",
        "f30",   "f30_c",   "f30_p",   "f30_f",
        "f15",   "f15_c",   "f15_p",   "f15_f",
        "f107",  "f107_c",  "f107_p",  "f107_f",
        "f8",    "f8_c",    "f8_p",    "f8_f",
        "f32",   "f32_c",   "f32_p",   "f32_f",
    ],
)

f107_df["DATE"] = pd.to_datetime(
    f107_df[["year", "month", "day"]].rename(
        columns={"year": "year", "month": "month", "day": "day"}
    )
)
f107_df = f107_df.set_index("DATE").sort_index()

# 欠損値（-1）をNaNに変換
f107_df["f107"] = f107_df["f107"].replace(-1, np.nan)

# 期間抽出
f107p = f107_df.loc[START_DATE:END_DATE, "f107"]
f107_doy = f107p.index.dayofyear

# =========================
# 1つのグラフに重ねてプロット（左軸: Kp、右軸: F10.7）
# =========================
fig, ax1 = plt.subplots(figsize=(10, 5))

# --- 左軸: Kp 棒グラフ ---
bars = ax1.bar(doy, kp_daily_mean.values, color="steelblue", alpha=0.7,
               edgecolor="white", linewidth=0.5, label="Daily mean Kp")
ax1.set_xlabel("Day of Year (DOY)")
ax1.set_ylabel("Daily mean Kp", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_ylim(0, max(kp_daily_mean.max() * 1.3, 5))
ax1.set_xlim(doy.min() - 0.5, doy.max() + 0.5)
ax1.grid(axis="y", alpha=0.25, color="steelblue")

# --- 右軸: F10.7 折れ線 ---
ax2 = ax1.twinx()
ax2.plot(f107_doy, f107p.values, color="tomato", marker="o", markersize=5,
         linewidth=1.8, label="F10.7 (observed)")
ax2.set_ylabel("F10.7 [sfu]", color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")
ax2.set_ylim(f107p.min() * 0.97, f107p.max() * 1.03)

# --- 凡例を統合 ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

plt.title("Daily Mean Kp & F10.7 (2018/02/05–02/21)", fontsize=12)
plt.tight_layout()

# --- 保存 ---
OUT_DIR = Path("Figure")
OUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUT_DIR / "KpindexandF10.7.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'KpindexandF10.7.png'}")

plt.show()
