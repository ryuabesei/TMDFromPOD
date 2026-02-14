# kp_barplot_daily.py
# 目的：
# - 2018/02/05〜02/21 の日平均Kpを棒グラフ表示

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("data/Kpindex/SW-All.csv")
START_DATE = "2018-02-05"
END_DATE   = "2018-02-21"

# =========================
# 読み込み
# =========================
df = pd.read_csv(CSV_PATH)

# DATEをdatetime化
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.set_index("DATE").sort_index()

# 期間抽出
dfp = df.loc[START_DATE:END_DATE]

# Kp列
kp_cols = [f"KP{i}" for i in range(1, 9)]

# 日平均Kp
kp_daily_mean = dfp[kp_cols].mean(axis=1)

# DOY作成
doy = kp_daily_mean.index.dayofyear

# =========================
# 棒グラフ
# =========================
plt.figure(figsize=(9,4))

plt.bar(doy, kp_daily_mean.values)

plt.grid(axis="y", alpha=0.3)
plt.xlabel("Day of Year (DOY)")
plt.ylabel("Daily mean Kp")
plt.title("Daily Mean Kp (2018/02/05–02/21)")
plt.tight_layout()
plt.show()
