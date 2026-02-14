# doy_daily_density_plot_limited.py
# 目的：
# - 2018/02/05〜2018/02/21 の期間のみ抽出
# - 正規化密度（density_norm）の1日平均を計算
# - DOYを横軸にプロット（概形把握用）

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 入力
# =========================
PARQUET_PATH = Path("normalizeddata/swarm_dnsapod_2018_normalized.parquet")

# 表示したい期間（UTC基準）
START_DATE = "2018-02-05"
END_DATE   = "2018-02-21"

# =========================
# 読み込み
# =========================
df = pd.read_parquet(PARQUET_PATH)

# datetime を index にする
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.set_index("datetime").sort_index()

# =========================
# 期間抽出
# =========================
df_period = df.loc[START_DATE:END_DATE]

print("抽出後データ数:", len(df_period))
print("期間:", df_period.index.min(), "→", df_period.index.max())

# =========================
# 1日平均（正規化密度）
# =========================
daily_mean = df_period["density_norm"].resample("D").mean()

# DOY 作成
doy = daily_mean.index.dayofyear

# =========================
# プロット
# =========================
plt.figure(figsize=(8, 4))

plt.plot(doy, daily_mean.values, marker="o", markersize=4, linewidth=1.5)

plt.grid(alpha=0.3)
plt.xlabel("Day of Year (DOY)")
plt.ylabel("Daily mean normalized density [kg/m³]")

# 熱圏密度は通常 log が見やすい
plt.yscale("log")

plt.title("Swarm-A normalized density (2018/02/05–02/21)")
plt.tight_layout()
plt.show()
