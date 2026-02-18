# plot_density_and_cosmic.py
# 目的:
# - すでに保存済みの COSMIC 日平均T10hPa CSV を読む
# - Swarm 正規化密度と重ねて表示
# - 重い処理は一切しない

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 入力
# =========================
SWARM_PARQUET = Path("normalizeddata/swarm_dnscpod_2018_normalized.parquet")
COSMIC_CSV = Path("cosmic_T10hPa_daily_2018_DOY032_050_lat60_90N.csv")

START_DATE = "2018-02-05"
END_DATE   = "2018-02-21"

# =========================
# Swarm density
# =========================
df_sw = pd.read_parquet(SWARM_PARQUET)
df_sw["datetime"] = pd.to_datetime(df_sw["datetime"], utc=True)
df_sw = df_sw.set_index("datetime").sort_index()

df_sw_period = df_sw.loc[START_DATE:END_DATE]
daily_density = df_sw_period["density_norm"].resample("D").mean()

# =========================
# COSMIC T10hPa
# =========================
df_cos = pd.read_csv(COSMIC_CSV, parse_dates=["datetime"])
df_cos["datetime"] = pd.to_datetime(df_cos["datetime"], utc=True)
df_cos = df_cos.set_index("datetime").sort_index()

df_cos_period = df_cos.loc[START_DATE:END_DATE]
T10 = df_cos_period["T10_K"]

# =========================
# プロット
# =========================
fig, ax1 = plt.subplots(figsize=(8,4))

# 左軸：密度
ax1.plot(daily_density.index.dayofyear,
         daily_density.values,
         marker="o",
         linewidth=1.5)

ax1.set_yscale("log")
ax1.set_xlabel("Day of Year (DOY)")
ax1.set_ylabel("Daily mean normalized density [kg/m³]")
ax1.grid(alpha=0.3)

# 右軸：T10hPa
ax2 = ax1.twinx()
ax2.plot(T10.index.dayofyear,
         T10.values,
         linewidth=2,
         color="hotpink")

ax2.set_ylabel("T (10 hPa) [K]")

plt.title("Swarm-C Density + COSMIC T(10 hPa)")
plt.tight_layout()
plt.show()
