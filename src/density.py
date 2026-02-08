import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 入力データ
# =========================
PARQUET = "normalizeddata/swarm_dnscpod_2018_normalized.parquet"

# -------------------------
# データ読み込み
# -------------------------
df = pd.read_parquet(PARQUET)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

# -------------------------
# 期間抽出（2018/02/05 ～ 2018/02/20）
# -------------------------
t_start = pd.Timestamp("2018-02-05 00:00:00", tz="UTC")
t_end   = pd.Timestamp("2018-02-20 23:59:59", tz="UTC")

df = df[(df["datetime"] >= t_start) & (df["datetime"] <= t_end)].copy()

# -------------------------
# Day of Year（連続値）
# -------------------------
dt = df["datetime"]
df["DOY"] = (
    dt.dt.dayofyear
    + dt.dt.hour / 24.0
    + dt.dt.minute / 1440.0
    + dt.dt.second / 86400.0
)

# =========================
# 1日平均を計算
# =========================
# 日付（UTC日）でグループ化
daily = (
    df
    .groupby(df["datetime"].dt.floor("D"))
    .agg(
        density_mean=("density_norm", "mean"),
        density_median=("density_norm", "median")
    )
)

# 日平均用の DOY（その日の中央 = +0.5）
daily["DOY"] = daily.index.dayofyear + 0.5

# =========================
# プロット
# =========================
plt.figure(figsize=(12, 4))

# --- 元データ（30秒サンプル） ---
plt.plot(
    df["DOY"],
    df["density_norm"],
    lw=0.6,
    alpha=0.3,
    label="30-s samples"
)

# --- 1日平均（太線） ---
plt.plot(
    daily["DOY"],
    daily["density_mean"],
    marker="o",
    lw=2.5,
    label="Daily mean"
)

plt.yscale("log")
plt.xlabel("Day of Year (2018)")
plt.ylabel("Normalized density [kg m$^{-3}$]")
plt.title("Swarm-C normalized thermospheric density\n2018-02-05 to 2018-02-20")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
