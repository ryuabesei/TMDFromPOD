import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# データ読み込み
# =========================
df = pd.read_parquet("normalizeddata/swarm_dnscpod_2018_normalized.parquet")
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

# 期間（先行研究と同じDOY範囲に近い例）
df = df[(df["datetime"] >= "2018-02-05") & (df["datetime"] <= "2018-02-20")].copy()

# =========================
# DOY（連続値）
# =========================
dt = df["datetime"]
df["DOY"] = (
    dt.dt.dayofyear
    + dt.dt.hour / 24
    + dt.dt.minute / 1440
    + dt.dt.second / 86400
)

# =========================
# LTセクター抽出（Afternoon）
# =========================
df_lt = df[(df["lst_h"] >= 17) & (df["lst_h"] < 21)].copy()

print("N points (17–21 LT):", len(df_lt))

# =========================
# グリッド作成
# =========================
doy_bins = np.arange(36, 51, 0.5)      # DOY: Feb 5–20
lat_bins = np.arange(-70, 70, 3)       # 緯度

# 2D binning（median）
Z = np.full((len(lat_bins)-1, len(doy_bins)-1), np.nan)

for i in range(len(doy_bins)-1):
    for j in range(len(lat_bins)-1):
        mask = (
            (df_lt["DOY"] >= doy_bins[i]) &
            (df_lt["DOY"] <  doy_bins[i+1]) &
            (df_lt["lat"] >= lat_bins[j]) &
            (df_lt["lat"] <  lat_bins[j+1])
        )
        if mask.any():
            Z[j, i] = np.median(df_lt.loc[mask, "density_norm"])

# =========================
# プロット
# =========================
plt.figure(figsize=(6, 3))

X, Y = np.meshgrid(doy_bins[:-1], lat_bins[:-1])

pcm = plt.contourf(
    X, Y, Z,
    levels=20,
    cmap="turbo"
)

plt.colorbar(pcm, label="Normalized density [kg m$^{-3}$]")
plt.xlabel("Day of Year (2018)")
plt.ylabel("Geographic Latitude")
plt.title("Swarm-C normalized density (17–21 LT)")

plt.tight_layout()
plt.show()
