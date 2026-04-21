# doy_daily_density_plot_with_msis.py

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymsis import msis

# =========================
# 入力
# =========================
PARQUET_PATH = Path("normalizeddata/swarm_dnsapod_2018_normalized.parquet")

START_DATE = "2018-02-05"
END_DATE   = "2018-02-21"

# =========================
# 読み込み
# =========================
df = pd.read_parquet(PARQUET_PATH)

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.set_index("datetime").sort_index()

# =========================
# 期間抽出
# =========================
df_period = df.loc[START_DATE:END_DATE]

print("抽出後データ数:", len(df_period))
print("期間:", df_period.index.min(), "→", df_period.index.max())

# =========================
# MSIS density 計算
# =========================

times = df_period.index.to_pydatetime()
lats  = df_period["lat"].values
lons  = df_period["lon"].values
alts  = df_period["norm_ref_alt_km"].values

N = len(df_period)

f107  = np.full(N, 70.0)
f107a = np.full(N, 70.0)
ap    = np.full((N,7), 4.0)

print("Running MSIS...")

result = msis.run(
    times,
    lons,
    lats,
    alts,
    f107,
    f107a,
    ap
)

df_period["rho_msis"] = result[:,0]

print("MSIS finished")

# =========================
# 1日平均
# =========================

daily_obs  = df_period["density_norm"].resample("D").mean()
daily_msis = df_period["rho_msis"].resample("D").mean()

# DOY
doy = daily_obs.index.dayofyear

# =========================
# プロット
# =========================

plt.figure(figsize=(8,4))

plt.plot(
    doy,
    daily_obs.values,
    marker="o",
    markersize=4,
    linewidth=1.5,
    label="Observed (normalized)"
)

plt.plot(
    doy,
    daily_msis.values,
    marker="s",
    markersize=4,
    linewidth=1.5,
    label="MSIS"
)

plt.grid(alpha=0.3)
plt.xlabel("Day of Year (DOY)")
plt.ylabel("Daily mean density [kg/m³]")

plt.yscale("log")

plt.title("Swarm-A density vs MSIS (2018/02/05–02/21)")
plt.legend()

plt.tight_layout()
plt.show()