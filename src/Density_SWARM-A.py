import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PARQUET = "normalizeddata/swarm_dnsapod_2018_normalized.parquet"
T_START = "2018-02-05 00:00:00"
T_END   = "2018-02-20 23:59:59"

# セクター（あなたが今使っている設定に合わせる）
SECTOR_LEFT  = (7, 9)     # 07–09
SECTOR_RIGHT = (18, 21)   # 18–21

LAT_MIN, LAT_MAX = -60, 60
DOY_BIN = 0.5
LAT_BIN = 3.0
N_LEVELS = 20

def add_doy(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    df = df.copy()
    df["DOY"] = (
        dt.dt.dayofyear
        + dt.dt.hour / 24.0
        + dt.dt.minute / 1440.0
        + dt.dt.second / 86400.0
    )
    return df

def grid_median(df: pd.DataFrame, doy_bins: np.ndarray, lat_bins: np.ndarray, value_col: str) -> np.ndarray:
    Z = np.full((len(lat_bins)-1, len(doy_bins)-1), np.nan)

    doy = df["DOY"].to_numpy()
    lat = df["lat"].to_numpy()
    val = df[value_col].to_numpy()

    doy_i = np.digitize(doy, doy_bins) - 1
    lat_i = np.digitize(lat, lat_bins) - 1

    ok = (doy_i >= 0) & (doy_i < len(doy_bins)-1) & (lat_i >= 0) & (lat_i < len(lat_bins)-1)
    doy_i, lat_i, val = doy_i[ok], lat_i[ok], val[ok]

    from collections import defaultdict
    bucket = defaultdict(list)
    for i, j, v in zip(lat_i, doy_i, val):
        bucket[(i, j)].append(v)

    for (i, j), arr in bucket.items():
        Z[i, j] = float(np.median(arr))

    return Z

def daily_mean_lt_line(df: pd.DataFrame, lt_min: float, lt_max: float) -> tuple[np.ndarray, np.ndarray]:
    g = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)].copy()
    if len(g) == 0:
        return np.array([]), np.array([])

    daily = g.set_index("datetime").resample("D")["lst_h"].mean().dropna()
    x = daily.index.dayofyear + 0.5
    y = daily.to_numpy()
    return x, y

# --- load ---
df = pd.read_parquet(PARQUET)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

df = df[(df["datetime"] >= pd.Timestamp(T_START, tz="UTC")) &
        (df["datetime"] <= pd.Timestamp(T_END, tz="UTC"))].copy()
df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)].copy()
df = add_doy(df)

# bins
doy_bins = np.arange(np.floor(df["DOY"].min()), np.ceil(df["DOY"].max()) + DOY_BIN, DOY_BIN)
lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

# sectors
df_left  = df[(df["lst_h"] >= SECTOR_LEFT[0])  & (df["lst_h"] < SECTOR_LEFT[1])].copy()
df_right = df[(df["lst_h"] >= SECTOR_RIGHT[0]) & (df["lst_h"] < SECTOR_RIGHT[1])].copy()

print("N left :", len(df_left))
print("N right:", len(df_right))

Z_left  = grid_median(df_left, doy_bins, lat_bins, "density_norm")
Z_right = grid_median(df_right, doy_bins, lat_bins, "density_norm")

# 共通カラースケール
all_vals = np.concatenate([Z_left[np.isfinite(Z_left)].ravel(), Z_right[np.isfinite(Z_right)].ravel()])
vmin = np.nanpercentile(all_vals, 2)
vmax = np.nanpercentile(all_vals, 98)

# --- figure layout: 最初から colorbar 用の列を確保する ---
fig = plt.figure(figsize=(12.8, 4.2))
gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1, 1, 0.04], wspace=0.15)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
cax = fig.add_subplot(gs[0, 2])  # ←ここにカラーバーを固定

X, Y = np.meshgrid(doy_bins[:-1], lat_bins[:-1])

cf1 = ax1.contourf(X, Y, Z_left,  levels=N_LEVELS, vmin=vmin, vmax=vmax, cmap="turbo")
cf2 = ax2.contourf(X, Y, Z_right, levels=N_LEVELS, vmin=vmin, vmax=vmax, cmap="turbo")

# 軸設定
ax1.set_title(f"Swarm-A normalized density ({SECTOR_LEFT[0]:02d}–{SECTOR_LEFT[1]:02d} LT)")
ax2.set_title(f"Swarm-A normalized density ({SECTOR_RIGHT[0]:02d}–{SECTOR_RIGHT[1]:02d} LT)")
ax1.set_xlabel("Day of Year (2018)")
ax2.set_xlabel("Day of Year (2018)")
ax1.set_ylabel("Geographic Latitude")
ax1.set_ylim(LAT_MIN, LAT_MAX)
ax1.grid(alpha=0.2); ax2.grid(alpha=0.2)
plt.setp(ax2.get_yticklabels(), visible=False)

# 右側に LT ラベル（目盛りは消してラベルだけ）
for ax, (ltmin, ltmax) in [(ax1, SECTOR_LEFT), (ax2, SECTOR_RIGHT)]:
    axr = ax.twinx()
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks([])
    axr.set_ylabel("LT (h)")

    # “平均LT”の線（右軸に対して）
    x_lt, y_lt = daily_mean_lt_line(df, ltmin, ltmax)
    if len(x_lt) > 0:
        axr.plot(x_lt, y_lt, color="k", lw=1.2)

# カラーバー（右外側に固定）
cb = fig.colorbar(cf2, cax=cax)
cb.set_label("Normalized density [kg m$^{-3}$] (450 km reference)")

plt.show()
