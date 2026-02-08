import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 設定
# =========================
PARQUET = "normalizeddata/swarm_dnscpod_2018_normalized.parquet"

# プロット期間（15日）
T_START = "2018-02-05 00:00:00"
T_END   = "2018-02-20 23:59:59"

# LT セクター（Swarm-C実データに合わせた）
SECTOR_LEFT  = (7, 9)     # 07–09 LT
SECTOR_RIGHT = (18, 21)   # 18–21 LT

# 緯度範囲（正規化データはすでに|lat|<=60のはずだが、図として明示）
LAT_MIN, LAT_MAX = -60, 60

# 2D bin の設定（DoY方向: 0.5日、緯度方向: 3度）
DOY_BIN = 0.5
LAT_BIN = 3.0

# 等高線レベル数
N_LEVELS = 20


# =========================
# ユーティリティ：DOY（連続値）を作る
# =========================
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


# =========================
# ユーティリティ：2D binning (median)
# =========================
def grid_median(df: pd.DataFrame, doy_bins: np.ndarray, lat_bins: np.ndarray, value_col: str) -> np.ndarray:
    Z = np.full((len(lat_bins)-1, len(doy_bins)-1), np.nan)

    # numpyで高速化（ループ最小化）
    doy = df["DOY"].to_numpy()
    lat = df["lat"].to_numpy()
    val = df[value_col].to_numpy()

    # bin index を作る
    doy_i = np.digitize(doy, doy_bins) - 1
    lat_i = np.digitize(lat, lat_bins) - 1

    # 範囲外を落とす
    ok = (doy_i >= 0) & (doy_i < len(doy_bins)-1) & (lat_i >= 0) & (lat_i < len(lat_bins)-1)
    doy_i, lat_i, val = doy_i[ok], lat_i[ok], val[ok]

    # 各セルに値をためて median
    # （セル数が多いので、groupbyの代わりに辞書で集計）
    from collections import defaultdict
    bucket = defaultdict(list)
    for i, j, v in zip(lat_i, doy_i, val):
        bucket[(i, j)].append(v)

    for (i, j), arr in bucket.items():
        Z[i, j] = float(np.median(arr))

    return Z


# =========================
# ユーティリティ：DOYごとの「平均LT軌道」を作る（黒線用）
# =========================
def daily_mean_lt_line(df: pd.DataFrame, lt_min: float, lt_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    各日について、そのLTセクター内の lst_h の平均を求め、
    (DOY + 0.5, mean_LT) の折れ線を返す
    """
    g = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)].copy()
    if len(g) == 0:
        return np.array([]), np.array([])

    daily = g.set_index("datetime").resample("D")["lst_h"].mean().dropna()

    x = daily.index.dayofyear + 0.5
    y = daily.to_numpy()
    return x, y


# =========================
# メイン
# =========================
df = pd.read_parquet(PARQUET)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

# 期間抽出
df = df[(df["datetime"] >= pd.Timestamp(T_START, tz="UTC")) &
        (df["datetime"] <= pd.Timestamp(T_END, tz="UTC"))].copy()

# 緯度範囲（念のため）
df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)].copy()

# DOY追加
df = add_doy(df)

# DOY bins を期間に合わせて作る（半日刻み）
doy_start = df["DOY"].min()
doy_end   = df["DOY"].max()
doy_bins = np.arange(np.floor(doy_start), np.ceil(doy_end) + DOY_BIN, DOY_BIN)

lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

# LTセクターごとに抽出
df_morning = df[(df["lst_h"] >= SECTOR_LEFT[0]) & (df["lst_h"] < SECTOR_LEFT[1])].copy()
df_evening = df[(df["lst_h"] >= SECTOR_RIGHT[0]) & (df["lst_h"] < SECTOR_RIGHT[1])].copy()

print("N morning (07–09):", len(df_morning))
print("N evening (18–20):", len(df_evening))

# 2Dグリッド（median）
Z_morning = grid_median(df_morning, doy_bins, lat_bins, "density_norm")
Z_evening = grid_median(df_evening, doy_bins, lat_bins, "density_norm")

# 両パネルでカラースケールを揃える（見た目と比較性のため）
# NaNを除いた範囲で vmin/vmax を決める
all_vals = np.concatenate([
    Z_morning[np.isfinite(Z_morning)].ravel(),
    Z_evening[np.isfinite(Z_evening)].ravel()
])
vmin = np.nanpercentile(all_vals, 2)
vmax = np.nanpercentile(all_vals, 98)

# 描画
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

X, Y = np.meshgrid(doy_bins[:-1], lat_bins[:-1])

for ax, Z, title, (ltmin, ltmax) in [
    (axes[0], Z_morning, f"Swarm-C normalized density ({SECTOR_LEFT[0]:02d}–{SECTOR_LEFT[1]:02d} LT)", SECTOR_LEFT),
    (axes[1], Z_evening, f"Swarm-C normalized density ({SECTOR_RIGHT[0]:02d}–{SECTOR_RIGHT[1]:02d} LT)", SECTOR_RIGHT),
]:
    cf = ax.contourf(X, Y, Z, levels=N_LEVELS, vmin=vmin, vmax=vmax, cmap="turbo")
    ax.set_xlabel("Day of Year (2018)")
    ax.grid(alpha=0.2)
    ax.set_title(title)

    # ---- 右側に「LT (h)」軸ラベルを置く（先行研究風） ----
    # 実際のy軸は緯度だが、右側の補助軸としてLT表記を置く
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_yticks([])  # 目盛りは消してラベルだけ
    ax_r.set_ylabel("LT (h)")

    # ---- DOYごとの平均LT（黒線）を重ねる ----
    # 緯度-DOY図の上に「その日の平均LT」を“表示として”載せたいので、
    # 右軸(LT)に対してプロットする
    x_lt, y_lt = daily_mean_lt_line(df, ltmin, ltmax)
    if len(x_lt) > 0:
        ax_r.plot(x_lt, y_lt, color="k", lw=1.2)

# 左の縦軸だけ緯度ラベル
axes[0].set_ylabel("Geographic Latitude")
axes[0].set_ylim(LAT_MIN, LAT_MAX)

# 共通カラーバー
cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), pad=0.02)
cbar.set_label("Normalized density [kg m$^{-3}$] (450 km reference)")

plt.tight_layout()
plt.show()
