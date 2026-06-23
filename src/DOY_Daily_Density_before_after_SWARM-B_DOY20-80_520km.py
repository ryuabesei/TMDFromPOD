"""
DOY_Daily_Density_before_after_SWARM-B.py

目的:
    正規化前の観測密度・正規化後の観測密度・MSIS参照密度を
    1日平均の折れ線グラフで重ね描きし、正規化の効果を確認する。

線の構成:
    ① Raw observed density      : integrateddata/2018/swarm_dnsbpod_2018.parquet
    ② Normalized density        : normalizeddata/2018/swarm_dnsbpod_2018_normalized(450km).parquet
    ③ MSIS reference density    : 基準条件（alt=450km, F10.7=70, Ap=4）で計算

出力:
    Figure/2018/DailyDensity_before_after_SWARM-B.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pymsis import msis

# =========================
# 設定
# =========================
RAW_PARQUET  = Path("integrateddata/2018/swarm_dnsbpod_2018.parquet")
NORM_PARQUET = Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized_DOY20-80.parquet")
OUT_PNG      = Path("Figure/2018/DailyDensity_before_after_SWARM-B_DOY20-80_520km.png")

START_DATE = "2018-01-20"
END_DATE   = "2018-03-21"

# MSIS基準条件（正規化と同じ設定）
ALT_REF_KM = 520.0
F107_REF   = 70.0
AP_REF     = 4.0


# =========================
# ユーティリティ
# =========================
def load_raw(path: Path) -> pd.DataFrame:
    """正規化前データを読み込む。インデックスがDatetimeIndexの場合に対応。"""
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        df = df.reset_index()
        col0 = df.columns[0]
        if col0 != "datetime":
            df = df.rename(columns={col0: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    return df


def load_norm(path: Path) -> pd.DataFrame:
    """正規化後データを読み込む。"""
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    return df


# =========================
# データ読み込み
# =========================
print("Loading raw data ...")
df_raw = load_raw(RAW_PARQUET)
df_raw_period = df_raw.loc[START_DATE:END_DATE].copy()
print(f"  raw  : {len(df_raw_period):,} rows")

print("Loading normalized data ...")
df_norm = load_norm(NORM_PARQUET)
df_norm_period = df_norm.loc[START_DATE:END_DATE].copy()
print(f"  norm : {len(df_norm_period):,} rows")

# =========================
# MSIS密度計算（基準条件）
# =========================
print("Running MSIS ...")
times = df_norm_period.index.to_pydatetime()
lats  = df_norm_period["lat"].to_numpy()
lons  = df_norm_period["lon"].to_numpy()
N     = len(df_norm_period)
alts  = np.full(N, ALT_REF_KM)
f107  = np.full(N, F107_REF)
f107a = np.full(N, F107_REF)
ap    = np.full((N, 7), AP_REF)

result = msis.run(times, lons, lats, alts, f107, f107a, ap)
df_norm_period = df_norm_period.copy()
df_norm_period["rho_msis"] = result[:, 0]
print("  MSIS finished")

# =========================
# 1日平均
# =========================
daily_raw  = df_raw_period["density"].resample("D").mean()
daily_norm = df_norm_period["density_norm"].resample("D").mean()
daily_msis = df_norm_period["rho_msis"].resample("D").mean()

# DOY
doy_raw  = daily_raw.index.dayofyear
doy_norm = daily_norm.index.dayofyear
doy_msis = daily_msis.index.dayofyear

# =========================
# プロット
# =========================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    doy_raw,
    daily_raw.values,
    marker="o",
    markersize=5,
    linewidth=1.8,
    color="#E05C2A",
    label="Before normalization (observed)",
    zorder=3,
)

ax.plot(
    doy_norm,
    daily_norm.values,
    marker="s",
    markersize=5,
    linewidth=1.8,
    color="#2A6AE0",
    label="After normalization",
    zorder=3,
)

ax.plot(
    doy_msis,
    daily_msis.values,
    marker="^",
    markersize=5,
    linewidth=1.8,
    linestyle="--",
    color="#3AAA45",
    label=f"MSIS ref (alt={ALT_REF_KM:.0f} km, F10.7={F107_REF:.0f}, Ap={AP_REF:.0f})",
    zorder=2,
)


ax.set_xlabel("Day of Year (2018)", fontsize=12)
ax.set_ylabel("Daily mean density [kg m$^{-3}$]", fontsize=12)
ax.set_title(
    "Swarm-B: Daily Mean Density – Before vs After Normalization\n"
    f"({START_DATE} – {END_DATE})",
    fontsize=13,
    fontweight="bold",
)

ax.set_yscale("log")

# =========================
# 縦軸：データの値域に合わせた目盛り設定
# =========================
all_vals = np.concatenate([
    daily_raw.dropna().values,
    daily_norm.dropna().values,
    daily_msis.dropna().values,
])
ymin_data = np.nanmin(all_vals)
ymax_data = np.nanmax(all_vals)

# --- y軸範囲：データに上下10%の余白だけ確保 ---
margin = 0.10
ax.set_ylim(ymin_data * (1 - margin), ymax_data * (1 + margin))

# --- major tick：データ桁の 0.x 刻みを自動生成 ---
# データ範囲が1桁未満なので、共通の指数を持つ線形刻みを作る
exp_base = int(np.floor(np.log10(ymin_data)))   # 例: -13
base = 10 ** exp_base                            # 例: 1e-13

# step を データレンジ / 希望tick数(~6) から決定し、きりの良い値に丸める
data_range = ymax_data - ymin_data
raw_step = data_range / 6.0 / base              # 基準単位での刻み幅
# 0.05, 0.1, 0.2, 0.5, 1.0, 2.0 から最も近いものを選ぶ
nice_steps = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
step = min(nice_steps, key=lambda s: abs(s - raw_step))
step_val = step * base  # 実際の刻み幅 (kg/m³)

# tick の開始をデータ最小値以下の切りの良い値に合わせる
tick_start = np.floor(ymin_data / step_val) * step_val
major_ticks = []
v = tick_start
while v <= ymax_data * (1 + margin) * 1.01:
    if v >= ymin_data * (1 - margin) * 0.99:
        major_ticks.append(v)
    v = round(v + step_val, 20)  # 浮動小数誤差対策

ax.set_yticks(major_ticks)

# ラベル: 例「1.2×10⁻¹³」形式
def fmt_density(x, _):
    mantissa = x / base
    return f"{mantissa:.2g}×10$^{{{exp_base}}}$"

ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_density))

# minor tick: major の間を4等分（log スケールでは FixedLocator で手動設定）
minor_ticks = []
for i in range(len(major_ticks) - 1):
    lo, hi = major_ticks[i], major_ticks[i + 1]
    for k in range(1, 4):
        minor_ticks.append(lo + (hi - lo) * k / 4)
ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

ax.tick_params(axis="y", which="major", labelsize=10, length=6, width=1.0)
ax.tick_params(axis="y", which="minor", length=3,  width=0.7)

ax.grid(which="major", alpha=0.45, linewidth=0.8)
ax.grid(which="minor", alpha=0.18, linewidth=0.5)
ax.legend(fontsize=10, loc="upper left")

plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\n✅ 保存完了: {OUT_PNG}")
plt.show()
