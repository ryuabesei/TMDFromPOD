"""
residual_density_SWARM-A.py

目的:
    1. DOY 37~42 の正規化密度の平均を non-SSW reference とする
    2. DOY 36~52 の観測密度から reference を引いて residual を計算
    3. residual を左Y軸、COSMIC T(10 hPa) を右Y軸に重ねてプロット

出力:
    Figure/residual_density_SWARM-A.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =========================
# 設定
# =========================
NORM_PARQUET = Path("normalizeddata/swarm_dnsapod_2018_normalized.parquet")
COSMIC_CSV   = Path("cosmic_T10hPa_daily_2018_DOY032_050_lat60_90N.csv")
OUT_PNG      = Path("Figure/residual_density_SWARM-A.png")

DOY_START   = 36
DOY_END     = 52
DOY_REF_MIN = 37   # non-SSW reference 期間（開始, 含む）
DOY_REF_MAX = 42   # non-SSW reference 期間（終了, 含む）

# 対応する日付
DATE_START = "2018-02-05"   # DOY 36
DATE_END   = "2018-02-21"   # DOY 52

# =========================
# 正規化密度 読み込み
# =========================
print("Loading normalized density ...")
df_norm = pd.read_parquet(NORM_PARQUET)
df_norm["datetime"] = pd.to_datetime(df_norm["datetime"], utc=True)
df_norm = df_norm.set_index("datetime").sort_index()

# DOY 36–52 を抽出
df_period = df_norm.loc[DATE_START:DATE_END].copy()
print(f"  rows in period: {len(df_period):,}")

# 1日平均密度
daily_norm = df_period["density_norm"].resample("D").mean()
daily_norm.index = daily_norm.index.tz_localize(None)  # tz除去して扱いやすく
doy_all = daily_norm.index.dayofyear

# =========================
# non-SSW reference（DOY 37~42 の平均）
# =========================
mask_ref = (doy_all >= DOY_REF_MIN) & (doy_all <= DOY_REF_MAX)
reference = float(daily_norm[mask_ref].mean())
print(f"  Non-SSW reference (DOY {DOY_REF_MIN}–{DOY_REF_MAX}): {reference:.4e} kg/m³")

# =========================
# Residual = observed - reference
# =========================
residual = daily_norm - reference
doy_plot = doy_all.to_numpy()
res_vals = residual.to_numpy()

# =========================
# COSMIC T(10 hPa) 読み込み
# =========================
print("Loading COSMIC T10hPa ...")
df_cos = pd.read_csv(COSMIC_CSV, parse_dates=["datetime"])
df_cos["datetime"] = pd.to_datetime(df_cos["datetime"], utc=True)
df_cos = df_cos.set_index("datetime").sort_index()

df_cos_period = df_cos.loc[DATE_START:DATE_END]
T10 = df_cos_period["T10_K"]
doy_cos = T10.index.dayofyear.to_numpy()
t10_vals = T10.to_numpy()

# =========================
# プロット
# =========================
fig, ax1 = plt.subplots(figsize=(10, 5))

# --- 左軸: Residual density ---
color_res = "#2A6AE0"
ax1.bar(
    doy_plot,
    res_vals,
    width=0.7,
    color=[color_res if v >= 0 else "#E05C2A" for v in res_vals],
    alpha=0.75,
    label="Residual density (obs − ref)",
    zorder=3,
)
ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)

ax1.set_xlabel("Day of Year (2018)", fontsize=12)
ax1.set_ylabel("Residual density [kg m$^{-3}$]", fontsize=12, color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.set_xlim(DOY_START - 0.5, DOY_END + 0.5)

# Y軸範囲をデータに合わせる
ylim_abs = np.nanmax(np.abs(res_vals[np.isfinite(res_vals)])) * 1.3
ax1.set_ylim(-ylim_abs, ylim_abs)
ax1.grid(axis="y", alpha=0.3, linewidth=0.7)

# reference 帯をハイライト
ax1.axvspan(DOY_REF_MIN - 0.5, DOY_REF_MAX + 0.5,
            color="gray", alpha=0.12, label=f"Reference period (DOY {DOY_REF_MIN}–{DOY_REF_MAX})")

# --- 右軸: COSMIC T(10 hPa) ---
ax2 = ax1.twinx()
color_T = "hotpink"
ax2.plot(
    doy_cos,
    t10_vals,
    marker="o",
    markersize=5,
    linewidth=2.0,
    color=color_T,
    label="COSMIC T (10 hPa)",
    zorder=4,
)
ax2.set_ylabel("T (10 hPa) [K]", fontsize=12, color=color_T)
ax2.tick_params(axis="y", labelcolor=color_T)
T_margin = (t10_vals.max() - t10_vals.min()) * 0.3
ax2.set_ylim(t10_vals.min() - T_margin, t10_vals.max() + T_margin)

# --- 凡例統合 ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

plt.title(
    f"Swarm-A Residual Density & COSMIC T(10 hPa)  [DOY {DOY_START}–{DOY_END}, 2018]\n"
    f"Reference: daily mean density averaged over DOY {DOY_REF_MIN}–{DOY_REF_MAX}",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()

# =========================
# 保存
# =========================
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\n✅ 保存完了: {OUT_PNG}")
plt.show()
