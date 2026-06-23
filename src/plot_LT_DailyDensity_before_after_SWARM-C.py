"""
plot_LT_DailyDensity_before_after_SWARM-A.py

目的:
    正規化前の観測密度・正規化後の観測密度・MSIS参照密度を
    LTセクターごと（Morning: 04-11 LT, Evening: 16-23 LT）に分けて
    1日平均の折れ線グラフで重ね描きし、正規化の効果を確認する。

出力:
    Figure/2018/LT_DailyDensity_before_after_SWARM-C_DOY20-80.png
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
RAW_PARQUET  = Path("integrateddata/2018/swarm_dnscpod_2018.parquet")
NORM_PARQUET = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_DOY20-80.parquet")
OUT_PNG      = Path("Figure/2018/LT_DailyDensity_before_after_SWARM-C_DOY20-80.png")

DOY_START, DOY_END = 20, 80

SECTOR_1 = (4, 11)   # Morning
SECTOR_2 = (16, 23)  # Evening
SECTOR_1_LABEL = "Morning (04–11 LT)"
SECTOR_2_LABEL = "Evening (16–23 LT)"
MORNING_WRAPS = False

# MSIS基準条件（正規化と同じ設定）
ALT_REF_KM = 450.0
F107_REF   = 70.0
AP_REF     = 4.0

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        df = df.reset_index()
        col0 = df.columns[0]
        if col0 != "datetime":
            df = df.rename(columns={col0: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.dropna(subset=["datetime", "lst_h"])
    
    # DOY integer
    df["DOY_int"] = df["datetime"].dt.dayofyear
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]
    return df

def filter_sector(df: pd.DataFrame, lt_min: float, lt_max: float, wraps: bool) -> pd.DataFrame:
    if wraps:
        return df[(df["lst_h"] >= lt_min) | (df["lst_h"] < lt_max)]
    else:
        return df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]

def main():
    print("Loading raw data ...")
    df_raw = load_data(RAW_PARQUET)
    
    # Check if we should fallback to the full year normalized file if the DOY20-80 one doesn't exist
    if not NORM_PARQUET.exists():
        norm_path = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized.parquet")
        print(f"Fallback to {norm_path}")
    else:
        norm_path = NORM_PARQUET
        
    print("Loading normalized data ...")
    df_norm = load_data(norm_path)
    
    print("Running MSIS ...")
    times = df_norm["datetime"].dt.to_pydatetime()
    lats  = df_norm["lat"].to_numpy()
    lons  = df_norm["lon"].to_numpy()
    N     = len(df_norm)
    alts  = np.full(N, ALT_REF_KM)
    f107  = np.full(N, F107_REF)
    f107a = np.full(N, F107_REF)
    ap    = np.full((N, 7), AP_REF)
    
    result = msis.run(times, lons, lats, alts, f107, f107a, ap)
    df_norm["rho_msis"] = result[:, 0]
    
    # Filter by sectors
    raw_s1 = filter_sector(df_raw, SECTOR_1[0], SECTOR_1[1], False)
    raw_s2 = filter_sector(df_raw, SECTOR_2[0], SECTOR_2[1], False)
    
    norm_s1 = filter_sector(df_norm, SECTOR_1[0], SECTOR_1[1], False)
    norm_s2 = filter_sector(df_norm, SECTOR_2[0], SECTOR_2[1], False)
    
    # Daily means (use DOY_int)
    daily_raw_s1 = raw_s1.groupby("DOY_int")["density"].mean()
    daily_raw_s2 = raw_s2.groupby("DOY_int")["density"].mean()
    
    # We might have density_norm or just density for the normalized file? Wait, normalization script adds 'density_norm'
    norm_col = "density_norm" if "density_norm" in df_norm.columns else "density"
    daily_norm_s1 = norm_s1.groupby("DOY_int")[norm_col].mean()
    daily_norm_s2 = norm_s2.groupby("DOY_int")[norm_col].mean()
    
    daily_msis_s1 = norm_s1.groupby("DOY_int")["rho_msis"].mean()
    daily_msis_s2 = norm_s2.groupby("DOY_int")["rho_msis"].mean()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    
    sectors_data = [
        (axes[0], SECTOR_1_LABEL, daily_raw_s1, daily_norm_s1, daily_msis_s1),
        (axes[1], SECTOR_2_LABEL, daily_raw_s2, daily_norm_s2, daily_msis_s2),
    ]
    
    all_vals = []
    
    for ax, label, d_raw, d_norm, d_msis in sectors_data:
        doy = d_raw.index.to_numpy()
        ax.plot(d_raw.index, d_raw.values, marker="o", markersize=4, lw=1.5, color="#E05C2A", label="Before norm", zorder=3)
        ax.plot(d_norm.index, d_norm.values, marker="s", markersize=4, lw=1.5, color="#2A6AE0", label="After norm", zorder=3)
        ax.plot(d_msis.index, d_msis.values, marker="^", markersize=4, lw=1.5, ls="--", color="#3AAA45", label=f"MSIS ref", zorder=2)
        
        all_vals.extend(d_raw.dropna().values)
        all_vals.extend(d_norm.dropna().values)
        all_vals.extend(d_msis.dropna().values)
        
        ax.set_ylabel("Density [kg m$^{-3}$]", fontsize=11)
        ax.grid(which="major", alpha=0.45, linewidth=0.8)
        ax.grid(which="minor", alpha=0.18, linewidth=0.5)
        ax.text(0.01, 0.97, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")
        ax.legend(fontsize=9, loc="upper right")
    
    axes[1].set_xlabel("Day of Year (2018)", fontsize=12)
    
    # Y-axis limits & log scale
    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = 0.1
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(ymin * (1 - margin), ymax * (1 + margin))
        
        exp_base = int(np.floor(np.log10(ymin)))
        base = 10 ** exp_base
        data_range = ymax - ymin
        raw_step = data_range / 5.0 / base
        step = min([0.05, 0.1, 0.2, 0.5, 1.0, 2.0], key=lambda s: abs(s - raw_step))
        step_val = step * base
        
        tick_start = np.floor(ymin / step_val) * step_val
        major_ticks = []
        v = tick_start
        while v <= ymax * (1 + margin) * 1.01:
            if v >= ymin * (1 - margin) * 0.99:
                major_ticks.append(v)
            v = round(v + step_val, 20)
            
        ax.set_yticks(major_ticks)
        def fmt(x, _, exp=exp_base, b=base):
            return f"{x/b:.2g}×10$^{{{exp}}}$"
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
        
        minor_ticks = []
        for i in range(len(major_ticks) - 1):
            lo, hi = major_ticks[i], major_ticks[i + 1]
            for k in range(1, 4):
                minor_ticks.append(lo + (hi - lo) * k / 4)
        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticks(range(DOY_START, DOY_END + 1, 5))

    fig.suptitle("Swarm-C: Daily Mean Density (DOY 20–80) by LT Sector\nBefore vs After Normalization", fontsize=13, fontweight="bold", y=0.96)
    
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {OUT_PNG}")

if __name__ == "__main__":
    main()
