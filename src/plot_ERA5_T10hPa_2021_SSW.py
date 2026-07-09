"""
plot_ERA5_T10hPa_2021_SSW.py

Purpose:
    Plot the stratospheric temperature at 10 hPa averaged over 60°N-90°N
    for the period 2020-12-01 to 2021-01-31 using ERA5 NetCDF files.
    This helps determine the correct onset, peak, and duration of the 2021 NH SSW.

Files:
    data/SSW2021/ERA5/c3e3cc0b20096c402ec4c9c79de87e8f.nc (Dec 2020)
    data/SSW2021/ERA5/689b0c92464c4c0b060a3908effdbec3.nc (Jan 2021)

Output:
    Figure/2021/plot_ERA5_T10hPa_2021_SSW.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ERA5_DIR = Path("data/SSW2021/ERA5")
OUT_PNG  = Path("/Users/aberyusei/TMDFromPOD/Figure/2021/plot_ERA5_T10hPa_2021_SSW.png")

def main():
    print("Loading ERA5 NetCDF files...")
    files = sorted(list(ERA5_DIR.glob("*.nc")))
    if len(files) == 0:
        raise FileNotFoundError(f"No NetCDF files found in {ERA5_DIR}")
    
    # 複数ファイルのロードと結合
    datasets = []
    for fp in files:
        print(f"  Reading {fp.name}...")
        ds = xr.open_dataset(fp)
        datasets.append(ds)
        
    ds_all = xr.concat(datasets, dim="valid_time")
    
    print("Selecting 10 hPa and 60°N–90°N...")
    # 緯度60°N以上、および10 hPa面を指定
    # latitude は 90.0 から -90.0 へ減少する順序
    ds_sub = ds_all.sel(latitude=slice(90.0, 60.0))
    
    # 面積重みの計算 (cos緯度)
    weights = np.cos(np.deg2rad(ds_sub["latitude"]))
    weights.name = "weights"
    
    # 重み付き空間平均の算出
    print("Computing area-weighted spatial mean...")
    weighted_temp = ds_sub["t"].weighted(weights)
    temp_1d = weighted_temp.mean(dim=["latitude", "longitude"])
    
    # pressure_level次元が1なので、selで消去するかsqueezeする
    temp_1d = temp_1d.squeeze()
    
    # pandas Seriesに変換
    df_temp = pd.Series(temp_1d.values, index=pd.to_datetime(temp_1d["valid_time"].values))
    
    # 日平均値の算出 (daily mean)
    df_daily = df_temp.resample("D").mean()
    print(f"  Daily series generated: {len(df_daily)} days ({df_daily.index.min().date()} to {df_daily.index.max().date()})")
    
    # 最低値・最高値の探索
    t_min_val, t_max_val = df_daily.min(), df_daily.max()
    t_min_date = df_daily.idxmin().date()
    t_max_date = df_daily.idxmax().date()
    print(f"  Minimum daily Temperature: {t_min_val:.2f} K on {t_min_date}")
    print(f"  Maximum daily Temperature: {t_max_val:.2f} K on {t_max_date}")

    # ── プロット ──────────────────────────────────────────
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # 折れ線
    ax.plot(df_daily.index, df_daily.values, color="#d62728", linewidth=2.5, marker="o", markersize=4, label="Daily Zonal Mean Temp (10 hPa, 60°N–90°N)")
    
    # 特異日マーク
    ax.axvline(pd.Timestamp("2021-01-05"), color="blue", linestyle="--", linewidth=1.5, label="Official Zonal Wind Onset (01/05)")
    ax.axvline(pd.Timestamp(t_max_date), color="green", linestyle="--", linewidth=1.5, label=f"T10hPa Peak ({t_max_date.strftime('%m/%d')}: {t_max_val:.1f} K)")
    
    # 軸・装飾
    ax.set_title("ERA5 Stratospheric Temperature at 10 hPa (60°N–90°N average)\nWinter 2020/2021 NH SSW Analysis (Dec-Feb)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Temperature [K]", fontsize=11, fontweight="bold")
    ax.set_xlim(df_daily.index.min() - pd.Timedelta(days=1), df_daily.index.max() + pd.Timedelta(days=1))
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.xticks(rotation=30)
    
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    
    # 温度急上昇の期間ハイライト用目安ライン
    ax.text(pd.Timestamp("2021-01-05"), ax.get_ylim()[0] + 2, " Onset\n (01/05)", color="blue", fontsize=9, fontweight="bold")
    ax.text(pd.Timestamp(t_max_date), ax.get_ylim()[0] + 2, f" Peak Temp\n ({t_max_date.strftime('%m/%d')})", color="green", fontsize=9, fontweight="bold")
    
    # 画像保存
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    plt.close()

if __name__ == "__main__":
    main()
