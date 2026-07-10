"""
plot_3years_joint_activity.py

Purpose:
    Plot density_ratio_msis, Ap index, and F10.7 index together in a single figure
    with multi-Y axes to clearly show their temporal alignment and correlation.
    
    Layout:
        3 rows (2018 NH SSW / 2019 SH SSW / 2021 NH SSW)
        Each row has a single panel with 3 Y-axes:
            - Left Y-axis (blue): density_ratio_msis (daily median)
            - Right Y-axis 1 (orange): Daily Mean Ap (bar or line)
            - Right Y-axis 2 (green, offset): Daily F10.7 (line)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Parquet paths
P2018 = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")
OUT_PNG = Path("Figure/debug/joint_activity_3years.png")

def get_daily_data(parquet_path: Path, doy_range: tuple[int, int] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if doy_range is not None:
        df["DOY"] = df["datetime"].dt.dayofyear
        df = df[(df["DOY"] >= doy_range[0]) & (df["DOY"] <= doy_range[1])]
    df = df.dropna(subset=["density_ratio_msis", "AP_AVG", "F107"])
    df["date"] = df["datetime"].dt.normalize()
    
    daily = df.groupby("date").agg(
        ratio_med = ("density_ratio_msis", "median"),
        ap_mean = ("AP_AVG", "mean"),
        f107_mean = ("F107", "mean")
    ).reset_index()
    return daily

def main():
    print("Loading data for all 3 years...")
    df18 = get_daily_data(P2018, (30, 65))
    df19 = get_daily_data(P2019, None)  # use full available range
    df2021 = get_daily_data(P2021, None)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor("white")
    
    events = [
        {"df": df18, "label": "2018 NH SSW (SWARM-A)", "ax": axes[0]},
        {"df": df19, "label": "2019 SH SSW (SWARM-A)", "ax": axes[1]},
        {"df": df2021, "label": "2021 NH SSW (SWARM-C)", "ax": axes[2]}
    ]

    for item in events:
        df = item["df"]
        label = item["label"]
        ax1 = item["ax"]

        # 3 Y-axes setup
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # Offset the third axis (F10.7) to the right
        ax3.spines["right"].set_position(("outward", 60))

        # --- Plot 1: Density Ratio (Left axis, Blue) ---
        p1, = ax1.plot(df["date"], df["ratio_med"], color="#1f77b4", linewidth=2.0, marker="o", 
                       label="Density Ratio (median)")
        ax1.set_ylabel("Density Ratio (rho_obs/rho_MSIS)", color="#1f77b4", fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        
        # --- Plot 2: Ap Index (Right axis 1, Orange) ---
        # Using a thin line with markers to avoid blocking
        p2, = ax2.plot(df["date"], df["ap_mean"], color="#e07b39", linewidth=1.5, marker="s", markersize=4,
                       label="Daily mean Ap")
        ax2.set_ylabel("Ap Index", color="#e07b39", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#e07b39")
        ax2.set_ylim(0, max(df["ap_mean"].max() * 1.15, 20)) # Ensure scale starts at 0

        # --- Plot 3: F10.7 Index (Right axis 2, Green) ---
        p3, = ax3.plot(df["date"], df["f107_mean"], color="#2ca02c", linewidth=1.5, marker="^", markersize=4,
                       label="F10.7")
        ax3.set_ylabel("F10.7 [sfu]", color="#2ca02c", fontweight="bold")
        ax3.tick_params(axis="y", labelcolor="#2ca02c")
        
        # Adjust F10.7 limits slightly for padding
        f107_min, f107_max = df["f107_mean"].min(), df["f107_mean"].max()
        if f107_max - f107_min > 1.0:
            ax3.set_ylim(f107_min - (f107_max-f107_min)*0.1, f107_max + (f107_max-f107_min)*0.1)

        # X-axis formatting
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax1.grid(True, alpha=0.2)
        ax1.set_title(label, fontsize=12, fontweight="bold", pad=8)
        
        # Legend (combined for all three axes)
        lines = [p1, p2, p3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=9, framealpha=0.85)

    plt.suptitle("Thermospheric Density Ratio & Solar/Geomagnetic Activity Comparison", 
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    plt.close()

if __name__ == "__main__":
    main()
