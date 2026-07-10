"""
plot_model_vs_activity_3years.py

Purpose:
    Analyze and plot the correlations and time series of:
    - rho_model_real (MSIS density under real conditions)
    - Ap Index (AP_AVG)
    - F10.7 (F107)
    For 2018 NH SSW, 2019 SH SSW, and 2021 NH SSW events.

    This helps understand how the MSIS model itself responds to these inputs
    and check if it correlates as expected.
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
OUT_PNG = Path("Figure/debug/model_vs_activity_3years.png")

def get_daily_data(parquet_path: Path, doy_range: tuple[int, int] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if doy_range is not None:
        df["DOY"] = df["datetime"].dt.dayofyear
        df = df[(df["DOY"] >= doy_range[0]) & (df["DOY"] <= doy_range[1])]
    df = df.dropna(subset=["rho_model_real", "AP_AVG", "F107"])
    df["date"] = df["datetime"].dt.normalize()
    
    daily = df.groupby("date").agg(
        model_med = ("rho_model_real", "median"),
        ap_mean = ("AP_AVG", "mean"),
        f107_mean = ("F107", "mean")
    ).reset_index()
    return daily

def generate_plot(daily_df: pd.DataFrame, year: int, label: str, fig, gs_row: int):
    # Calculate correlation
    corr_ap = daily_df["model_med"].corr(daily_df["ap_mean"])
    corr_f107 = daily_df["model_med"].corr(daily_df["f107_mean"])
    
    # Sub-grid for this row
    # Left: Time series of Model, Ap, F10.7
    # Right: Scatters vs Ap and F10.7
    sub_gs = gs_row.subgridspec(1, 3, width_ratios=[2, 1, 1])
    
    # 1. Time Series (Left)
    ax0 = fig.add_subplot(sub_gs[0, 0])
    ax0_ap = ax0.twinx()
    ax0_f107 = ax0.twinx()
    ax0_f107.spines["right"].set_position(("outward", 50))
    
    scale = 1e13
    p1, = ax0.plot(daily_df["date"], daily_df["model_med"] * scale, color="#d62728", linewidth=2.0, marker="o", 
                   label="MSIS (rho_model_real)")
    ax0.set_ylabel("MSIS Density\n[10$^{-13}$ kg/m$^3$]", color="#d62728", fontweight="bold")
    ax0.tick_params(axis="y", labelcolor="#d62728")
    
    p2, = ax0_ap.plot(daily_df["date"], daily_df["ap_mean"], color="#e07b39", linewidth=1.5, marker="s", markersize=4, linestyle=":",
                       label="Ap Index")
    ax0_ap.set_ylabel("Ap Index", color="#e07b39", fontweight="bold")
    ax0_ap.tick_params(axis="y", labelcolor="#e07b39")
    ax0_ap.set_ylim(0, max(daily_df["ap_mean"].max() * 1.15, 20))
    
    p3, = ax0_f107.plot(daily_df["date"], daily_df["f107_mean"], color="#2ca02c", linewidth=1.5, marker="^", markersize=4, linestyle="--",
                        label="F10.7")
    ax0_f107.set_ylabel("F10.7 [sfu]", color="#2ca02c", fontweight="bold")
    ax0_f107.tick_params(axis="y", labelcolor="#2ca02c")
    
    ax0.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax0.grid(True, alpha=0.2)
    ax0.set_title(f"{label} (Time Series)", fontsize=11, fontweight="bold")
    
    lines = [p1, p2, p3]
    labels = [l.get_label() for l in lines]
    ax0.legend(lines, labels, loc="upper left", fontsize=8, framealpha=0.85)

    # 2. Scatter: Model vs Ap (Middle)
    ax1 = fig.add_subplot(sub_gs[0, 1])
    ax1.scatter(daily_df["ap_mean"], daily_df["model_med"] * scale, color="#ff7f0e", s=40, edgecolors="k")
    if len(daily_df) > 2:
        m, c = np.polyfit(daily_df["ap_mean"], daily_df["model_med"] * scale, 1)
        ax1.plot(daily_df["ap_mean"], m*daily_df["ap_mean"] + c, color="black", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Ap Index")
    ax1.set_ylabel("MSIS Density [10$^{-13}$]")
    ax1.set_title(f"vs Ap (Corr = {corr_ap:.3f})", fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.2)

    # 3. Scatter: Model vs F10.7 (Right)
    ax2 = fig.add_subplot(sub_gs[0, 2])
    ax2.scatter(daily_df["f107_mean"], daily_df["model_med"] * scale, color="#2ca02c", s=40, edgecolors="k")
    if len(daily_df) > 2:
        m, c = np.polyfit(daily_df["f107_mean"], daily_df["model_med"] * scale, 1)
        ax2.plot(daily_df["f107_mean"], m*daily_df["f107_mean"] + c, color="black", linestyle="--", alpha=0.7)
    ax2.set_xlabel("F10.7 index")
    ax2.set_ylabel("MSIS Density [10$^{-13}$]")
    ax2.set_title(f"vs F10.7 (Corr = {corr_f107:.3f})", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.2)

def main():
    print("Loading data...")
    df18 = get_daily_data(P2018, (30, 65))
    df19 = get_daily_data(P2019, None)
    df2021 = get_daily_data(P2021, None)

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("white")
    
    # 3 rows of gridspec
    gs = fig.add_gridspec(3, 1, hspace=0.3)
    
    generate_plot(df18, 2018, "2018 NH SSW (SWARM-A)", fig, gs[0])
    generate_plot(df19, 2019, "2019 SH SSW (SWARM-A)", fig, gs[1])
    generate_plot(df2021, 2021, "2021 NH SSW (SWARM-C)", fig, gs[2])

    plt.suptitle("MSIS Model Density Sensitivity to Ap and F10.7 Indices", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    plt.close()

if __name__ == "__main__":
    main()
