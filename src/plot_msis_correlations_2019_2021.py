"""
plot_msis_correlations_2019_2021.py

Purpose:
    Analyze and plot the correlations and time series of:
    - density_ratio_msis
    - Ap Index (AP_AVG)
    - F10.7 (F107)
    For 2019 SH SSW (SWARM-A) and 2021 NH SSW (SWARM-C).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")
OUT_DIR = Path("Figure/debug")

def generate_analysis_plots(parquet_path: Path, year: int, out_png: Path, label: str):
    print(f"Analyzing {label} ({year})...")
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["density", "rho_model_real", "density_ratio_msis", "AP_AVG", "F107"])
    df["date"] = df["datetime"].dt.normalize()
    
    # Calculate daily median / mean
    daily = df.groupby("date").agg(
        ratio_med = ("density_ratio_msis", "median"),
        ap_mean = ("AP_AVG", "mean"),
        f107_mean = ("F107", "mean")
    ).reset_index()
    
    # Calculate correlation coefficients
    corr_ap = daily["ratio_med"].corr(daily["ap_mean"])
    corr_f107 = daily["ratio_med"].corr(daily["f107_mean"])
    
    # Plotting
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("white")
    
    # Grid specification
    # Left: Time series of Ratio, Ap, F10.7
    # Right: Scatters vs Ap and F10.7
    gs = fig.add_gridspec(3, 2, width_ratios=[1.8, 1])
    
    # 1. Time Series: Density Ratio
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(daily["date"], daily["ratio_med"], color="#1f77b4", linewidth=2.0, marker="o", label="density_ratio_msis (median)")
    ax0.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax0.set_ylabel("Density Ratio\n(rho_obs/rho_MSIS)")
    ax0.set_title(f"{label} Time Series & Activity Indices", fontsize=12, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left")
    
    # 2. Time Series: Ap Index
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.bar(daily["date"], daily["ap_mean"], color="#e07b39", alpha=0.8, width=0.6, label="Daily mean Ap")
    ax1.set_ylabel("Ap Index")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    
    # 3. Time Series: F10.7 Index
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.plot(daily["date"], daily["f107_mean"], color="#2ca02c", linewidth=1.8, marker="s", markersize=4, label="F10.7 index")
    ax2.set_ylabel("F10.7 [sfu]")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    
    # Rotate x labels
    plt.setp(ax2.get_xticklabels(), rotation=15)
    
    # 4. Scatter: Ratio vs Ap
    ax3 = fig.add_subplot(gs[0:2, 1])
    ax3.scatter(daily["ap_mean"], daily["ratio_med"], color="#d62728", s=60, edgecolors="k", zorder=3)
    # Fit line if there are enough points
    if len(daily) > 2:
        m, c = np.polyfit(daily["ap_mean"], daily["ratio_med"], 1)
        ax3.plot(daily["ap_mean"], m*daily["ap_mean"] + c, color="black", linestyle="--", alpha=0.7, 
                 label=f"Fit (slope={m:.4f})")
    ax3.set_xlabel("Daily Mean Ap Index")
    ax3.set_ylabel("Density Ratio (median)")
    ax3.set_title(f"Ratio vs Ap (Corr = {corr_ap:.3f})", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 5. Scatter: Ratio vs F10.7
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.scatter(daily["f107_mean"], daily["ratio_med"], color="#9467bd", s=40, edgecolors="k", zorder=3)
    if len(daily) > 2:
        m, c = np.polyfit(daily["f107_mean"], daily["ratio_med"], 1)
        ax4.plot(daily["f107_mean"], m*daily["f107_mean"] + c, color="black", linestyle="--", alpha=0.7)
    ax4.set_xlabel("Daily Mean F10.7 Index")
    ax4.set_ylabel("Density Ratio (median)")
    ax4.set_title(f"Ratio vs F10.7 (Corr = {corr_f107:.3f})", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f"SWARM Activity & MSIS Ratio Analysis: {year} SSW Event", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_png}")
    plt.close()

def main():
    generate_analysis_plots(P2019, 2019, OUT_DIR / "msis_activity_correlation_2019.png", "SWARM-A 2019 (SH SSW)")
    generate_analysis_plots(P2021, 2021, OUT_DIR / "msis_activity_correlation_2021.png", "SWARM-C 2021 (NH SSW)")

if __name__ == "__main__":
    main()
