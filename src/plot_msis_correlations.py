"""
plot_msis_correlations.py

Purpose:
    Plot correlations to check:
    1. Observed density (rho_obs) vs MSIS modeled density (rho_model_real)
       to see how well MSIS captures the general variations.
    2. density_ratio_msis vs Ap index (AP_AVG)
       to see how much residual variation is correlated with geomagnetic activity.

    For both 2018 NH SSW and 2019 SH SSW events (using SWARM-A).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
P2018 = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
OUT_DIR = Path("Figure/debug")

def process_and_plot(parquet_path: Path, year: int, doy_range: tuple[int, int], out_png: Path):
    print(f"Processing {year} data from {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["DOY"] = df["datetime"].dt.dayofyear
    
    # Filter by DOY
    df = df[(df["DOY"] >= doy_range[0]) & (df["DOY"] <= doy_range[1])]
    df = df.dropna(subset=["density", "rho_model_real", "density_ratio_msis"])
    
    # Check for Ap column
    ap_col = "AP_AVG" if "AP_AVG" in df.columns else None
    
    # Sample data for scatter
    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
    
    # Daily averages
    daily = df.groupby("DOY").agg(
        obs_med = ("density", "median"),
        model_med = ("rho_model_real", "median"),
        ratio_med = ("density_ratio_msis", "median"),
        ap_mean = (ap_col, "mean") if ap_col else ("DOY", "count")
    ).reset_index()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("white")
    
    # --- 1. Scatter plot: rho_obs vs rho_model_real (All points) ---
    ax = axes[0, 0]
    ax.scatter(sample_df["rho_model_real"] * 1e13, sample_df["density"] * 1e13, alpha=0.3, color="#1f77b4", s=5)
    # 1:1 line
    max_val = max(sample_df["rho_model_real"].max(), sample_df["density"].max()) * 1e13
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="1:1 Line")
    
    corr_raw = df["density"].corr(df["rho_model_real"])
    ax.set_title(f"Observed vs MSIS Density (All Obs)\nCorr = {corr_raw:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("MSIS Density [10^-13 kg/m^3]")
    ax.set_ylabel("Observed Density [10^-13 kg/m^3]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    # --- 2. Scatter plot: rho_obs vs rho_model_real (Daily Median) ---
    ax = axes[0, 1]
    ax.scatter(daily["model_med"] * 1e13, daily["obs_med"] * 1e13, color="#2ca02c", s=50, edgecolors="k", zorder=3)
    # 1:1 line
    max_val_d = max(daily["model_med"].max(), daily["obs_med"].max()) * 1e13
    ax.plot([0, max_val_d], [0, max_val_d], color="red", linestyle="--")
    
    corr_daily = daily["obs_med"].corr(daily["model_med"])
    ax.set_title(f"Observed vs MSIS (Daily Median)\nCorr = {corr_daily:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("MSIS Density [10^-13 kg/m^3]")
    ax.set_ylabel("Observed Density [10^-13 kg/m^3]")
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- 3. Scatter plot: density_ratio_msis vs Ap (All points) ---
    ax = axes[1, 0]
    if ap_col:
        ax.scatter(sample_df[ap_col], sample_df["density_ratio_msis"], alpha=0.2, color="#ff7f0e", s=5)
        corr_ap_raw = df["density_ratio_msis"].corr(df[ap_col])
        ax.set_title(f"Density Ratio vs Ap Index (All Obs)\nCorr = {corr_ap_raw:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Ap Index (3-hour / daily)")
    else:
        ax.text(0.5, 0.5, "Ap data not found", ha="center", va="center")
    ax.set_ylabel("density_ratio_msis (rho_obs / rho_MSIS)")
    ax.axhline(1.0, color="gray", linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- 4. Scatter plot: density_ratio_msis vs Ap (Daily Median) ---
    ax = axes[1, 1]
    if ap_col:
        ax.scatter(daily["ap_mean"], daily["ratio_med"], color="#d62728", s=50, edgecolors="k", zorder=3)
        corr_ap_daily = daily["ratio_med"].corr(daily["ap_mean"])
        ax.set_title(f"Density Ratio vs Ap (Daily Median)\nCorr = {corr_ap_daily:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Daily Mean Ap Index")
    else:
        ax.text(0.5, 0.5, "Ap data not found", ha="center", va="center")
    ax.set_ylabel("density_ratio_msis (rho_obs / rho_MSIS)")
    ax.axhline(1.0, color="gray", linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"SWARM-A MSIS Correlation Analysis ({year})\nDOY {doy_range[0]} - {doy_range[1]}", 
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_png}")
    plt.close()

def main():
    # 2018 range (DOY 30-65)
    process_and_plot(P2018, 2018, (30, 65), OUT_DIR / "msis_correlation_2018.png")
    
    # 2019 range (DOY 232-266, which is Aug 20 to Sep 23)
    process_and_plot(P2019, 2019, (232, 266), OUT_DIR / "msis_correlation_2019.png")

if __name__ == "__main__":
    main()
