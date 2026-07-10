"""
plot_obs_vs_model_3years.py

Purpose:
    Plot observed density (rho_obs) and MSIS modeled density (rho_model_real)
    directly together in a single figure to visualize the absolute discrepancy
    and how MSIS responds compared to observations during SSW/storm events.
    
    Layout:
        3 rows (2018 NH SSW / 2019 SH SSW / 2021 NH SSW)
        Each row has a single panel with rho_obs and rho_model_real plotted together.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Parquet paths
P2018 = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")
OUT_PNG = Path("Figure/debug/obs_vs_model_density_3years.png")

def get_daily_density(parquet_path: Path, doy_range: tuple[int, int] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if doy_range is not None:
        df["DOY"] = df["datetime"].dt.dayofyear
        df = df[(df["DOY"] >= doy_range[0]) & (df["DOY"] <= doy_range[1])]
    df = df.dropna(subset=["density", "rho_model_real"])
    df["date"] = df["datetime"].dt.normalize()
    
    daily = df.groupby("date").agg(
        obs_med = ("density", "median"),
        model_med = ("rho_model_real", "median")
    ).reset_index()
    return daily

def main():
    print("Loading density data...")
    df18 = get_daily_density(P2018, (30, 65))
    df19 = get_daily_density(P2019, None)
    df2021 = get_daily_density(P2021, None)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.patch.set_facecolor("white")
    
    events = [
        {"df": df18, "label": "2018 NH SSW (SWARM-A)", "ax": axes[0]},
        {"df": df19, "label": "2019 SH SSW (SWARM-A)", "ax": axes[1]},
        {"df": df2021, "label": "2021 NH SSW (SWARM-C)", "ax": axes[2]}
    ]

    for item in events:
        df = item["df"]
        label = item["label"]
        ax = item["ax"]

        # Convert to 10^-13 kg/m^3 for readability
        scale = 1e13
        obs_scaled = df["obs_med"] * scale
        model_scaled = df["model_med"] * scale

        # Plot Observed (solid line, blue)
        ax.plot(df["date"], obs_scaled, color="#1f77b4", linewidth=2.0, marker="o", label="Observed Density (rho_obs)")
        
        # Plot MSIS (dashed line, red)
        ax.plot(df["date"], model_scaled, color="#d62728", linewidth=2.0, marker="s", markersize=4, linestyle="--", 
                label="MSIS Real Condition (rho_model_real)")

        # Axes & labels
        ax.set_ylabel("Density [10$^{-13}$ kg/m$^3$]", fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        # X-axis formatting
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    plt.suptitle("Observed Thermospheric Density vs MSIS Modeled Density", 
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    plt.close()

if __name__ == "__main__":
    main()
