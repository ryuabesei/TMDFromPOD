"""
plot_joint_activity_Kp_filtered.py

Purpose:
    Re-analyze the relationship between density_ratio_msis, rho_obs, rho_model_real
    and geomagnetic/solar activity indices (Ap, F10.7) after filtering out
    geomagnetically disturbed periods with Kp >= 3 (Ap >= 15).

    Filter criterion:
        - Remove entire days when daily mean Ap (AP_AVG) >= 15
          (equivalent to Kp < 3 on daily average basis)
        - If 3-hour Kp is available in parquet, filter per 3-hour interval instead.

    Produces side-by-side comparison of:
        1. Time series of ratio, Ap, F10.7 (all data vs Kp-filtered)
        2. Correlation scatter plots

    Output:
        Figure/debug/joint_activity_Kp3_filtered_3years.png
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
OUT_PNG = Path("Figure/debug/joint_activity_Kp3_filtered_3years.png")

# Kp < 3 threshold in Ap units
# Official conversion: Kp=3 -> Ap=15
AP_KP3_THRESHOLD = 15.0

def get_daily_data(parquet_path: Path, doy_range: tuple[int, int] | None = None,
                   kp_filter: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    if doy_range is not None:
        df["DOY"] = df["datetime"].dt.dayofyear
        df = df[(df["DOY"] >= doy_range[0]) & (df["DOY"] <= doy_range[1])]

    df = df.dropna(subset=["density_ratio_msis", "rho_model_real", "density", "AP_AVG", "F107"])
    df["date"] = df["datetime"].dt.normalize()

    if kp_filter:
        # Compute daily mean Ap per calendar day, then mark days exceeding threshold
        daily_ap = df.groupby("date")["AP_AVG"].mean()
        disturbed_days = daily_ap[daily_ap >= AP_KP3_THRESHOLD].index
        n_removed = disturbed_days.shape[0]
        total_days = daily_ap.shape[0]
        df = df[~df["date"].isin(disturbed_days)].copy()
        print(f"  Kp<3 filter: removed {n_removed}/{total_days} days (Ap >= {AP_KP3_THRESHOLD})")
    
    daily = df.groupby("date").agg(
        ratio_med = ("density_ratio_msis", "median"),
        obs_med = ("density", "median"),
        model_med = ("rho_model_real", "median"),
        ap_mean = ("AP_AVG", "mean"),
        f107_mean = ("F107", "mean")
    ).reset_index()
    return daily


def draw_row(fig, gs_row, daily_all: pd.DataFrame, daily_filt: pd.DataFrame,
             label: str):
    """Draw one row: time series (left) + 2 correlation scatters (right)."""
    sub = gs_row.subgridspec(1, 3, width_ratios=[2.2, 1, 1], wspace=0.35)

    corr_ap_all  = daily_all["ratio_med"].corr(daily_all["ap_mean"])
    corr_ap_filt = daily_filt["ratio_med"].corr(daily_filt["ap_mean"]) if len(daily_filt) > 2 else float("nan")
    corr_f107_all  = daily_all["ratio_med"].corr(daily_all["f107_mean"])
    corr_f107_filt = daily_filt["ratio_med"].corr(daily_filt["f107_mean"]) if len(daily_filt) > 2 else float("nan")

    # ── 1. Time Series ─────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(sub[0, 0])
    ax0_ap = ax0.twinx()
    ax0_f107 = ax0.twinx()
    ax0_f107.spines["right"].set_position(("outward", 55))

    # All data (transparent blue)
    ax0.plot(daily_all["date"], daily_all["ratio_med"], color="#1f77b4", linewidth=1.2,
             alpha=0.35, linestyle="--", label="ratio (all data)")
    # Filtered data (solid blue)
    ax0.plot(daily_filt["date"], daily_filt["ratio_med"], color="#1f77b4", linewidth=2.0,
             marker="o", markersize=4, label="ratio (Kp<3 only)")
    ax0.set_ylabel("Density Ratio\n(ρ_obs/ρ_MSIS)", color="#1f77b4", fontweight="bold", fontsize=9)
    ax0.tick_params(axis="y", labelcolor="#1f77b4")
    ax0.axhline(1.0, color="gray", linestyle="--", alpha=0.4)

    p2, = ax0_ap.plot(daily_all["date"], daily_all["ap_mean"], color="#e07b39",
                      linewidth=1.2, linestyle=":", label="Daily mean Ap")
    ax0_ap.axhline(AP_KP3_THRESHOLD, color="#e07b39", linestyle="--", alpha=0.6, lw=1.0)
    ax0_ap.set_ylabel("Ap Index", color="#e07b39", fontweight="bold", fontsize=9)
    ax0_ap.tick_params(axis="y", labelcolor="#e07b39")
    ax0_ap.set_ylim(0, max(daily_all["ap_mean"].max() * 1.2, 20))

    p3, = ax0_f107.plot(daily_all["date"], daily_all["f107_mean"], color="#2ca02c",
                        linewidth=1.2, linestyle="-.", marker="^", markersize=3, label="F10.7")
    ax0_f107.set_ylabel("F10.7 [sfu]", color="#2ca02c", fontweight="bold", fontsize=9)
    ax0_f107.tick_params(axis="y", labelcolor="#2ca02c")

    ax0.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax0.grid(True, alpha=0.2)
    ax0.set_title(f"{label}", fontsize=11, fontweight="bold")

    handles = ax0.get_legend_handles_labels()[0][:2] + [p2, p3]
    labels_leg = ax0.get_legend_handles_labels()[1][:2] + [p2.get_label(), p3.get_label()]
    ax0.legend(handles, labels_leg, loc="upper left", fontsize=7.5, framealpha=0.85)

    # ── 2. Scatter vs Ap ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(sub[0, 1])
    ax1.scatter(daily_all["ap_mean"], daily_all["ratio_med"],
                color="#aec7e8", s=30, alpha=0.6, label=f"All (R={corr_ap_all:.3f})", edgecolors="none")
    ax1.scatter(daily_filt["ap_mean"], daily_filt["ratio_med"],
                color="#d62728", s=40, edgecolors="k", zorder=3, label=f"Kp<3 (R={corr_ap_filt:.3f})")
    if len(daily_filt) > 2:
        m, c = np.polyfit(daily_filt["ap_mean"], daily_filt["ratio_med"], 1)
        x_fit = np.linspace(daily_filt["ap_mean"].min(), daily_filt["ap_mean"].max(), 50)
        ax1.plot(x_fit, m*x_fit + c, color="#d62728", linestyle="--", alpha=0.7, lw=1.2)
    ax1.axvline(AP_KP3_THRESHOLD, color="gray", linestyle="--", alpha=0.5, lw=1.0)
    ax1.set_xlabel("Daily Mean Ap", fontsize=9)
    ax1.set_ylabel("Density Ratio (median)", fontsize=9)
    ax1.set_title(f"vs Ap\nAll: R={corr_ap_all:.3f} | Kp<3: R={corr_ap_filt:.3f}", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=7.5)
    ax1.grid(True, alpha=0.2)

    # ── 3. Scatter vs F10.7 ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(sub[0, 2])
    ax2.scatter(daily_all["f107_mean"], daily_all["ratio_med"],
                color="#c5b0d5", s=30, alpha=0.6, label=f"All (R={corr_f107_all:.3f})", edgecolors="none")
    ax2.scatter(daily_filt["f107_mean"], daily_filt["ratio_med"],
                color="#9467bd", s=40, edgecolors="k", zorder=3, label=f"Kp<3 (R={corr_f107_filt:.3f})")
    if len(daily_filt) > 2:
        m, c = np.polyfit(daily_filt["f107_mean"], daily_filt["ratio_med"], 1)
        x_fit = np.linspace(daily_filt["f107_mean"].min(), daily_filt["f107_mean"].max(), 50)
        ax2.plot(x_fit, m*x_fit + c, color="#9467bd", linestyle="--", alpha=0.7, lw=1.2)
    ax2.set_xlabel("Daily Mean F10.7", fontsize=9)
    ax2.set_ylabel("Density Ratio (median)", fontsize=9)
    ax2.set_title(f"vs F10.7\nAll: R={corr_f107_all:.3f} | Kp<3: R={corr_f107_filt:.3f}", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7.5)
    ax2.grid(True, alpha=0.2)


def main():
    print("Loading data for all 3 years...")
    
    configs = [
        dict(path=P2018, year=2018, doy_range=(30, 65), label="2018 NH SSW (SWARM-A)"),
        dict(path=P2019, year=2019, doy_range=None,     label="2019 SH SSW (SWARM-A)"),
        dict(path=P2021, year=2021, doy_range=None,     label="2021 NH SSW (SWARM-C)"),
    ]

    fig = plt.figure(figsize=(17, 14))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 1, hspace=0.42)

    for i, cfg in enumerate(configs):
        print(f"\n{cfg['label']}:")
        df_all  = get_daily_data(cfg["path"], cfg["doy_range"], kp_filter=False)
        df_filt = get_daily_data(cfg["path"], cfg["doy_range"], kp_filter=True)
        draw_row(fig, gs[i], df_all, df_filt, cfg["label"])

    plt.suptitle(
        "Density Ratio vs. Geomagnetic/Solar Activity\n"
        "(Gray: all data | Red/Purple: Kp < 3 only  |  Dashed orange line: Ap=15 threshold)",
        fontsize=13, fontweight="bold", y=0.99
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
