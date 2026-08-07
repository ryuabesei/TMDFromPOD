"""
plot_obs_vs_msis_density_and_ap_3years.py

Purpose:
    Plot Observed Density (rho_obs), MSIS 2.1 Density (rho_MSIS), and Ap Index
    SIMULTANEOUSLY on the SAME graph panel for 3 major SSW events:
      - 2018 NH SSW (SWARM-A, Row 1)
      - 2019 SH SSW (SWARM-A, Row 2)
      - 2021 NH SSW (SWARM-C, Row 3)

Layout:
    3 rows x 1 column (wide aspect ratio, stacked vertically)
    Left Y-axis : rho_obs (Blue solid line) & rho_MSIS (Green solid line) [10^-13 kg/m^3]
    Right Y-axis: Daily mean Ap index (Orange solid line with square markers)

Output:
    Figure/summary/obs_vs_msis_density_and_ap_3years.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─── Paths ───────────────────────────────────────────────────────────────────
P2018 = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")

OUT_DIR = Path("Figure/summary")
OUT_PNG = OUT_DIR / "obs_vs_msis_density_and_ap_3years.png"

AP_KP3 = 15.0

# ─── Event Settings ──────────────────────────────────────────────────────────
EVENTS = [
    dict(
        label="2018 NH SSW", sat="SWARM-A",
        parquet=P2018, mode="doy",
        doy_start=30, doy_end=65,
        ref_doy=[(30, 40), (61, 65)],
        ssw_peak_doy=43, year=2018,
    ),
    dict(
        label="2019 SH SSW", sat="SWARM-A",
        parquet=P2019, mode="date",
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"), year=2019,
    ),
    dict(
        label="2021 NH SSW", sat="SWARM-C",
        parquet=P2021, mode="date",
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end=pd.Timestamp("2021-02-05", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29", tz="UTC")),
            (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2021-01-04", tz="UTC"), year=2021,
    ),
]


# ─── Loaders ─────────────────────────────────────────────────────────────────
def load_event_df(ev: dict) -> pd.DataFrame:
    df = pd.read_parquet(ev["parquet"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for cname in ["lat", "latitude", "geod_lat"]:
        if cname in df.columns and cname != "lat":
            df = df.rename(columns={cname: "lat"})
            break
    df = df.dropna(subset=["datetime", "density", "rho_model_real", "AP_AVG"])

    if ev["mode"] == "doy":
        df["key"] = df["datetime"].dt.dayofyear
        df = df[(df["key"] >= ev["doy_start"]) & (df["key"] <= ev["doy_end"])]
    else:
        df["key"] = df["datetime"].dt.normalize()
        df = df[(df["datetime"] >= ev["date_start"]) &
                (df["datetime"] <= ev["date_end"] + pd.Timedelta(hours=23, minutes=59))]
    return df


# ─── Main Script ─────────────────────────────────────────────────────────────
def main() -> None:
    all_data = []

    for ev in EVENTS:
        print(f"Processing {ev['label']} ({ev['sat']})...")
        df = load_event_df(ev)

        daily = df.groupby("key").agg(
            rho_obs=("density", "median"),
            rho_msis=("rho_model_real", "median"),
            ap=("AP_AVG", "mean"),
        ).reset_index()

        s_obs  = pd.Series(daily["rho_obs"].values  * 1e13, index=daily["key"])
        s_msis = pd.Series(daily["rho_msis"].values * 1e13, index=daily["key"])
        s_ap   = pd.Series(daily["ap"].values,              index=daily["key"])

        all_data.append({
            "ev": ev,
            "s_obs": s_obs,
            "s_msis": s_msis,
            "s_ap": s_ap,
        })

    # Create figure: 3 rows x 1 column (wide aspect ratio)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    fig.suptitle(
        r"Observed Density ($\rho_{obs}$), MSIS 2.1 Model Density ($\rho_{MSIS}$), & Ap Index" "\n"
        r"Blue line: $\rho_{obs}$  |  Green line: $\rho_{MSIS}$ (Left Y-axis, $10^{-13}\ \mathrm{kg/m}^3$)  |  Orange line: Ap index (Right Y-axis)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    for row, data in enumerate(all_data):
        ev = data["ev"]
        ax_d = axes[row]
        ax_a = ax_d.twinx()

        s_obs  = data["s_obs"]
        s_msis = data["s_msis"]
        s_ap   = data["s_ap"]

        # Ref period shading
        if ev["mode"] == "doy":
            for lo, hi in ev["ref_doy"]:
                ax_d.axvspan(lo, hi, color="lightblue", alpha=0.22, lw=0)
        else:
            for s, e in ev["ref_dates"]:
                ax_d.axvspan(s, e, color="lightblue", alpha=0.22, lw=0)

        # ── 1. Ap Line Plot (Right Y-axis, Orange line with square markers) ────
        ax_a.plot(s_ap.index, s_ap.values,
                  color="#e07b39", lw=1.8, marker="s", ms=4.5, alpha=0.9,
                  zorder=3, label="Daily mean Ap")
        ax_a.axhline(AP_KP3, color="#e07b39", lw=1.0, ls=":", alpha=0.7, zorder=2)
        ax_a.set_ylabel("Ap Index", fontsize=10, fontweight="bold", color="#e07b39")
        ax_a.tick_params(axis="y", labelcolor="#e07b39", labelsize=8.5)
        ap_max = s_ap.max() if not s_ap.empty else AP_KP3
        ax_a.set_ylim(0, max(ap_max * 1.25, AP_KP3 * 1.3))
        ax_a.spines["right"].set_edgecolor("#e07b39")

        # ── 2. Density Lines (Left Y-axis, Blue & Green lines) ─────────────────
        ax_d.plot(s_obs.index, s_obs.values,
                  color="#1f77b4", lw=2.2, marker="o", ms=4.5,
                  zorder=5, label=r"Observed Density ($\rho_{obs}$)")
        ax_d.plot(s_msis.index, s_msis.values,
                  color="#2ca02c", lw=2.0, marker="^", ms=4.5,
                  zorder=4, label=r"MSIS 2.1 Model Density ($\rho_{MSIS}$)")
        ax_d.set_ylabel(r"Density [$10^{-13}\ \mathrm{kg/m}^3$]", fontsize=10, fontweight="bold", color="black")
        ax_d.tick_params(axis="y", labelcolor="black", labelsize=8.5)

        # SSW Peak line
        pk = ev.get("ssw_peak_doy") or ev.get("ssw_peak")
        if pk is not None:
            ax_d.axvline(pk, color="red", lw=1.8, ls="--", alpha=0.8, zorder=7)
            ax_d.text(pk, 0.95, " SSW Peak", transform=ax_d.get_xaxis_transform(),
                      fontsize=8.5, color="red", fontweight="bold", alpha=0.9, va="top")

        # Correlation between rho_obs and rho_msis
        r_dens = float(np.corrcoef(s_obs.values, s_msis.values)[0, 1])
        ax_d.text(
            0.015, 0.95,
            f"corr(rho_obs, rho_MSIS) = {r_dens:.3f}",
            transform=ax_d.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

        ax_d.set_title(f"{ev['label']}  ({ev['sat']})", fontsize=11, fontweight="bold")
        ax_d.grid(True, ls=":", alpha=0.40)

        if ev["mode"] == "date":
            ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax_d.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            plt.setp(ax_d.xaxis.get_majorticklabels(), rotation=15, ha="right")
        ax_d.set_xlabel(f"{'DOY' if ev['mode']=='doy' else 'Date'} ({ev['year']})", fontsize=9.5)

        # Combine legends for each panel
        lines1, labels1 = ax_d.get_legend_handles_labels()
        lines2, labels2 = ax_a.get_legend_handles_labels()
        ax_d.legend(lines1 + lines2, labels1 + labels2,
                    loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure: {OUT_PNG}")


if __name__ == "__main__":
    main()
