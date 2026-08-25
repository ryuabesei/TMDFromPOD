"""Plot the 2020/2021 NH SSW GRACE-FO ratio using the three geomagnetic drivers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_2d_detrend_only_3years import compute_2d_grid, load_temp
from run_high_cadence_geomagnetic_analysis import ROOT, attach_driver


INPUT = ROOT / "standardizeddata/karman_pretrained/2021/grace_fo_2021_karman.parquet"
OUT = ROOT / "reports/2020_2021_NH_GRACE_FO_three_index_linear_detrending"
METHODS = {"Ap": (24.0, 0.0), "ap": (24.0, 0.0), "ap60": (24.0, 1.0)}
FILE_TAG = {"Ap": "daily_Ap", "ap": "ap_3hour", "ap60": "ap60_1hour"}
START = pd.Timestamp("2020-12-25", tz="UTC")
END = pd.Timestamp("2021-02-05 23:59:59", tz="UTC")
ONSET = pd.Timestamp("2021-01-04", tz="UTC")
SSW_END = pd.Timestamp("2021-01-16 23:59:59", tz="UTC")


def main() -> None:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(INPUT)
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
    physical = (
        raw["datetime"].between(START, END)
        & np.isfinite(raw["density_ratio_msis"])
        & np.isfinite(raw["density"])
        & np.isfinite(raw["rho_model_real"])
        & (raw["density"] > 0)
        & (raw["rho_model_real"] > 0)
    )
    raw = raw.loc[physical].copy()
    raw["block"] = raw["datetime"].dt.floor("90min")
    blocks = raw.groupby("block", as_index=False).agg(
        datetime=("datetime", "median"),
        density_ratio_msis=("density_ratio_msis", "median"),
    )
    temp = load_temp(dict(
        temp_type="era5", temp_dir=ROOT / "data/SSW2021/ERA5",
        lat_range=(60.0, 90.0), date_start=START, date_end=END,
    ))
    rows = []
    for index, (window, lag) in METHODS.items():
        fit_data = blocks.copy()
        fit_data["driver"] = attach_driver(fit_data, "2021_NH", index, window, lag)
        fit = fit_data.dropna(subset=["driver", "density_ratio_msis"])
        slope, intercept = np.polyfit(fit["driver"], fit["density_ratio_msis"], 1)
        corr = float(np.corrcoef(fit["driver"], fit["density_ratio_msis"])[0, 1])

        plotted = raw.copy()
        plotted["driver"] = attach_driver(plotted, "2021_NH", index, window, lag)
        plotted["residual"] = plotted["density_ratio_msis"] - (
            intercept + slope * plotted["driver"]
        )
        plotted["date"] = plotted["datetime"].dt.normalize()
        daily = plotted.groupby("date").agg(
            residual=("residual", "median"), driver=("driver", "mean")
        )
        pre = daily.loc[daily.index < ONSET, "residual"]
        during = daily.loc[daily.index.to_series().between(ONSET, SSW_END), "residual"]
        rows.append({
            "event": "2021_NH", "satellite": "GRACE-FO", "index": index,
            "window_h": window, "lag_h": lag, "slope": slope,
            "intercept": intercept, "corr": corr, "pre_days": len(pre),
            "ssw_days": len(during), "residual_pre_median": pre.median(),
            "residual_ssw_median": during.median(),
            "ssw_minus_pre_residual_pp": 100 * (during.median() - pre.median()),
        })

        edges = pd.date_range(START.normalize(), END.normalize() + pd.Timedelta(days=1), freq="D")
        lat_edges = np.arange(-60, 63, 3)
        z = compute_2d_grid(plotted, "residual", edges, lat_edges)
        fig = plt.figure(figsize=(14, 8), layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[1, .045], height_ratios=[1, 1.55])
        ax = fig.add_subplot(gs[0, 0]); spacer = fig.add_subplot(gs[0, 1]); spacer.axis("off")
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax); cax = fig.add_subplot(gs[1, 1])
        at = ax.twinx(); ag = ax.twinx(); ag.spines["right"].set_position(("axes", 1.08))
        l1 = ax.plot(daily.index, daily.residual, "o-", lw=2, ms=3.5, color="#1f77b4", label=f"ratio residual after {index}")
        l2 = at.plot(temp.index, temp.values, "^-", lw=2, ms=3.5, color="#dc143c", label="T(10 hPa) polar-cap mean")
        if index == "Ap":
            driver_label = "daily Ap (same UTC day; baseline)"
            subtitle = "same-UTC-day daily Ap; linear detrending"
        else:
            native = {"ap": "3-hour ap", "ap60": "1-hour ap60"}[index]
            driver_label = f"{native}, {window:g} h causal trailing, lag {lag:g} h"
            subtitle = f"{native}; {window:g} h causal trailing window; lag {lag:g} h"
        l3 = ag.plot(daily.index, daily.driver, "s-", lw=1.4, ms=3.5, color="#e67e22", label=driver_label)
        ax.axhline(0, color=".4", ls=":"); ax.axvline(ONSET, color="red", ls="--")
        ax.set_ylabel("Density-ratio residual", color="#1f77b4", fontweight="bold")
        at.set_ylabel("T(10 hPa) [K]", color="#dc143c", fontweight="bold")
        ag.set_ylabel(index, color="#e67e22", fontweight="bold")
        ax.tick_params(axis="y", labelcolor="#1f77b4"); at.tick_params(axis="y", labelcolor="#dc143c"); ag.tick_params(axis="y", labelcolor="#e67e22")
        ax.text(.012, .96, f"slope a={slope:.5f}  intercept b={intercept:.5f}  corr(R,G)={corr:.3f}", transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round,pad=.25", fc="white", ec=".25", alpha=.88))
        lines = l1 + l2 + l3; ax.legend(lines, [x.get_label() for x in lines], loc="upper right", fontsize=8)
        ax.grid(ls=":", alpha=.35); plt.setp(ax.get_xticklabels(), visible=False)
        im = ax2.pcolormesh(mdates.date2num(edges), lat_edges, z, cmap="RdBu_r", vmin=-.2, vmax=.2, shading="flat")
        ax2.axvline(ONSET, color="red", ls="--"); ax2.set_ylim(-60, 60); ax2.set_xlim(START, END)
        ax2.set_ylabel("Geographic latitude [deg]", fontweight="bold"); ax2.set_xlabel("Date (UTC)", fontweight="bold")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); ax2.grid(ls=":", alpha=.25)
        fig.colorbar(im, cax=cax, label="density-ratio residual")
        fig.suptitle(f"2020/2021 NH SSW GRACE-FO — standardized MSIS ratio linearly detrended with {index}\n{subtitle}", fontsize=14, fontweight="bold")
        fig.savefig(figures / f"SSW_2021_NH_GRACE_FO_{FILE_TAG[index]}.png", dpi=190)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "linear_detrending_comparison.csv", index=False)
    print(f"Generated {len(rows)} GRACE-FO figures in {figures}")


if __name__ == "__main__":
    main()
