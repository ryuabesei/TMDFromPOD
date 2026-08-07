r"""
plot_ratio_detrend_Kp3_filtered_ap_3years.py

Purpose:
    Same layout as ratio_detrend_temp_ap_combined_3years.png but with
    Kp > 3 (AP_AVG >= 15) days EXCLUDED from the regression and residual
    calculation.

    Steps per event:
        1. Load data, apply Kp < 3 filter (AP_AVG < 15).
        2. Compute daily-median ratio and Ap from Kp-filtered data only.
        3. Fit linear regression: y = a * Ap + b (on filtered data).
        4. Compute residual: Dy_Ap = y_i - (a * Ap_i + b) for filtered days.
        5. Plot scatter (ratio vs Ap) with regression line for each event.
        6. Plot 1D time series (same style as reference figure) where
           Kp-removed days are connected through (NOT left as NaN gaps).

Output:
    Figure/Ap_removal/ratio_detrend_Kp3_scatter_3years.png   (scatter panels)
    Figure/Ap_removal/ratio_detrend_Kp3_temp_ap_combined_3years.png (1D combined)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# Kp < 3 threshold (Ap units)
AP_KP3 = 15.0  # Kp=3 <=> Ap=15

# File Paths
P2018 = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")

COSMIC_2018 = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
ERA5_2019_DIR = Path("data/SSW2019/ERA5")
ERA5_2021_DIR = Path("data/SSW2021/ERA5")

OUT_DIR = Path("Figure/Ap_removal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Event Configurations
EVENTS = [
    dict(
        year=2018,
        label="2018 NH SSW (SWARM-A)",
        sat="SWARM-A",
        parquet=P2018,
        date_start=pd.Timestamp("2018-01-30", tz="UTC"),
        date_end=pd.Timestamp("2018-03-06 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2018-01-30", tz="UTC"), pd.Timestamp("2018-02-09", tz="UTC")),
            (pd.Timestamp("2018-03-02", tz="UTC"), pd.Timestamp("2018-03-06", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2018-02-12", tz="UTC"),
        ssw_peak_label="SSW Peak (DOY 43)",
        temp_type="cosmic",
        temp_file=COSMIC_2018,
        temp_label="T(10 hPa) 60-90N",
        xlabel="DOY (2018)",
        x_date_interval=4,
    ),
    dict(
        year=2019,
        label="2019 SH SSW (SWARM-A)",
        sat="SWARM-A",
        parquet=P2019,
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"),
        ssw_peak_label="SSW Peak (Sep 19)",
        temp_type="era5",
        temp_dir=ERA5_2019_DIR,
        lat_range=(-90.0, -60.0),
        temp_label="T(10 hPa) 60-90S",
        xlabel="Date (2019)",
        x_date_interval=4,
    ),
    dict(
        year=2021,
        label="2021 NH SSW (SWARM-C)",
        sat="SWARM-C",
        parquet=P2021,
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end=pd.Timestamp("2021-02-05 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29", tz="UTC")),
            (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2021-01-04", tz="UTC"),
        ssw_peak_label="SSW Peak (Jan 04)",
        temp_type="era5",
        temp_dir=ERA5_2021_DIR,
        lat_range=(60.0, 90.0),
        temp_label="T(10 hPa) 60-90N",
        xlabel="Date (2020-2021)",
        x_date_interval=5,
    ),
]


def load_event_df(ev: dict) -> pd.DataFrame:
    df = pd.read_parquet(ev["parquet"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for cname in ["lat", "latitude", "geod_lat"]:
        if cname in df.columns and cname != "lat":
            df = df.rename(columns={cname: "lat"})
            break
    df = df.dropna(subset=["datetime", "lat", "density_ratio_msis", "AP_AVG"])
    df = df[(df["datetime"] >= ev["date_start"]) & (df["datetime"] <= ev["date_end"])].copy()
    df["date"] = df["datetime"].dt.normalize()
    return df


def load_temp(ev: dict) -> pd.Series:
    if ev["temp_type"] == "cosmic":
        df = pd.read_csv(ev["temp_file"], parse_dates=["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["date"] = df["datetime"].dt.normalize()
        df = df[(df["datetime"] >= ev["date_start"]) & (df["datetime"] <= ev["date_end"])]
        return df.groupby("date")["T10_K"].mean().sort_index()
    else:
        files = sorted(ev["temp_dir"].glob("*.nc"))
        if not files:
            return pd.Series(dtype=float)
        series_list = []
        lat_lo, lat_hi = min(ev["lat_range"]), max(ev["lat_range"])
        for fp in files:
            with xr.open_dataset(fp) as ds:
                ds_sub = ds.sel(latitude=slice(lat_hi, lat_lo))
                weights = np.cos(np.deg2rad(ds_sub["latitude"]))
                w_temp = ds_sub["t"].weighted(weights)
                t_avg = w_temp.mean(dim=["latitude", "longitude"]).squeeze()
                times = pd.to_datetime(t_avg["valid_time"].values)
                vals = t_avg.values.ravel()
                s = pd.Series(vals, index=times)
                s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
                series_list.append(s)
        combined = pd.concat(series_list).sort_index().resample("D").mean()
        combined.index = combined.index.normalize()
        combined = combined[(combined.index >= ev["date_start"]) & (combined.index <= ev["date_end"])]
        return combined


def main() -> None:
    processed_events = []

    for ev in EVENTS:
        print(f"Processing {ev['label']}...")
        df_all = load_event_df(ev)

        # Full daily index covering the event window
        all_dates = pd.date_range(
            ev["date_start"].normalize(),
            ev["date_end"].normalize(),
            freq="D",
            tz="UTC",
        )

        # Daily aggregation from ALL data (for Ap series + kp_mask)
        daily_all = df_all.groupby("date").agg(
            ratio=("density_ratio_msis", "median"),
            ap=("AP_AVG", "mean"),
        ).reset_index()

        # Kp filter: keep only days where AP_AVG < 15 (Kp < 3)
        # Use .tolist() to keep Timestamps (not numpy.datetime64) for isin() to work
        kp_ok_dates = set(daily_all.loc[daily_all["ap"] < AP_KP3, "date"].tolist())
        df_filt = df_all[df_all["date"].isin(kp_ok_dates)].copy()

        # Daily aggregation from FILTERED data
        daily_filt = df_filt.groupby("date").agg(
            ratio=("density_ratio_msis", "median"),
            ap=("AP_AVG", "mean"),
        ).reset_index()

        # Regression on filtered data only
        x = daily_filt["ap"].values
        y = daily_filt["ratio"].values
        p = np.polyfit(x, y, 1)
        slope, intercept = p[0], p[1]
        r = float(np.corrcoef(x, y)[0, 1])

        # Residual for filtered days
        daily_filt = daily_filt.copy()
        daily_filt["delta_ap"] = daily_filt["ratio"] - (slope * daily_filt["ap"] + intercept)

        # Series on FULL date range (NaN on Kp-removed days)
        s_delta_filt = daily_filt.set_index("date")["delta_ap"].reindex(all_dates)
        s_ap_all = daily_all.set_index("date")["ap"].reindex(all_dates)

        # Connected line: interpolate linearly across removed gaps
        s_delta_connected = s_delta_filt.interpolate(method="time", limit_area="inside")

        # Boolean mask: True = Kp-removed day (ratio NaN in filtered series)
        s_ratio_filt = daily_filt.set_index("date")["ratio"].reindex(all_dates)
        kp_removed_mask = s_ratio_filt.isna()

        s_temp = load_temp(ev)

        processed_events.append({
            "ev": ev,
            "daily_filt": daily_filt,
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "s_delta_filt": s_delta_filt,
            "s_delta_connected": s_delta_connected,
            "s_ap_all": s_ap_all,
            "kp_removed_mask": kp_removed_mask,
            "s_temp": s_temp,
            "all_dates": all_dates,
        })

    # =========================================================================
    # Figure 1: Scatter plots (ratio vs Ap) with regression line
    # =========================================================================
    out_scatter = OUT_DIR / "ratio_detrend_Kp3_scatter_3years.png"
    print(f"\nGenerating scatter plot: {out_scatter}...")

    fig_sc, axes_sc = plt.subplots(1, 3, figsize=(15, 5))
    fig_sc.suptitle(
        "Density Ratio $y_i$ vs Ap Index  (Kp < 3 filtered)  —  Linear Regression for Ap Detrending",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ci, item in enumerate(processed_events):
        ev = item["ev"]
        df_d = item["daily_filt"]
        slope = item["slope"]
        intercept = item["intercept"]
        r = item["r"]

        ax = axes_sc[ci]
        x_data = df_d["ap"].values
        y_data = df_d["ratio"].values

        ax.scatter(x_data, y_data, color="#1f77b4", s=40, alpha=0.75, zorder=3, label="Obs (Kp<3)")

        x_fit = np.linspace(max(0, x_data.min() - 1), x_data.max() + 1, 200)
        ax.plot(x_fit, slope * x_fit + intercept, color="#e74c3c", lw=2.0, zorder=4,
                label=f"y = {slope:.5f}*Ap + {intercept:.4f}")

        ax.set_xlabel("Daily mean Ap index", fontweight="bold")
        ax.set_ylabel("Daily median $y_i$", fontweight="bold")
        ax.set_title(f"{ev['label']}\ncorr(y, Ap) = {r:.3f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.5)

        ann = f"a={slope:.5f}\nb={intercept:.5f}\nr={r:.3f}"
        ax.text(0.97, 0.05, ann, transform=ax.transAxes, fontsize=8.5,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    fig_sc.tight_layout()
    fig_sc.savefig(out_scatter, dpi=200, bbox_inches="tight")
    plt.close(fig_sc)
    print(f"  Saved: {out_scatter}")

    # =========================================================================
    # Figure 2: 1D Combined time-series (same style as reference figure)
    #   Blue:    Dy_Ap  (left axis)
    #   Crimson: T(10 hPa) (right axis 1)
    #   Orange:  Ap index  (right axis 2)
    #   Kp-removed days: connected with interpolated line, not NaN gap
    # =========================================================================
    out_comb = OUT_DIR / "ratio_detrend_Kp3_temp_ap_combined_3years.png"
    print(f"Generating combined 1D plot: {out_comb}...")

    fig = plt.figure(figsize=(15, 13))
    fig.suptitle(
        "Thermospheric Density Ratio Residual ($\\Delta y_{Ap}$), Stratospheric Temp T(10 hPa), & Ap Index\n"
        "Blue line: $\\Delta y_{Ap} = y_i - (a\\cdot Ap_i + b)$ (Left Y-axis)  |  "
        "Crimson line: T(10 hPa) (Right Y-axis 1)  |  Orange line: Ap index (Right Y-axis 2)\n"
        "[Kp > 3 days removed from regression; gaps connected by interpolated line]",
        fontsize=11, fontweight="bold", y=1.005,
    )

    gs_outer = gridspec.GridSpec(3, 1, hspace=0.45)

    for i, item in enumerate(processed_events):
        ev = item["ev"]
        slope = item["slope"]
        intercept = item["intercept"]
        r = item["r"]

        s_delta_conn = item["s_delta_connected"]   # Dy_Ap gaps interpolated
        s_delta_filt = item["s_delta_filt"]        # Dy_Ap only on filtered days
        s_ap = item["s_ap_all"]                    # Ap for ALL days
        s_temp = item["s_temp"]
        kp_mask = item["kp_removed_mask"]          # True = Kp>3 removed

        ax1d = fig.add_subplot(gs_outer[i])
        ax_t = ax1d.twinx()
        ax_a = ax1d.twinx()
        ax_a.spines["right"].set_position(("axes", 1.07))
        ax_a.spines["right"].set_visible(True)

        # Reference period shading
        for r_start, r_end in ev["ref_dates"]:
            ax1d.axvspan(r_start, r_end, color="#008080", alpha=0.10, zorder=0)

        # Blue: connected interpolated line (thin dashed) over ALL days
        dates_conn = s_delta_conn.index.to_pydatetime()
        vals_conn = s_delta_conn.values
        ax1d.plot(dates_conn, vals_conn, "-", color="#1f77b4", lw=1.2,
                  alpha=0.45, zorder=2)

        # Blue: solid markers only on actual Kp<3 days
        s_filt_valid = s_delta_filt.dropna()
        dates_filt = s_filt_valid.index.to_pydatetime()
        vals_filt = s_filt_valid.values
        l1 = ax1d.plot(dates_filt, vals_filt, "o-", color="#1f77b4", lw=1.8,
                       ms=3.5, zorder=3, label="$\\Delta y_{Ap} = y_i - y_{pred}(Ap_i)$")

        ax1d.axhline(0.0, color="gray", ls=":", lw=1.0)

        # Crimson: T(10 hPa)
        l2 = []
        if len(s_temp) > 0:
            l2 = ax_t.plot(s_temp.index, s_temp.values, "^-",
                           color="#dc143c", lw=1.8, ms=3.5,
                           label=ev["temp_label"])
            ax_t.set_ylabel(f"{ev['temp_label']} [K]", color="#dc143c",
                            fontweight="bold", fontsize=9)
            ax_t.tick_params(axis="y", labelcolor="#dc143c", labelsize=8)

        # Orange: Daily-mean Ap (all days)
        l3 = ax_a.plot(s_ap.index, s_ap.values, "s-", color="#e67e22",
                       lw=1.2, ms=3.5, alpha=0.85, label="Daily mean Ap")
        # Mark Kp>3 days with a red X marker on the Ap line
        kp_bad = s_ap[kp_mask]
        l4 = []
        if len(kp_bad) > 0:
            l4 = ax_a.plot(kp_bad.index, kp_bad.values, "x", color="#c0392b",
                           ms=7, mew=2, zorder=5,
                           label=f"Kp>=3 (Ap>={AP_KP3:.0f}, removed)")
        # Threshold line
        ax_a.axhline(AP_KP3, color="#c0392b", lw=1.0, ls=":", alpha=0.7)
        ax_a.set_ylabel("Ap Index", color="#e67e22", fontweight="bold", fontsize=9)
        ax_a.tick_params(axis="y", labelcolor="#e67e22", labelsize=8)

        # SSW peak line
        ax1d.axvline(ev["ssw_peak"], color="red", ls="--", lw=1.5, zorder=6)
        y_lim_now = ax1d.get_ylim()
        ax1d.text(
            ev["ssw_peak"],
            y_lim_now[0] + 0.05 * (y_lim_now[1] - y_lim_now[0]),
            f" {ev['ssw_peak_label']}",
            color="red", fontweight="bold", fontsize=7.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85),
        )

        # Annotation box (top-left)
        ann_text = (
            f"slope a={slope:.5f}  intercept b={intercept:.5f}  "
            f"corr(y,Ap)={r:.3f}"
        )
        ax1d.text(
            0.015, 0.88, ann_text, transform=ax1d.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
        )

        # Y-axis labels
        ax1d.set_ylabel("$\\Delta y_{Ap}$ (Density Ratio Residual)",
                        color="#1f77b4", fontweight="bold", fontsize=9)
        ax1d.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8)
        ax1d.grid(True, linestyle=":", alpha=0.4)
        ax1d.set_title(
            f"{ev['label']}  [Kp < 3 filtered]",
            fontsize=11, fontweight="bold", loc="center", pad=10,
        )

        # X-axis
        ax1d.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax1d.xaxis.set_major_locator(mdates.DayLocator(interval=ev["x_date_interval"]))
        ax1d.set_xlabel(ev["xlabel"], fontweight="bold", fontsize=9)

        # Legend
        lines_main = l1
        labels_main = [l.get_label() for l in lines_main]
        lines_t_labels = [(l, l.get_label()) for l in l2]
        lines_a = l3 + l4
        labels_a = [l.get_label() for l in lines_a]

        all_lines = lines_main + l2 + lines_a
        all_labels = labels_main + [ll for _, ll in lines_t_labels] + labels_a
        ax1d.legend(all_lines, all_labels, loc="upper right", fontsize=8,
                    framealpha=0.9, ncol=2)

    plt.savefig(out_comb, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_comb}")
    print("Done.")


if __name__ == "__main__":
    main()
