r"""
plot_2d_obs_minus_msis_all_ssw.py

Purpose:
    Plot the absolute difference of observed thermospheric density minus
    NRLMSIS model density at real conditions:

        delta_rho = rho_obs - rho_MSISreal   [kg/m^3]

    For 2018/2019/2021: uses rho_model_real column directly.
    For 2024/2026: reconstructs rho_MSISreal = density / density_ratio_msis.

    Events (all SWARM-A unless noted):
      1. 2018 NH SSW  (SWARM-A)
      2. 2019 SH SSW  (SWARM-A)
      3. 2021 NH SSW  (SWARM-C)
      4. 2024 NH SSW  (SWARM-A)
      5. 2024 SH SSW  (SWARM-A)
      6. 2026 NH SSW  (SWARM-A)

Output:
    Figure/Ap_removal/2D_obs_minus_msis_2018_NH.png
    Figure/Ap_removal/2D_obs_minus_msis_2019_SH.png
    Figure/Ap_removal/2D_obs_minus_msis_2021_NH.png
    Figure/Ap_removal/2D_obs_minus_msis_2024_NH.png
    Figure/Ap_removal/2D_obs_minus_msis_2024_SH.png
    Figure/Ap_removal/2D_obs_minus_msis_2026_NH.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms

LAT_MIN, LAT_MAX = -60.0, 60.0
LAT_BIN = 3.0

OUT_DIR = Path("Figure/Ap_removal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS = [
    dict(
        label="2018 NH SSW (SWARM-A)",
        parquet=Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        date_start=pd.Timestamp("2018-01-30", tz="UTC"),
        date_end=pd.Timestamp("2018-03-06 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2018-01-30", tz="UTC"), pd.Timestamp("2018-02-09", tz="UTC")),
            (pd.Timestamp("2018-03-02", tz="UTC"), pd.Timestamp("2018-03-06", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2018-02-12", tz="UTC"),
        ssw_peak_label="SSW Peak (Feb 12)",
        temp_type="cosmic",
        temp_file=Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv"),
        temp_label="T(10 hPa) 60-90N",
        out_png=OUT_DIR / "2D_obs_minus_msis_2018_NH.png",
        x_interval=4,
        x_label="Date 2018",
    ),
    dict(
        label="2019 SH SSW (SWARM-A)",
        parquet=Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-09", tz="UTC"),
        ssw_peak_label="SSW Peak (Sep 9)",
        temp_type="era5",
        temp_dir=Path("data/SSW2019/ERA5"),
        lat_range=(-90.0, -60.0),
        temp_label="T(10 hPa) 60-90S",
        out_png=OUT_DIR / "2D_obs_minus_msis_2019_SH.png",
        x_interval=4,
        x_label="Date 2019",
    ),
    dict(
        label="2021 NH SSW (SWARM-C)",
        parquet=Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet"),
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end=pd.Timestamp("2021-02-05 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29", tz="UTC")),
            (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2021-01-04", tz="UTC"),
        ssw_peak_label="SSW Peak (Jan 4)",
        temp_type="era5",
        temp_dir=Path("data/SSW2021/ERA5"),
        lat_range=(60.0, 90.0),
        temp_label="T(10 hPa) 60-90N",
        out_png=OUT_DIR / "2D_obs_minus_msis_2021_NH.png",
        x_interval=5,
        x_label="Date 2020-2021",
    ),
    dict(
        label="2024 NH SSW (SWARM-A)",
        parquet=Path("normalizeddata/2024/swarm_dnsapod_2024_normalized_with_LT.parquet"),
        date_start=pd.Timestamp("2023-12-22", tz="UTC"),
        date_end=pd.Timestamp("2024-02-28 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2023-12-22", tz="UTC"), pd.Timestamp("2024-01-04", tz="UTC")),
            (pd.Timestamp("2024-01-31", tz="UTC"), pd.Timestamp("2024-02-28", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2024-01-16", tz="UTC"),
        ssw_peak_label="SSW Peak (Jan 16)",
        temp_type="era5",
        temp_dir=Path("data/SSW2024/ERA5"),
        lat_range=(60.0, 90.0),
        temp_label="T(10 hPa) 60-90N",
        out_png=OUT_DIR / "2D_obs_minus_msis_2024_NH.png",
        x_interval=5,
        x_label="Date 2023-2024",
    ),
    dict(
        label="2024 SH SSW (SWARM-A)",
        parquet=Path("normalizeddata/2024_SH/swarm_dnsapod_2024_SH_normalized_with_LT.parquet"),
        date_start=pd.Timestamp("2024-06-15", tz="UTC"),
        date_end=pd.Timestamp("2024-08-25 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2024-06-15", tz="UTC"), pd.Timestamp("2024-07-04", tz="UTC")),
            (pd.Timestamp("2024-08-10", tz="UTC"), pd.Timestamp("2024-08-25", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2024-07-07", tz="UTC"),
        ssw_peak_label="SSW Peak (Jul 7)",
        ssw_peak2=pd.Timestamp("2024-08-05", tz="UTC"),
        ssw_peak2_label="SSW Peak (Aug 5)",
        temp_type="era5",
        temp_dir=Path("data/SSW2024_SH/ERA5"),
        lat_range=(-90.0, -60.0),
        temp_label="T(10 hPa) 60-90S",
        out_png=OUT_DIR / "2D_obs_minus_msis_2024_SH.png",
        x_interval=5,
        x_label="Date 2024",
    ),
    dict(
        label="2026 NH SSW (SWARM-A)",
        parquet=Path("normalizeddata/2026/swarm_dnsapod_2026_normalized_with_LT.parquet"),
        date_start=pd.Timestamp("2025-12-20", tz="UTC"),
        date_end=pd.Timestamp("2026-03-15 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2025-12-20", tz="UTC"), pd.Timestamp("2026-01-10", tz="UTC")),
            (pd.Timestamp("2026-02-26", tz="UTC"), pd.Timestamp("2026-03-15", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2026-02-05", tz="UTC"),
        ssw_peak_label="SSW Peak (Feb 5)",
        temp_type="era5",
        temp_dir=Path("data/SSW2026/ERA5"),
        lat_range=(60.0, 90.0),
        temp_label="T(10 hPa) 60-90N",
        out_png=OUT_DIR / "2D_obs_minus_msis_2026_NH.png",
        x_interval=7,
        x_label="Date 2025-2026",
    ),
]


def load_density(ev):
    df = pd.read_parquet(ev["parquet"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for cname in ["lat", "latitude", "geod_lat"]:
        if cname in df.columns and cname != "lat":
            df = df.rename(columns={cname: "lat"})
            break
    df = df.dropna(subset=["datetime", "lat", "density_ratio_msis", "AP_AVG"])
    df = df[(df["datetime"] >= ev["date_start"]) & (df["datetime"] <= ev["date_end"])].copy()
    df["date"] = df["datetime"].dt.normalize()
    # delta_rho = rho_obs - rho_MSISreal  [kg/m^3]
    if "rho_model_real" in df.columns:
        # 2018/2019/2021: rho_model_real column exists
        df["delta_rho"] = df["density"] - df["rho_model_real"]
    else:
        # 2024/2026: reconstruct rho_MSISreal = density / density_ratio_msis
        df["rho_model_real_calc"] = df["density"] / df["density_ratio_msis"]
        df["delta_rho"] = df["density"] - df["rho_model_real_calc"]
    return df


def load_temp(ev):
    if ev.get("temp_type") == "cosmic":
        tdf = pd.read_csv(ev["temp_file"], parse_dates=["datetime"])
        tdf["datetime"] = pd.to_datetime(tdf["datetime"], utc=True)
        tdf["date"] = tdf["datetime"].dt.normalize()
        tdf = tdf[(tdf["datetime"] >= ev["date_start"]) & (tdf["datetime"] <= ev["date_end"])]
        return tdf.groupby("date")["T10_K"].mean().sort_index()

    temp_dir = ev.get("temp_dir", Path("."))
    files = sorted(temp_dir.glob("*.nc"))
    if not files:
        print(f"  [WARN] No ERA5 files in {temp_dir}")
        return pd.Series(dtype=float)

    lat_lo, lat_hi = min(ev["lat_range"]), max(ev["lat_range"])
    series_list = []
    for fp in files:
        try:
            with xr.open_dataset(fp) as ds:
                lats = ds["latitude"].values
                if lats[0] > lats[-1]:
                    ds_sub = ds.sel(latitude=slice(lat_hi, lat_lo))
                else:
                    ds_sub = ds.sel(latitude=slice(lat_lo, lat_hi))
                weights = np.cos(np.deg2rad(ds_sub["latitude"]))
                t_avg = ds_sub["t"].weighted(weights).mean(dim=["latitude", "longitude"]).squeeze()
                time_dim = "valid_time" if "valid_time" in t_avg.dims else "time"
                times = pd.to_datetime(t_avg[time_dim].values)
                vals = t_avg.values.ravel()
                s = pd.Series(vals, index=times)
                s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
                series_list.append(s)
        except Exception as e:
            print(f"  [WARN] Failed to read {fp.name}: {e}")

    if not series_list:
        return pd.Series(dtype=float)
    combined = pd.concat(series_list).sort_index().resample("D").mean()
    combined.index = combined.index.normalize()
    combined = combined[(combined.index >= ev["date_start"]) & (combined.index <= ev["date_end"])]
    return combined


def compute_2d_grid(df, val_col, date_edges, lat_edges):
    n_lat = len(lat_edges) - 1
    n_date = len(date_edges) - 1
    Z = np.full((n_lat, n_date), np.nan)
    if len(df) == 0:
        return Z
    df = df.copy()
    dt_secs = df["datetime"].astype("datetime64[ns, UTC]").astype("int64").values // 10**9
    edge_secs = date_edges.astype("datetime64[ns, UTC]").astype("int64").values // 10**9
    df["lat_i"] = np.digitize(df["lat"].values, lat_edges) - 1
    df["date_i"] = np.digitize(dt_secs, edge_secs) - 1
    valid = (
        (df["lat_i"] >= 0) & (df["lat_i"] < n_lat) &
        (df["date_i"] >= 0) & (df["date_i"] < n_date) &
        np.isfinite(df[val_col].values)
    )
    grouped = df[valid].groupby(["lat_i", "date_i"])[val_col].median()
    for (i, j), val in grouped.items():
        Z[i, j] = val
    return Z


def plot_event(ev):
    print(f"\nProcessing {ev['label']}...")
    df = load_density(ev)
    print(f"  Data rows: {len(df)}")

    daily = df.groupby("date").agg(
        delta_rho=("delta_rho", "median"),
        ap=("AP_AVG", "mean"),
    ).reset_index()

    s_delta = pd.Series(daily["delta_rho"].values, index=daily["date"])
    s_ap = pd.Series(daily["ap"].values, index=daily["date"])
    s_temp = load_temp(ev)
    print(f"  ERA5/COSMIC data points: {len(s_temp)}")

    date_edges = pd.date_range(
        ev["date_start"].normalize() - pd.Timedelta(hours=12),
        ev["date_end"].normalize() + pd.Timedelta(hours=36),
        freq="D",
    )
    lat_edges = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    Z = compute_2d_grid(df, "delta_rho", date_edges, lat_edges)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.6], hspace=0.22)
    ax1d = fig.add_subplot(gs[0])
    ax2d = fig.add_subplot(gs[1], sharex=ax1d)

    ax_t = ax1d.twinx()
    ax_a = ax1d.twinx()
    ax_a.spines["right"].set_position(("axes", 1.08))
    ax_a.spines["right"].set_visible(True)

    for r_start, r_end in ev["ref_dates"]:
        ax1d.axvspan(r_start, r_end, color="#008080", alpha=0.10, zorder=0)

    l1 = ax1d.plot(s_delta.index, s_delta.values, "o-",
                   color="#1f77b4", lw=2, ms=4.5,
                   label=r"$\rho_{\rm obs} - \rho_{\rm MSIS}$ [kg/m$^3$]")
    ax1d.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax1d.set_ylabel(r"$\rho_{\rm obs} - \rho_{\rm MSIS}$ [kg m$^{-3}$]",
                    color="#1f77b4", fontweight="bold")
    ax1d.tick_params(axis="y", labelcolor="#1f77b4")

    l2 = []
    if len(s_temp) > 0:
        l2 = ax_t.plot(s_temp.index, s_temp.values, "^-",
                       color="#dc143c", lw=2, ms=4.5, label=ev["temp_label"])
        ax_t.set_ylabel(f"{ev['temp_label']} [K]", color="#dc143c", fontweight="bold")
        ax_t.tick_params(axis="y", labelcolor="#dc143c")

    l3 = ax_a.plot(s_ap.index, s_ap.values, "s-",
                   color="#e67e22", lw=1.5, ms=4.5, alpha=0.85, label="Daily mean Ap")
    ax_a.set_ylabel("Ap Index", color="#e67e22", fontweight="bold")
    ax_a.tick_params(axis="y", labelcolor="#e67e22")

    trans = mtransforms.blended_transform_factory(ax1d.transData, ax1d.transAxes)
    ax1d.axvline(ev["ssw_peak"], color="red", ls="--", lw=1.8)
    ax1d.text(ev["ssw_peak"], 0.97, f" {ev['ssw_peak_label']}",
              transform=trans, color="red", fontweight="bold",
              verticalalignment="top", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85))
    if "ssw_peak2" in ev:
        ax1d.axvline(ev["ssw_peak2"], color="darkred", ls="--", lw=1.8)
        ax1d.text(ev["ssw_peak2"], 0.97, f" {ev['ssw_peak2_label']}",
                  transform=trans, color="darkred", fontweight="bold",
                  verticalalignment="top", fontsize=8.5,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkred", alpha=0.85))

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1d.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.9)
    ax1d.grid(True, linestyle=":", alpha=0.5)
    ax1d.set_title(
        f"{ev['label']}  --  "
        r"$\rho_{\rm obs} - \rho_{\rm MSIS{real}}$ [kg/m$^3$] & Stratospheric T(10 hPa)",
        fontsize=12, fontweight="bold", pad=14)
    plt.setp(ax1d.get_xticklabels(), visible=False)

    X_edges = mdates.date2num(date_edges)
    Y_edges = lat_edges

    # auto-scale vmin/vmax from data (symmetric around 0)
    z_abs_max = np.nanpercentile(np.abs(Z), 97)
    vmax = max(z_abs_max, 1e-14)  # avoid zero range
    mesh = ax2d.pcolormesh(X_edges, Y_edges, Z,
                           cmap="RdBu_r", vmin=-vmax, vmax=+vmax, shading="flat")
    cbar = plt.colorbar(mesh, ax=ax2d, pad=0.015, aspect=15)
    cbar.set_label(r"$\rho_{\rm obs} - \rho_{\rm MSIS{real}}$ [kg m$^{-3}$]",
                   fontsize=9, fontweight="bold")

    X_centers = 0.5 * (X_edges[:-1] + X_edges[1:])
    Y_centers = 0.5 * (Y_edges[:-1] + Y_edges[1:])
    Z_finite = np.where(np.isfinite(Z), Z, 0.0)
    ax2d.contour(X_centers, Y_centers, Z_finite,
                 levels=[0.0], colors="black", linewidths=0.8, linestyles="-", alpha=0.4)
    ax2d.contour(X_centers, Y_centers, Z_finite,
                 levels=[-vmax/3, vmax/3], colors="gray", linewidths=0.6, linestyles="--", alpha=0.4)

    ax2d.axvline(mdates.date2num(ev["ssw_peak"]), color="red", ls="--", lw=1.8)
    if "ssw_peak2" in ev:
        ax2d.axvline(mdates.date2num(ev["ssw_peak2"]), color="darkred", ls="--", lw=1.8)

    ax2d.axhline(0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax2d.set_ylabel("Latitude [deg]", fontweight="bold")
    ax2d.set_ylim(LAT_MIN, LAT_MAX)
    ax2d.grid(True, linestyle=":", alpha=0.4)
    ax2d.set_title(
        r"$\rho_{\rm obs} - \rho_{\rm MSIS{real}}$ [kg m$^{-3}$]"
        "  (Red: obs > MSIS,  Blue: obs < MSIS)",
        fontsize=10, fontweight="bold", loc="left")

    ax2d.xaxis_date()
    ax2d.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2d.xaxis.set_major_locator(mdates.DayLocator(interval=ev["x_interval"]))
    ax2d.set_xlabel(f"Date (MM/DD)  --  {ev['x_label']}", fontweight="bold")

    fig.canvas.draw()
    pos_2d = ax2d.get_position()
    pos_1d = ax1d.get_position()
    ax1d.set_position([pos_2d.x0, pos_1d.y0, pos_2d.width, pos_1d.height])

    plt.savefig(ev["out_png"], dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {ev['out_png']}")


def main():
    for ev in EVENTS:
        plot_event(ev)
    print("\nAll events completed!")


if __name__ == "__main__":
    main()
