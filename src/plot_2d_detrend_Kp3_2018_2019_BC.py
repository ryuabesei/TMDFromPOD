r"""
plot_2d_detrend_Kp3_2018_2019_BC.py

Purpose:
    Same layout/method as 2D_detrend_Kp3_YYYY.png (plot_2d_detrend_Kp3_filtered_3years.py),
    but for SWARM-B and SWARM-C for the 2018 NH SSW and 2019 SH SSW events.

    Method:
        - Regression fitted on Kp<3 days only (AP_AVG < 15)
        - delta_y_ap = density_ratio_msis - (a*Ap + b) computed for ALL data
        - 1D panel: connected line + solid markers on Kp<3 days
        - 2D panel: pcolormesh of delta_y_ap (RdBu_r, ±0.20)
        - Kp>=3 days shaded in red in 2D panel

Output:
    Figure/Ap_removal/2D_detrend_Kp3_2018_SWARM-B.png
    Figure/Ap_removal/2D_detrend_Kp3_2018_SWARM-C.png
    Figure/Ap_removal/2D_detrend_Kp3_2019_SWARM-B.png
    Figure/Ap_removal/2D_detrend_Kp3_2019_SWARM-C.png
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

AP_KP3 = 15.0   # Kp=3 <=> Ap=15

LAT_MIN, LAT_MAX = -60.0, 60.0
LAT_BIN = 3.0

OUT_DIR = Path("Figure/Ap_removal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS = [
    # ── 2018 NH SSW SWARM-B ──────────────────────────────────────────────────
    dict(
        year=2018, sat="SWARM-B",
        label="2018 NH SSW (SWARM-B)",
        parquet=Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        date_start=pd.Timestamp("2018-01-30", tz="UTC"),
        date_end=pd.Timestamp("2018-03-06 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2018-01-30", tz="UTC"), pd.Timestamp("2018-02-09", tz="UTC")),
            (pd.Timestamp("2018-03-02", tz="UTC"), pd.Timestamp("2018-03-06", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2018-02-12", tz="UTC"),
        ssw_peak_label="SSW Peak (DOY 43)",
        temp_type="cosmic",
        temp_file=Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv"),
        temp_label="T(10 hPa) 60-90N",
        x_label="Date 2018 (DOY 30-65)",
        x_interval=4,
        out_png=OUT_DIR / "2D_detrend_Kp3_2018_SWARM-B.png",
    ),
    # ── 2018 NH SSW SWARM-C ──────────────────────────────────────────────────
    dict(
        year=2018, sat="SWARM-C",
        label="2018 NH SSW (SWARM-C)",
        parquet=Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        date_start=pd.Timestamp("2018-01-30", tz="UTC"),
        date_end=pd.Timestamp("2018-03-06 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2018-01-30", tz="UTC"), pd.Timestamp("2018-02-09", tz="UTC")),
            (pd.Timestamp("2018-03-02", tz="UTC"), pd.Timestamp("2018-03-06", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2018-02-12", tz="UTC"),
        ssw_peak_label="SSW Peak (DOY 43)",
        temp_type="cosmic",
        temp_file=Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv"),
        temp_label="T(10 hPa) 60-90N",
        x_label="Date 2018 (DOY 30-65)",
        x_interval=4,
        out_png=OUT_DIR / "2D_detrend_Kp3_2018_SWARM-C.png",
    ),
    # ── 2019 SH SSW SWARM-B ──────────────────────────────────────────────────
    dict(
        year=2019, sat="SWARM-B",
        label="2019 SH SSW (SWARM-B)",
        parquet=Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"),
        ssw_peak_label="SSW Peak (Sep 19)",
        temp_type="era5",
        temp_dir=Path("data/SSW2019/ERA5"),
        lat_range=(-90.0, -60.0),
        temp_label="T(10 hPa) 60-90S",
        x_label="Date 2019 (08/20 - 09/23)",
        x_interval=4,
        out_png=OUT_DIR / "2D_detrend_Kp3_2019_SWARM-B.png",
    ),
    # ── 2019 SH SSW SWARM-C ──────────────────────────────────────────────────
    dict(
        year=2019, sat="SWARM-C",
        label="2019 SH SSW (SWARM-C)",
        parquet=Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"),
        ssw_peak_label="SSW Peak (Sep 19)",
        temp_type="era5",
        temp_dir=Path("data/SSW2019/ERA5"),
        lat_range=(-90.0, -60.0),
        temp_label="T(10 hPa) 60-90S",
        x_label="Date 2019 (08/20 - 09/23)",
        x_interval=4,
        out_png=OUT_DIR / "2D_detrend_Kp3_2019_SWARM-C.png",
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
    return df


def load_temp(ev):
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
                print(f"  [WARN] {fp.name}: {e}")
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

    df_all = load_density(ev)
    print(f"  Data rows: {len(df_all)}")

    all_dates = pd.date_range(
        ev["date_start"].normalize(),
        ev["date_end"].normalize(),
        freq="D", tz="UTC",
    )

    daily_all = df_all.groupby("date").agg(
        ratio=("density_ratio_msis", "median"),
        ap=("AP_AVG", "mean"),
    ).reset_index()

    # Kp<3 only for regression
    kp_ok_dates = set(daily_all.loc[daily_all["ap"] < AP_KP3, "date"].tolist())
    daily_filt = daily_all[daily_all["date"].isin(kp_ok_dates)].copy()

    x = daily_filt["ap"].values
    y = daily_filt["ratio"].values
    p = np.polyfit(x, y, 1)
    slope, intercept = p[0], p[1]
    r = float(np.corrcoef(x, y)[0, 1])
    print(f"  Regression (Kp<3): a={slope:.5f}, b={intercept:.5f}, r={r:.3f}")

    # Apply to ALL data
    df_all["ap_pred"] = slope * df_all["AP_AVG"] + intercept
    df_all["delta_y_ap"] = df_all["density_ratio_msis"] - df_all["ap_pred"]

    daily_all["delta_ap"] = daily_all["ratio"] - (slope * daily_all["ap"] + intercept)

    s_delta_all = daily_all.set_index("date")["delta_ap"].reindex(all_dates)
    s_ap_all = daily_all.set_index("date")["ap"].reindex(all_dates)

    s_ratio_filt = daily_filt.set_index("date")["ratio"].reindex(all_dates)
    kp_removed = s_ratio_filt.isna()

    # Solid markers: Kp<3 days only; thin line interpolated
    s_delta_solid = s_delta_all.copy()
    s_delta_solid[kp_removed] = np.nan
    s_delta_conn2 = s_delta_solid.interpolate(method="time", limit_area="inside")

    s_temp = load_temp(ev)
    print(f"  ERA5/COSMIC data points: {len(s_temp)}")

    date_edges = pd.date_range(
        ev["date_start"].normalize() - pd.Timedelta(hours=12),
        ev["date_end"].normalize() + pd.Timedelta(hours=36),
        freq="D",
    )
    lat_edges = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    Z_detrend = compute_2d_grid(df_all, "delta_y_ap", date_edges, lat_edges)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.6], hspace=0.22)
    ax1d = fig.add_subplot(gs[0])
    ax2d = fig.add_subplot(gs[1], sharex=ax1d)

    # ── 1D Panel ──────────────────────────────────────────────────────────────
    ax_t = ax1d.twinx()
    ax_a = ax1d.twinx()
    ax_a.spines["right"].set_position(("axes", 1.08))
    ax_a.spines["right"].set_visible(True)

    for r_start, r_end in ev["ref_dates"]:
        ax1d.axvspan(r_start, r_end, color="#008080", alpha=0.10, zorder=0)

    # Thin background line connecting over Kp>3 gaps
    ax1d.plot(s_delta_conn2.index, s_delta_conn2.values, "-",
              color="#1f77b4", lw=1.2, alpha=0.4, zorder=2)
    # Solid markers on Kp<3 days only
    s_valid = s_delta_solid.dropna()
    l1 = ax1d.plot(s_valid.index, s_valid.values, "o-",
                   color="#1f77b4", lw=2.0, ms=4.5, zorder=3,
                   label=r"$\Delta y_{Ap}$ (Left)")

    ax1d.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax1d.set_ylabel(r"$\Delta y_{Ap}$ (Density Ratio Residual)", color="#1f77b4", fontweight="bold")
    ax1d.tick_params(axis="y", labelcolor="#1f77b4")

    l2 = []
    if len(s_temp) > 0:
        l2 = ax_t.plot(s_temp.index, s_temp.values, "^-",
                       color="#dc143c", lw=2, ms=4.5, label=ev["temp_label"])
        ax_t.set_ylabel(f"{ev['temp_label']} [K]", color="#dc143c", fontweight="bold")
        ax_t.tick_params(axis="y", labelcolor="#dc143c")

    l3 = ax_a.plot(s_ap_all.index, s_ap_all.values, "s-",
                   color="#e67e22", lw=1.5, ms=4.5, alpha=0.85, label="Daily mean Ap")
    kp_bad = s_ap_all[kp_removed]
    l4 = []
    if len(kp_bad) > 0:
        l4 = ax_a.plot(kp_bad.index, kp_bad.values, "x",
                       color="#c0392b", ms=7, mew=2.0, zorder=5,
                       label=f"Kp>=3 (Ap>={AP_KP3:.0f}, removed)")
    ax_a.axhline(AP_KP3, color="#c0392b", lw=1.0, ls=":", alpha=0.7)
    ax_a.set_ylabel("Ap Index", color="#e67e22", fontweight="bold")
    ax_a.tick_params(axis="y", labelcolor="#e67e22")

    # SSW peak line
    ax1d.axvline(ev["ssw_peak"], color="red", ls="--", lw=1.8)
    trans = mtransforms.blended_transform_factory(ax1d.transData, ax1d.transAxes)
    ax1d.text(ev["ssw_peak"], 0.97, f" {ev['ssw_peak_label']}",
              transform=trans, color="red", fontweight="bold",
              verticalalignment="top", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85))

    ann_text = (f"slope a={slope:.5f}  intercept b={intercept:.5f}  "
                f"corr(y,Ap)={r:.3f}  [Kp<3 filtered regression]")
    ax1d.text(0.015, 0.88, ann_text, transform=ax1d.transAxes,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85), fontsize=8.5)

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1d.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.9)
    ax1d.grid(True, linestyle=":", alpha=0.5)
    ax1d.set_title(
        f"{ev['label']}  [Kp < 3 filtered]  -- 1D Metrics & 2D Latitude-Time Density Distributions",
        fontsize=12, fontweight="bold", pad=14)
    plt.setp(ax1d.get_xticklabels(), visible=False)

    # ── 2D Panel ──────────────────────────────────────────────────────────────
    X_edges = mdates.date2num(date_edges)
    Y_edges = lat_edges

    mesh = ax2d.pcolormesh(X_edges, Y_edges, Z_detrend,
                           cmap="RdBu_r", vmin=-0.20, vmax=+0.20, shading="flat")
    cbar = plt.colorbar(mesh, ax=ax2d, pad=0.015, aspect=15)
    cbar.set_label(r"Residual $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$",
                   fontsize=9, fontweight="bold")

    X_centers = 0.5 * (X_edges[:-1] + X_edges[1:])
    Y_centers = 0.5 * (Y_edges[:-1] + Y_edges[1:])
    ax2d.contour(X_centers, Y_centers, Z_detrend,
                 levels=[0.0], colors="black", linewidths=0.8, linestyles="-", alpha=0.4)
    ax2d.contour(X_centers, Y_centers, Z_detrend,
                 levels=[-0.1, 0.1], colors="gray", linewidths=0.6, linestyles="--", alpha=0.4)

    # Shade Kp>=3 columns
    kp_bad_dates = s_ap_all[kp_removed].index
    for bad_date in kp_bad_dates:
        ax2d.axvspan(
            mdates.date2num(bad_date),
            mdates.date2num(bad_date + pd.Timedelta(days=1)),
            color="#c0392b", alpha=0.10, zorder=1)

    ax2d.axvline(mdates.date2num(ev["ssw_peak"]), color="red", ls="--", lw=1.8)
    ax2d.axhline(0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax2d.set_ylabel("Latitude [deg]", fontweight="bold")
    ax2d.set_ylim(LAT_MIN, LAT_MAX)
    ax2d.grid(True, linestyle=":", alpha=0.4)
    ax2d.set_title(
        r"(b) Ap-Detrended Residual $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$"
        "  [Kp<3 regression; red shading = Kp>=3 days]",
        fontsize=10, fontweight="bold", loc="left")

    ax2d.xaxis_date()
    ax2d.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2d.xaxis.set_major_locator(mdates.DayLocator(interval=ev["x_interval"]))
    ax2d.set_xlabel("Date (MM/DD)", fontweight="bold")

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
    print("\nAll 4 events completed!")


if __name__ == "__main__":
    main()
