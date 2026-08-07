r"""
plot_2d_detrend_Kp3_filtered_3years.py

Purpose:
    Same layout as 2D_detrend_only_temp_ap_YYYY.png (plot_2d_detrend_only_3years.py),
    but with Kp > 3 (AP_AVG >= 15) days EXCLUDED from the Ap regression.

    For each event:
        - Regression fitted on Kp<3 days only  -> slope a, intercept b
        - delta_y_ap = obs_ratio - (a*Ap + b) computed for ALL observations
          (so the 2D map covers the full period including Kp>3 days)
        - 1D panel: connected line (interpolated over Kp>3 gaps) + solid markers
          on Kp<3 days (same style as ratio_detrend_Kp3_temp_ap_combined_3years.png)
        - 2D panel: pcolormesh of delta_y_ap over full lat-time grid (RdBu_r)

    Per-event output (same shape as 2D_detrend_only_temp_ap_YYYY.png):
        Figure/Ap_removal/2D_detrend_Kp3_2018.png
        Figure/Ap_removal/2D_detrend_Kp3_2019.png
        Figure/Ap_removal/2D_detrend_Kp3_2021.png
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

# Kp<3 threshold
AP_KP3 = 15.0   # Kp=3 <=> Ap=15

# Paths
P2018       = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019       = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021       = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")
COSMIC_2018 = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
ERA5_2019   = Path("data/SSW2019/ERA5")
ERA5_2021   = Path("data/SSW2021/ERA5")
OUT_DIR     = Path("Figure/Ap_removal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = -60.0, 60.0
LAT_BIN = 3.0

EVENTS = [
    dict(
        year=2018,
        label="2018 NH SSW (SWARM-A)",
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
        x_label="Date 2018 (DOY 30-65)",
        x_interval=4,
    ),
    dict(
        year=2019,
        label="2019 SH SSW (SWARM-A)",
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
        temp_dir=ERA5_2019,
        lat_range=(-90.0, -60.0),
        temp_label="T(10 hPa) 60-90S",
        x_label="Date 2019 (08/20 - 09/23)",
        x_interval=4,
    ),
    dict(
        year=2021,
        label="2021 NH SSW (SWARM-C)",
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
        temp_dir=ERA5_2021,
        lat_range=(60.0, 90.0),
        temp_label="T(10 hPa) 60-90N",
        x_label="Date 2020-2021 (12/25 - 02/05)",
        x_interval=5,
    ),
]


def load_density(ev: dict) -> pd.DataFrame:
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
        lat_lo, lat_hi = min(ev["lat_range"]), max(ev["lat_range"])
        series_list = []
        for fp in files:
            with xr.open_dataset(fp) as ds:
                ds_sub = ds.sel(latitude=slice(lat_hi, lat_lo))
                weights = np.cos(np.deg2rad(ds_sub["latitude"]))
                t_avg = ds_sub["t"].weighted(weights).mean(dim=["latitude", "longitude"]).squeeze()
                times = pd.to_datetime(t_avg["valid_time"].values)
                vals  = t_avg.values.ravel()
                s = pd.Series(vals, index=times)
                s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
                series_list.append(s)
        combined = pd.concat(series_list).sort_index().resample("D").mean()
        combined.index = combined.index.normalize()
        combined = combined[(combined.index >= ev["date_start"]) & (combined.index <= ev["date_end"])]
        return combined


def compute_2d_grid(df: pd.DataFrame, val_col: str,
                    date_edges: pd.DatetimeIndex, lat_edges: np.ndarray) -> np.ndarray:
    n_lat  = len(lat_edges) - 1
    n_date = len(date_edges) - 1
    Z = np.full((n_lat, n_date), np.nan)
    if len(df) == 0:
        return Z
    df = df.copy()
    dt_secs   = df["datetime"].astype("datetime64[ns, UTC]").astype("int64").values // 10**9
    edge_secs = date_edges.astype("datetime64[ns, UTC]").astype("int64").values // 10**9
    df["lat_i"]  = np.digitize(df["lat"].values, lat_edges) - 1
    df["date_i"] = np.digitize(dt_secs, edge_secs) - 1
    valid = (
        (df["lat_i"]  >= 0) & (df["lat_i"]  < n_lat) &
        (df["date_i"] >= 0) & (df["date_i"] < n_date) &
        np.isfinite(df[val_col].values)
    )
    grouped = df[valid].groupby(["lat_i", "date_i"])[val_col].median()
    for (i, j), val in grouped.items():
        Z[i, j] = val
    return Z


def plot_event(ev: dict) -> None:
    year    = ev["year"]
    out_png = OUT_DIR / f"2D_detrend_Kp3_{year}.png"
    print(f"Processing {ev['label']}...")

    df_all = load_density(ev)

    # Full daily index
    all_dates = pd.date_range(
        ev["date_start"].normalize(),
        ev["date_end"].normalize(),
        freq="D", tz="UTC",
    )

    # Daily aggregation from ALL data
    daily_all = df_all.groupby("date").agg(
        ratio=("density_ratio_msis", "median"),
        ap=("AP_AVG", "mean"),
    ).reset_index()

    # Kp<3 filter: regression fitted only on days with AP_AVG < 15
    kp_ok_dates = set(daily_all.loc[daily_all["ap"] < AP_KP3, "date"].tolist())
    daily_filt  = daily_all[daily_all["date"].isin(kp_ok_dates)].copy()

    x = daily_filt["ap"].values
    y = daily_filt["ratio"].values
    p = np.polyfit(x, y, 1)
    slope, intercept = p[0], p[1]
    r = float(np.corrcoef(x, y)[0, 1])

    # Apply regression to ALL data for 2D map
    df_all["ap_pred"]    = slope * df_all["AP_AVG"] + intercept
    df_all["delta_y_ap"] = df_all["density_ratio_msis"] - df_all["ap_pred"]

    # Daily residual series for 1D panel
    daily_all["delta_ap"] = daily_all["ratio"] - (slope * daily_all["ap"] + intercept)

    # Series indexed on full date range
    s_delta_all = daily_all.set_index("date")["delta_ap"].reindex(all_dates)
    s_ap_all    = daily_all.set_index("date")["ap"].reindex(all_dates)

    # Kp-removed mask (True = day with Kp>=3)
    s_ratio_filt = daily_filt.set_index("date")["ratio"].reindex(all_dates)
    kp_removed   = s_ratio_filt.isna()

    # Connected (interpolated) residual line
    s_delta_conn = s_delta_all.copy()
    # delta is already present for all days (regression applied to all),
    # so the 1D series has no NaN from Kp removal; mark Kp>3 days as thin dashed
    # To visually match the Kp3-filtered 1D plot, set Kp>3 days to NaN in
    # the "solid marker" series and only interpolate for the thin background line.
    s_delta_solid = s_delta_all.copy()
    s_delta_solid[kp_removed] = np.nan
    s_delta_conn2 = s_delta_solid.interpolate(method="time", limit_area="inside")

    s_temp = load_temp(ev)

    # 2D grid
    date_edges = pd.date_range(
        ev["date_start"].normalize() - pd.Timedelta(hours=12),
        ev["date_end"].normalize()   + pd.Timedelta(hours=36),
        freq="D",
    )
    lat_edges = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    Z_detrend = compute_2d_grid(df_all, "delta_y_ap", date_edges, lat_edges)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.6], hspace=0.22)
    ax1d         = fig.add_subplot(gs[0])
    ax2d_detrend = fig.add_subplot(gs[1], sharex=ax1d)

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
    # Mark Kp>3 days with red X on Ap axis
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
    ax1d.text(
        ev["ssw_peak"], 0.97,
        f" {ev['ssw_peak_label']}",
        transform=trans, color="red", fontweight="bold",
        verticalalignment="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85),
    )

    ann_text = (f"slope a={slope:.5f}  intercept b={intercept:.5f}  "
                f"corr(y,Ap)={r:.3f}  [Kp<3 filtered regression]")
    ax1d.text(0.015, 0.88, ann_text, transform=ax1d.transAxes,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85), fontsize=8.5)

    lines  = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1d.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.9)
    ax1d.grid(True, linestyle=":", alpha=0.5)
    ax1d.set_title(
        f"{ev['label']}  [Kp < 3 filtered]  — 1D Metrics & 2D Latitude-Time Density Distributions",
        fontsize=12, fontweight="bold", pad=14,
    )
    plt.setp(ax1d.get_xticklabels(), visible=False)

    # ── 2D Ap-Detrended Panel ─────────────────────────────────────────────────
    X_edges = mdates.date2num(date_edges)
    Y_edges = lat_edges

    mesh = ax2d_detrend.pcolormesh(
        X_edges, Y_edges, Z_detrend,
        cmap="RdBu_r", vmin=-0.20, vmax=+0.20, shading="flat",
    )
    cbar = plt.colorbar(mesh, ax=ax2d_detrend, pad=0.015, aspect=15)
    cbar.set_label(r"Residual $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$",
                   fontsize=9, fontweight="bold")

    # Contour lines
    X_centers = 0.5 * (X_edges[:-1] + X_edges[1:])
    Y_centers = 0.5 * (Y_edges[:-1] + Y_edges[1:])
    ax2d_detrend.contour(X_centers, Y_centers, Z_detrend,
                         levels=[0.0], colors="black", linewidths=0.8,
                         linestyles="-", alpha=0.4)
    ax2d_detrend.contour(X_centers, Y_centers, Z_detrend,
                         levels=[-0.1, 0.1], colors="gray", linewidths=0.6,
                         linestyles="--", alpha=0.4)

    # Shade Kp>3 columns with light hatching
    kp_bad_dates = s_ap_all[kp_removed].index
    for bad_date in kp_bad_dates:
        ax2d_detrend.axvspan(
            mdates.date2num(bad_date),
            mdates.date2num(bad_date + pd.Timedelta(days=1)),
            color="#c0392b", alpha=0.10, zorder=1,
        )

    ax2d_detrend.axvline(mdates.date2num(ev["ssw_peak"]), color="red", ls="--", lw=1.8)
    ax2d_detrend.axhline(0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax2d_detrend.set_ylabel("Latitude [deg]", fontweight="bold")
    ax2d_detrend.set_ylim(LAT_MIN, LAT_MAX)
    ax2d_detrend.grid(True, linestyle=":", alpha=0.4)
    ax2d_detrend.set_title(
        r"(b) Ap-Detrended Residual $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$"
        "  [Kp<3 regression; red shading = Kp>=3 days]",
        fontsize=10, fontweight="bold", loc="left",
    )

    # Date X-axis
    ax2d_detrend.xaxis_date()
    ax2d_detrend.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2d_detrend.xaxis.set_major_locator(mdates.DayLocator(interval=ev["x_interval"]))
    ax2d_detrend.set_xlabel("Date (MM/DD)", fontweight="bold")

    # Align 1D panel width to 2D panel (accounts for colorbar)
    fig.canvas.draw()
    pos_2d = ax2d_detrend.get_position()
    pos_1d = ax1d.get_position()
    ax1d.set_position([pos_2d.x0, pos_1d.y0, pos_2d.width, pos_1d.height])

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main() -> None:
    for ev in EVENTS:
        plot_event(ev)
    print("\nAll 3 events completed!")


if __name__ == "__main__":
    main()
