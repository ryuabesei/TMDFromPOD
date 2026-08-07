r"""
plot_2d_detrend_2024_2026_SSW.py

Purpose:
    Apply the same Ap-detrending method as 2D_detrend_only_temp_ap_2018.png
    to three new SSW events:
      1. 2024 NH SSW  (SWARM-A)  2023-12-22 to 2024-02-28
      2. 2024 SH SSW  (SWARM-A)  2024-06-15 to 2024-08-25  (dual peaks)
      3. 2026 NH SSW  (SWARM-A)  2025-12-20 to 2026-03-15

    Per-event figure layout (matches 2D_detrend_only_temp_ap_2018.png):
        Top panel   : 1D time series  Delta y_Ap / T(10 hPa) / Ap
        Bottom panel: 2D latitude-time pcolormesh of Delta y_Ap  [RdBu_r]

Output:
    Figure/Ap_removal/2D_detrend_2024_NH.png
    Figure/Ap_removal/2D_detrend_2024_SH.png
    Figure/Ap_removal/2D_detrend_2026_NH.png
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

# Grid
LAT_MIN, LAT_MAX = -60.0, 60.0
LAT_BIN = 3.0

OUT_DIR = Path("Figure/Ap_removal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Event configurations
EVENTS = [
    dict(
        label="2024 NH SSW (SWARM-A)",
        sat="SWARM-A",
        parquet=Path("normalizeddata/2024/swarm_dnsapod_2024_normalized_with_LT.parquet"),
        date_start=pd.Timestamp("2023-12-22", tz="UTC"),
        date_end=pd.Timestamp("2024-02-28 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2023-12-22", tz="UTC"), pd.Timestamp("2024-01-04", tz="UTC")),
            (pd.Timestamp("2024-01-31", tz="UTC"), pd.Timestamp("2024-02-28", tz="UTC")),
        ],
        # SSW peak: major warming peak mid-January 2024
        ssw_peak=pd.Timestamp("2024-01-16", tz="UTC"),
        ssw_peak_label="SSW Peak (Jan 16)",
        temp_type="era5",
        temp_dir=Path("data/SSW2024/ERA5"),
        lat_range=(60.0, 90.0),
        temp_label="T(10 hPa) 60-90N",
        out_png=OUT_DIR / "2D_detrend_2024_NH.png",
        x_interval=5,
        x_label="Date (2023-2024)",
    ),
    dict(
        label="2024 SH SSW (SWARM-A)",
        sat="SWARM-A",
        parquet=Path("normalizeddata/2024_SH/swarm_dnsapod_2024_SH_normalized_with_LT.parquet"),
        date_start=pd.Timestamp("2024-06-15", tz="UTC"),
        date_end=pd.Timestamp("2024-08-25 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2024-06-15", tz="UTC"), pd.Timestamp("2024-07-04", tz="UTC")),
            (pd.Timestamp("2024-08-10", tz="UTC"), pd.Timestamp("2024-08-25", tz="UTC")),
        ],
        # Dual peaks: Jul 7 and Aug 5
        ssw_peak=pd.Timestamp("2024-07-07", tz="UTC"),
        ssw_peak2=pd.Timestamp("2024-08-05", tz="UTC"),
        ssw_peak_label="SSW Peak (Jul 7)",
        ssw_peak2_label="SSW Peak (Aug 5)",
        temp_type="era5",
        temp_dir=Path("data/SSW2024_SH/ERA5"),
        lat_range=(-90.0, -60.0),
        temp_label="T(10 hPa) 60-90S",
        out_png=OUT_DIR / "2D_detrend_2024_SH.png",
        x_interval=5,
        x_label="Date (2024)",
    ),
    dict(
        label="2026 NH SSW (SWARM-A)",
        sat="SWARM-A",
        parquet=Path("normalizeddata/2026/swarm_dnsapod_2026_normalized_with_LT.parquet"),
        date_start=pd.Timestamp("2025-12-20", tz="UTC"),
        date_end=pd.Timestamp("2026-03-15 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2025-12-20", tz="UTC"), pd.Timestamp("2026-01-10", tz="UTC")),
            (pd.Timestamp("2026-02-26", tz="UTC"), pd.Timestamp("2026-03-15", tz="UTC")),
        ],
        # SSW peak: major warming peak early February 2026
        ssw_peak=pd.Timestamp("2026-02-05", tz="UTC"),
        ssw_peak_label="SSW Peak (Feb 5)",
        temp_type="era5",
        temp_dir=Path("data/SSW2026/ERA5"),
        lat_range=(60.0, 90.0),
        temp_label="T(10 hPa) 60-90N",
        out_png=OUT_DIR / "2D_detrend_2026_NH.png",
        x_interval=7,
        x_label="Date (2025-2026)",
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
    files = sorted(ev["temp_dir"].glob("*.nc"))
    if not files:
        print(f"  [WARN] No ERA5 files in {ev['temp_dir']}")
        return pd.Series(dtype=float)

    lat_lo, lat_hi = min(ev["lat_range"]), max(ev["lat_range"])
    series_list = []
    for fp in files:
        try:
            with xr.open_dataset(fp) as ds:
                # latitude slice (ERA5 may be descending)
                lats = ds["latitude"].values
                if lats[0] > lats[-1]:  # descending
                    ds_sub = ds.sel(latitude=slice(lat_hi, lat_lo))
                else:
                    ds_sub = ds.sel(latitude=slice(lat_lo, lat_hi))

                weights = np.cos(np.deg2rad(ds_sub["latitude"]))
                t_avg = ds_sub["t"].weighted(weights).mean(
                    dim=["latitude", "longitude"]
                ).squeeze()

                # time dim may be 'valid_time' or 'time'
                time_dim = "valid_time" if "valid_time" in t_avg.dims else "time"
                times = pd.to_datetime(t_avg[time_dim].values)
                vals  = t_avg.values.ravel()
                s = pd.Series(vals, index=times)
                s.index = (
                    s.index.tz_localize("UTC")
                    if s.index.tz is None
                    else s.index.tz_convert("UTC")
                )
                series_list.append(s)
        except Exception as e:
            print(f"  [WARN] Failed to read {fp.name}: {e}")

    if not series_list:
        return pd.Series(dtype=float)

    combined = pd.concat(series_list).sort_index().resample("D").mean()
    combined.index = combined.index.normalize()
    combined = combined[
        (combined.index >= ev["date_start"]) & (combined.index <= ev["date_end"])
    ]
    return combined


def compute_2d_grid(
    df: pd.DataFrame,
    val_col: str,
    date_edges: pd.DatetimeIndex,
    lat_edges: np.ndarray,
) -> np.ndarray:
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
    print(f"\nProcessing {ev['label']}...")

    df = load_density(ev)
    print(f"  Data rows: {len(df)}")

    # Daily aggregation for regression
    daily = df.groupby("date").agg(
        ratio=("density_ratio_msis", "median"),
        ap=("AP_AVG", "mean"),
    ).reset_index()

    x = daily["ap"].values
    y = daily["ratio"].values
    p = np.polyfit(x, y, 1)
    slope, intercept = p[0], p[1]
    r = float(np.corrcoef(x, y)[0, 1])
    print(f"  Regression: a={slope:.5f}, b={intercept:.5f}, r={r:.3f}")

    # Compute residuals on all data
    df["ap_pred"]    = slope * df["AP_AVG"] + intercept
    df["delta_y_ap"] = df["density_ratio_msis"] - df["ap_pred"]

    # Daily residual & Ap for 1D panel
    daily["delta_ap"] = daily["ratio"] - (slope * daily["ap"] + intercept)
    s_delta = pd.Series(daily["delta_ap"].values, index=daily["date"])
    s_ap    = pd.Series(daily["ap"].values,        index=daily["date"])
    s_temp  = load_temp(ev)
    print(f"  ERA5 data points: {len(s_temp)}")

    # 2D grid
    date_edges = pd.date_range(
        ev["date_start"].normalize() - pd.Timedelta(hours=12),
        ev["date_end"].normalize()   + pd.Timedelta(hours=36),
        freq="D",
    )
    lat_edges = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    Z_detrend = compute_2d_grid(df, "delta_y_ap", date_edges, lat_edges)

    # Figure
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.6], hspace=0.22)
    ax1d         = fig.add_subplot(gs[0])
    ax2d_detrend = fig.add_subplot(gs[1], sharex=ax1d)

    # 1D Panel
    ax_t = ax1d.twinx()
    ax_a = ax1d.twinx()
    ax_a.spines["right"].set_position(("axes", 1.08))
    ax_a.spines["right"].set_visible(True)

    for r_start, r_end in ev["ref_dates"]:
        ax1d.axvspan(r_start, r_end, color="#008080", alpha=0.10, zorder=0)

    l1 = ax1d.plot(s_delta.index, s_delta.values, "o-",
                   color="#1f77b4", lw=2, ms=4.5,
                   label=r"$\Delta y_{Ap}$ (Left)")
    ax1d.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax1d.set_ylabel(r"$\Delta y_{Ap}$ (Density Ratio Residual)",
                    color="#1f77b4", fontweight="bold")
    ax1d.tick_params(axis="y", labelcolor="#1f77b4")

    l2 = []
    if len(s_temp) > 0:
        l2 = ax_t.plot(s_temp.index, s_temp.values, "^-",
                       color="#dc143c", lw=2, ms=4.5,
                       label=ev["temp_label"])
        ax_t.set_ylabel(f"{ev['temp_label']} [K]", color="#dc143c", fontweight="bold")
        ax_t.tick_params(axis="y", labelcolor="#dc143c")

    l3 = ax_a.plot(s_ap.index, s_ap.values, "s-",
                   color="#e67e22", lw=1.5, ms=4.5, alpha=0.85,
                   label="Daily mean Ap")
    ax_a.set_ylabel("Ap Index", color="#e67e22", fontweight="bold")
    ax_a.tick_params(axis="y", labelcolor="#e67e22")

    # SSW peak line(s)
    ax1d.axvline(ev["ssw_peak"], color="red", ls="--", lw=1.8)
    trans = mtransforms.blended_transform_factory(ax1d.transData, ax1d.transAxes)
    ax1d.text(
        ev["ssw_peak"], 0.97, f" {ev['ssw_peak_label']}",
        transform=trans, color="red", fontweight="bold",
        verticalalignment="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85),
    )
    # 2nd peak for 2024 SH dual-peak event
    if "ssw_peak2" in ev:
        ax1d.axvline(ev["ssw_peak2"], color="darkred", ls="--", lw=1.8)
        ax1d.text(
            ev["ssw_peak2"], 0.97, f" {ev['ssw_peak2_label']}",
            transform=trans, color="darkred", fontweight="bold",
            verticalalignment="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkred", alpha=0.85),
        )

    ann_text = f"slope a={slope:.5f}  intercept b={intercept:.5f}  corr(y,Ap)={r:.3f}"
    ax1d.text(0.015, 0.88, ann_text, transform=ax1d.transAxes,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
              fontsize=9)

    lines  = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1d.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.9)
    ax1d.grid(True, linestyle=":", alpha=0.5)
    ax1d.set_title(
        f"{ev['label']} — 1D Metrics & 2D Latitude-Time Density Distributions",
        fontsize=12, fontweight="bold", pad=14,
    )
    plt.setp(ax1d.get_xticklabels(), visible=False)

    # 2D Ap-Detrended Panel
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

    ax2d_detrend.axvline(mdates.date2num(ev["ssw_peak"]), color="red", ls="--", lw=1.8)
    if "ssw_peak2" in ev:
        ax2d_detrend.axvline(mdates.date2num(ev["ssw_peak2"]), color="darkred", ls="--", lw=1.8)

    ax2d_detrend.axhline(0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax2d_detrend.set_ylabel("Latitude [deg]", fontweight="bold")
    ax2d_detrend.set_ylim(LAT_MIN, LAT_MAX)
    ax2d_detrend.grid(True, linestyle=":", alpha=0.4)
    ax2d_detrend.set_title(
        r"(b) Ap-Detrended Residual $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$",
        fontsize=10, fontweight="bold", loc="left",
    )

    # X-axis date formatting
    ax2d_detrend.xaxis_date()
    ax2d_detrend.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2d_detrend.xaxis.set_major_locator(mdates.DayLocator(interval=ev["x_interval"]))
    ax2d_detrend.set_xlabel("Date (MM/DD)", fontweight="bold")

    # Align 1D panel width to 2D
    fig.canvas.draw()
    pos_2d = ax2d_detrend.get_position()
    pos_1d = ax1d.get_position()
    ax1d.set_position([pos_2d.x0, pos_1d.y0, pos_2d.width, pos_1d.height])

    plt.savefig(ev["out_png"], dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {ev['out_png']}")


def main() -> None:
    for ev in EVENTS:
        plot_event(ev)
    print("\nAll 3 events completed!")


if __name__ == "__main__":
    main()
