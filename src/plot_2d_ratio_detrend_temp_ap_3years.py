r"""
plot_2d_ratio_detrend_temp_ap_3years.py

Purpose:
    Create 2D (Latitude vs Time/Date) maps of Ap-detrended thermospheric density ratio residual
    \Delta y_{Ap} = y_i - (a * Ap_i + b), along with 1D time series of T(10 hPa) stratospheric temperature
    and Ap index for 3 major SSW events:
      1. 2018 NH SSW (SWARM-A)
      2. 2019 SH SSW (SWARM-A)
      3. 2021 NH SSW (SWARM-C)

Output:
    - Figure/Ap_removal/2D_ratio_detrend_temp_ap_2018.png
    - Figure/Ap_removal/2D_ratio_detrend_temp_ap_2019.png
    - Figure/Ap_removal/2D_ratio_detrend_temp_ap_2021.png
    - Figure/Ap_removal/2D_ratio_detrend_temp_ap_combined_3years.png
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# ─── File Paths ───────────────────────────────────────────────────────────────
P2018 = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")

COSMIC_2018 = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
ERA5_2019_DIR = Path("data/SSW2019/ERA5")
ERA5_2021_DIR = Path("data/SSW2021/ERA5")

OUT_DIR = Path("Figure/Ap_removal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Grid Settings ────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -60.0, 60.0
LAT_BIN = 3.0  # 3 degrees lat bin

# ─── Event Configurations ─────────────────────────────────────────────────────
EVENTS = [
    dict(
        year=2018,
        label="2018 NH SSW (SWARM-A)",
        sat="SWARM-A",
        parquet=P2018,
        mode="doy",
        date_start=pd.Timestamp("2018-01-30", tz="UTC"),
        date_end=pd.Timestamp("2018-03-06 23:59:59", tz="UTC"),
        doy_start=30, doy_end=65,
        ref_dates=[
            (pd.Timestamp("2018-01-30", tz="UTC"), pd.Timestamp("2018-02-09", tz="UTC")),
            (pd.Timestamp("2018-03-02", tz="UTC"), pd.Timestamp("2018-03-06", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2018-02-12", tz="UTC"),  # DOY 43
        ssw_peak_label="SSW Peak (DOY 43)",
        temp_type="cosmic",
        temp_file=COSMIC_2018,
        temp_label="T(10 hPa) 60–90°N",
    ),
    dict(
        year=2019,
        label="2019 SH SSW (SWARM-A)",
        sat="SWARM-A",
        parquet=P2019,
        mode="date",
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
        temp_label="T(10 hPa) 60–90°S",
    ),
    dict(
        year=2021,
        label="2021 NH SSW (SWARM-C)",
        sat="SWARM-C",
        parquet=P2021,
        mode="date",
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
        temp_label="T(10 hPa) 60–90°N",
    ),
]


# ─── Data Loaders & Calculations ──────────────────────────────────────────────
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
    else:  # era5
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


def compute_2d_grid(
    df: pd.DataFrame,
    val_col: str,
    date_edges: pd.DatetimeIndex,
    lat_edges: np.ndarray,
) -> np.ndarray:
    """Compute 2D median grid for (latitude, date)."""
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
    df_valid = df[valid]

    grouped = df_valid.groupby(["lat_i", "date_i"])[val_col].median()
    for (i, j), val in grouped.items():
        Z[i, j] = val

    return Z


# ─── Main Script ─────────────────────────────────────────────────────────────
def main() -> None:
    processed_events = []

    for ev in EVENTS:
        print(f"Processing {ev['label']}...")
        df = load_event_df(ev)

        daily = df.groupby("date").agg(
            ratio=("density_ratio_msis", "median"),
            ap=("AP_AVG", "mean"),
        ).reset_index()

        x = daily["ap"].values
        y = daily["ratio"].values
        p = np.polyfit(x, y, 1)
        slope, intercept = p[0], p[1]
        r = float(np.corrcoef(x, y)[0, 1])

        df["ap_pred"] = slope * df["AP_AVG"] + intercept
        df["delta_y_ap"] = df["density_ratio_msis"] - df["ap_pred"]

        daily["delta_ap"] = daily["ratio"] - (slope * daily["ap"] + intercept)
        s_delta = pd.Series(daily["delta_ap"].values, index=daily["date"])
        s_ap = pd.Series(daily["ap"].values, index=daily["date"])
        s_temp = load_temp(ev)

        dates_unique = pd.date_range(ev["date_start"].normalize(), ev["date_end"].normalize(), freq="D")
        date_edges = pd.date_range(
            ev["date_start"].normalize() - pd.Timedelta(hours=12),
            ev["date_end"].normalize() + pd.Timedelta(hours=36),
            freq="D"
        )
        lat_edges = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

        Z_raw = compute_2d_grid(df, "density_ratio_msis", date_edges, lat_edges)
        Z_detrend = compute_2d_grid(df, "delta_y_ap", date_edges, lat_edges)

        processed_events.append({
            "ev": ev,
            "df": df,
            "daily": daily,
            "s_delta": s_delta,
            "s_temp": s_temp,
            "s_ap": s_ap,
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "dates_unique": dates_unique,
            "date_edges": date_edges,
            "lat_edges": lat_edges,
            "Z_raw": Z_raw,
            "Z_detrend": Z_detrend,
        })

    # =========================================================================
    # 1. Create Individual 2D Plots for Each Year
    # =========================================================================
    for item in processed_events:
        ev = item["ev"]
        year = ev["year"]
        out_png = OUT_DIR / f"2D_ratio_detrend_temp_ap_{year}.png"
        print(f"Generating individual plot: {out_png}...")

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.3, 1.3], hspace=0.28)

        ax1d = fig.add_subplot(gs[0])
        ax2d_raw = fig.add_subplot(gs[1], sharex=ax1d)
        ax2d_detrend = fig.add_subplot(gs[2], sharex=ax1d)

        # ── 1D Top Panel ─────────────────────────────────────────────────────
        ax_t = ax1d.twinx()
        ax_a = ax1d.twinx()
        ax_a.spines["right"].set_position(("axes", 1.08))
        ax_a.spines["right"].set_visible(True)

        s_delta = item["s_delta"]
        s_temp  = item["s_temp"]
        s_ap    = item["s_ap"]

        for r_start, r_end in ev["ref_dates"]:
            ax1d.axvspan(r_start, r_end, color="#008080", alpha=0.10, zorder=0)

        l1 = ax1d.plot(s_delta.index, s_delta.values, "o-", color="#1f77b4", lw=2, ms=4.5, label=r"$\Delta y_{Ap}$ (Left)")
        ax1d.axhline(0.0, color="gray", ls=":", lw=1.0)
        ax1d.set_ylabel(r"$\Delta y_{Ap}$ (Density Ratio Residual)", color="#1f77b4", fontweight="bold")
        ax1d.tick_params(axis="y", labelcolor="#1f77b4")

        l2 = []
        if len(s_temp) > 0:
            l2 = ax_t.plot(s_temp.index, s_temp.values, "^-", color="#dc143c", lw=2, ms=4.5, label=ev["temp_label"])
            ax_t.set_ylabel(f"{ev['temp_label']} [K]", color="#dc143c", fontweight="bold")
            ax_t.tick_params(axis="y", labelcolor="#dc143c")

        l3 = ax_a.plot(s_ap.index, s_ap.values, "s-", color="#e67e22", lw=1.5, ms=4.5, alpha=0.85, label="Daily mean Ap")
        ax_a.set_ylabel("Ap Index", color="#e67e22", fontweight="bold")
        ax_a.tick_params(axis="y", labelcolor="#e67e22")

        # SSW Peak Line & Text
        ax1d.axvline(ev["ssw_peak"], color="red", ls="--", lw=1.8, label=ev["ssw_peak_label"])
        y_min, y_max = ax1d.get_ylim()
        ax1d.text(ev["ssw_peak"], y_min + 0.05 * (y_max - y_min), f" {ev['ssw_peak_label']}",
                  color="red", fontweight="bold", verticalalignment="bottom", fontsize=8.5,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85))

        ann_text = f"slope a={item['slope']:.5f}  intercept b={item['intercept']:.5f}  corr(y,Ap)={item['r']:.3f}"
        ax1d.text(0.015, 0.88, ann_text, transform=ax1d.transAxes,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85), fontsize=9)

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1d.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.9)
        ax1d.grid(True, linestyle=":", alpha=0.5)
        ax1d.set_title(f"{ev['label']} — 1D Metrics & 2D Latitude-Time Density Distributions", fontsize=12, fontweight="bold", pad=14)

        # ── 2D Middle Panel: Raw Density Ratio ─────────────────────────────────
        X_edges = mdates.date2num(item["date_edges"])
        Y_edges = item["lat_edges"]

        mesh_raw = ax2d_raw.pcolormesh(
            X_edges, Y_edges, item["Z_raw"],
            cmap="turbo", vmin=0.6, vmax=1.4, shading="flat"
        )
        cbar_raw = plt.colorbar(mesh_raw, ax=ax2d_raw, pad=0.015, aspect=15)
        cbar_raw.set_label(r"Raw Ratio $y_i = \rho_{\mathrm{obs}} / \rho_{\mathrm{MSIS}}$", fontsize=9, fontweight="bold")

        ax2d_raw.axvline(mdates.date2num(ev["ssw_peak"]), color="red", ls="--", lw=1.8)
        ax2d_raw.axhline(0, color="white", ls=":", lw=1.0, alpha=0.7)
        ax2d_raw.set_ylabel("Latitude [deg]", fontweight="bold")
        ax2d_raw.set_ylim(LAT_MIN, LAT_MAX)
        ax2d_raw.grid(True, linestyle=":", alpha=0.4, color="white")
        ax2d_raw.set_title(r"(a) Raw Density Ratio $y_i = \rho_{\mathrm{obs}} / \rho_{\mathrm{MSIS}}$", fontsize=10, fontweight="bold", loc="left")

        # ── 2D Bottom Panel: Ap-Detrended Density Residual ────────────────────
        mesh_detrend = ax2d_detrend.pcolormesh(
            X_edges, Y_edges, item["Z_detrend"],
            cmap="RdBu_r", vmin=-0.20, vmax=+0.20, shading="flat"
        )
        cbar_detrend = plt.colorbar(mesh_detrend, ax=ax2d_detrend, pad=0.015, aspect=15)
        cbar_detrend.set_label(r"Residual $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$", fontsize=9, fontweight="bold")

        # Contour lines for 0.0 and +-0.1
        X_centers = 0.5 * (X_edges[:-1] + X_edges[1:])
        Y_centers = 0.5 * (Y_edges[:-1] + Y_edges[1:])
        ax2d_detrend.contour(X_centers, Y_centers, item["Z_detrend"], levels=[0.0], colors="black", linewidths=0.8, linestyles="-", alpha=0.4)
        ax2d_detrend.contour(X_centers, Y_centers, item["Z_detrend"], levels=[-0.1, 0.1], colors="gray", linewidths=0.6, linestyles="--", alpha=0.4)

        ax2d_detrend.axvline(mdates.date2num(ev["ssw_peak"]), color="red", ls="--", lw=1.8)
        ax2d_detrend.axhline(0, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax2d_detrend.set_ylabel("Latitude [deg]", fontweight="bold")
        ax2d_detrend.set_ylim(LAT_MIN, LAT_MAX)
        ax2d_detrend.grid(True, linestyle=":", alpha=0.4)
        ax2d_detrend.set_title(r"(b) Ap-Detrended Residual $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$", fontsize=10, fontweight="bold", loc="left")

        # Date formatting for X-axis
        ax2d_detrend.xaxis_date()
        ax2d_detrend.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax2d_detrend.xaxis.set_major_locator(mdates.DayLocator(interval=4 if year != 2021 else 5))
        ax2d_detrend.set_xlabel("Date (MM/DD)", fontweight="bold")

        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # 2. Create 3-Year Combined 2D Figure (Vertical 3-panel layout)
    # =========================================================================
    out_comb = OUT_DIR / "2D_ratio_detrend_temp_ap_combined_3years.png"
    print(f"Generating combined 3-year 2D plot: {out_comb}...")

    fig = plt.figure(figsize=(15, 13))
    gs = gridspec.GridSpec(3, 1, hspace=0.38)

    fig.suptitle(
        r"Thermospheric Density Ratio Residual ($\Delta y_{Ap}$) 2D Maps (Latitude vs Date) & Stratospheric T(10 hPa) / Ap Index" "\n"
        r"Ap-Detrended Residual: $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$  |  Colormap centered at 0.0 (Red: Positive anomaly, Blue: Negative anomaly)",
        fontsize=12, fontweight="bold", y=0.988
    )

    for i, item in enumerate(processed_events):
        ev = item["ev"]
        
        gs_row = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i], height_ratios=[0.75, 1.4], hspace=0.18)
        
        ax1d = fig.add_subplot(gs_row[0])
        ax2d = fig.add_subplot(gs_row[1], sharex=ax1d)

        # ── 1D Overlays ──────────────────────────────────────────────────────
        ax_t = ax1d.twinx()
        ax_a = ax1d.twinx()
        ax_a.spines["right"].set_position(("axes", 1.07))
        ax_a.spines["right"].set_visible(True)

        s_delta = item["s_delta"]
        s_temp  = item["s_temp"]
        s_ap    = item["s_ap"]

        for r_start, r_end in ev["ref_dates"]:
            ax1d.axvspan(r_start, r_end, color="#008080", alpha=0.10, zorder=0)

        l1 = ax1d.plot(s_delta.index, s_delta.values, "o-", color="#1f77b4", lw=1.8, ms=3.5, label=r"$\Delta y_{Ap}$")
        ax1d.axhline(0.0, color="gray", ls=":", lw=1.0)
        ax1d.set_ylabel(r"$\Delta y_{Ap}$", color="#1f77b4", fontweight="bold", fontsize=9)
        ax1d.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8)

        l2 = []
        if len(s_temp) > 0:
            l2 = ax_t.plot(s_temp.index, s_temp.values, "^-", color="#dc143c", lw=1.8, ms=3.5, label=ev["temp_label"])
            ax_t.set_ylabel(f"{ev['temp_label']} [K]", color="#dc143c", fontweight="bold", fontsize=8)
            ax_t.tick_params(axis="y", labelcolor="#dc143c", labelsize=8)

        l3 = ax_a.plot(s_ap.index, s_ap.values, "s-", color="#e67e22", lw=1.2, ms=3.5, alpha=0.85, label="Ap index")
        ax_a.set_ylabel("Ap", color="#e67e22", fontweight="bold", fontsize=8)
        ax_a.tick_params(axis="y", labelcolor="#e67e22", labelsize=8)

        # SSW Peak
        ax1d.axvline(ev["ssw_peak"], color="red", ls="--", lw=1.5)
        y_min, y_max = ax1d.get_ylim()
        ax1d.text(ev["ssw_peak"], y_min + 0.05 * (y_max - y_min), f" {ev['ssw_peak_label']}",
                  color="red", fontweight="bold", fontsize=7.5, verticalalignment="bottom",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85))

        ann_text = f"a={item['slope']:.4f}, b={item['intercept']:.4f}, corr={item['r']:.3f}"
        ax1d.text(0.015, 0.85, ann_text, transform=ax1d.transAxes,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), fontsize=8)

        ax1d.grid(True, linestyle=":", alpha=0.4)
        ax1d.set_title(f"{ev['label']}", fontsize=11, fontweight="bold", loc="left", pad=12)
        plt.setp(ax1d.get_xticklabels(), visible=False)

        # ── 2D Residual Map ──────────────────────────────────────────────────
        X_edges = mdates.date2num(item["date_edges"])
        Y_edges = item["lat_edges"]

        mesh = ax2d.pcolormesh(
            X_edges, Y_edges, item["Z_detrend"],
            cmap="RdBu_r", vmin=-0.20, vmax=+0.20, shading="flat"
        )
        cbar = plt.colorbar(mesh, ax=ax2d, pad=0.015, aspect=12)
        cbar.set_label(r"Residual $\Delta y_{Ap}$", fontsize=8, fontweight="bold")
        cbar.ax.tick_params(labelsize=8)

        X_centers = 0.5 * (X_edges[:-1] + X_edges[1:])
        Y_centers = 0.5 * (Y_edges[:-1] + Y_edges[1:])
        ax2d.contour(X_centers, Y_centers, item["Z_detrend"], levels=[0.0], colors="black", linewidths=0.6, linestyles="-", alpha=0.35)

        ax2d.axvline(mdates.date2num(ev["ssw_peak"]), color="red", ls="--", lw=1.5)
        ax2d.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax2d.set_ylabel("Latitude [deg]", fontweight="bold", fontsize=9)
        ax2d.set_ylim(LAT_MIN, LAT_MAX)
        ax2d.grid(True, linestyle=":", alpha=0.3)
        ax2d.tick_params(labelsize=8)

        # Date X-axis formatting
        ax2d.xaxis_date()
        ax2d.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        if ev["year"] == 2018:
            ax2d.xaxis.set_major_locator(mdates.DayLocator(interval=4))
            ax2d.set_xlabel("DOY / Date 2018 (DOY 30–65)", fontweight="bold", fontsize=9)
        elif ev["year"] == 2019:
            ax2d.xaxis.set_major_locator(mdates.DayLocator(interval=4))
            ax2d.set_xlabel("Date 2019 (08/20 – 09/23)", fontweight="bold", fontsize=9)
        else:
            ax2d.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax2d.set_xlabel("Date 2020–2021 (12/25 – 02/05)", fontweight="bold", fontsize=9)

    plt.savefig(out_comb, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("All 2D plots completed successfully!")


if __name__ == "__main__":
    main()
