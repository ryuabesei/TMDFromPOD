r"""
plot_2d_detrend_only_2018.py

Purpose:
    2018 NH SSW (SWARM-A) version of the 2D plot WITHOUT panel (a) Raw Density Ratio.
    Layout: 2 rows
        Row 0 : 1D time series  (Delta y_Ap, T(10 hPa), Ap)
        Row 1 : (b) Ap-Detrended Residual  Delta y_Ap = y_i - (a*Ap_i + b)  [RdBu_r]

Output:
    Figure/Ap_removal/2D_detrend_only_temp_ap_2018.png
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
import matplotlib.transforms as mtransforms

# ─── Paths ────────────────────────────────────────────────────────────────────
P2018       = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
COSMIC_2018 = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
OUT_DIR     = Path("Figure/Ap_removal")
OUT_PNG     = OUT_DIR / "2D_detrend_only_temp_ap_2018.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Grid Settings ────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -60.0, 60.0
LAT_BIN = 3.0

# ─── Event Config ─────────────────────────────────────────────────────────────
EV = dict(
    year=2018,
    label="2018 NH SSW (SWARM-A)",
    date_start=pd.Timestamp("2018-01-30", tz="UTC"),
    date_end=pd.Timestamp("2018-03-06 23:59:59", tz="UTC"),
    ref_dates=[
        (pd.Timestamp("2018-01-30", tz="UTC"), pd.Timestamp("2018-02-09", tz="UTC")),
        (pd.Timestamp("2018-03-02", tz="UTC"), pd.Timestamp("2018-03-06", tz="UTC")),
    ],
    ssw_peak=pd.Timestamp("2018-02-12", tz="UTC"),
    ssw_peak_label="SSW Peak (DOY 43)",
    temp_label="T(10 hPa) 60–90°N",
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_density(ev: dict) -> pd.DataFrame:
    df = pd.read_parquet(P2018)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for cname in ["lat", "latitude", "geod_lat"]:
        if cname in df.columns and cname != "lat":
            df = df.rename(columns={cname: "lat"})
            break
    df = df.dropna(subset=["datetime", "lat", "density_ratio_msis", "AP_AVG"])
    df = df[(df["datetime"] >= ev["date_start"]) & (df["datetime"] <= ev["date_end"])].copy()
    df["date"] = df["datetime"].dt.normalize()
    return df


def load_temp_cosmic(ev: dict) -> pd.Series:
    df = pd.read_csv(COSMIC_2018, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["date"] = df["datetime"].dt.normalize()
    df = df[(df["datetime"] >= ev["date_start"]) & (df["datetime"] <= ev["date_end"])]
    return df.groupby("date")["T10_K"].mean().sort_index()


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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ev = EV
    print(f"Processing {ev['label']}...")

    df = load_density(ev)

    daily = df.groupby("date").agg(
        ratio=("density_ratio_msis", "median"),
        ap=("AP_AVG", "mean"),
    ).reset_index()

    x = daily["ap"].values
    y = daily["ratio"].values
    p = np.polyfit(x, y, 1)
    slope, intercept = p[0], p[1]
    r = float(np.corrcoef(x, y)[0, 1])

    df["ap_pred"]   = slope * df["AP_AVG"] + intercept
    df["delta_y_ap"] = df["density_ratio_msis"] - df["ap_pred"]

    daily["delta_ap"] = daily["ratio"] - (slope * daily["ap"] + intercept)
    s_delta = pd.Series(daily["delta_ap"].values, index=daily["date"])
    s_ap    = pd.Series(daily["ap"].values,        index=daily["date"])
    s_temp  = load_temp_cosmic(ev)

    date_edges = pd.date_range(
        ev["date_start"].normalize() - pd.Timedelta(hours=12),
        ev["date_end"].normalize()   + pd.Timedelta(hours=36),
        freq="D",
    )
    lat_edges  = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    Z_detrend  = compute_2d_grid(df, "delta_y_ap", date_edges, lat_edges)

    # ── Figure: 2 rows (1D + 2D detrend only) ─────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.6], hspace=0.22)

    ax1d        = fig.add_subplot(gs[0])
    ax2d_detrend = fig.add_subplot(gs[1], sharex=ax1d)

    # ── 1D Panel ──────────────────────────────────────────────────────────────
    ax_t = ax1d.twinx()
    ax_a = ax1d.twinx()
    ax_a.spines["right"].set_position(("axes", 1.08))
    ax_a.spines["right"].set_visible(True)

    for r_start, r_end in ev["ref_dates"]:
        ax1d.axvspan(r_start, r_end, color="#008080", alpha=0.10, zorder=0)

    l1 = ax1d.plot(s_delta.index, s_delta.values, "o-", color="#1f77b4", lw=2, ms=4.5,
                   label=r"$\Delta y_{Ap}$ (Left)")
    ax1d.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax1d.set_ylabel(r"$\Delta y_{Ap}$ (Density Ratio Residual)", color="#1f77b4", fontweight="bold")
    ax1d.tick_params(axis="y", labelcolor="#1f77b4")

    l2 = []
    if len(s_temp) > 0:
        l2 = ax_t.plot(s_temp.index, s_temp.values, "^-", color="#dc143c", lw=2, ms=4.5,
                       label=ev["temp_label"])
        ax_t.set_ylabel(f"{ev['temp_label']} [K]", color="#dc143c", fontweight="bold")
        ax_t.tick_params(axis="y", labelcolor="#dc143c")

    l3 = ax_a.plot(s_ap.index, s_ap.values, "s-", color="#e67e22", lw=1.5, ms=4.5, alpha=0.85,
                   label="Daily mean Ap")
    ax_a.set_ylabel("Ap Index", color="#e67e22", fontweight="bold")
    ax_a.tick_params(axis="y", labelcolor="#e67e22")

    # SSW peak vertical line + label at top of panel (no overlap with data)
    ax1d.axvline(ev["ssw_peak"], color="red", ls="--", lw=1.8)
    # blended transform: x in data coords, y in axes coords (0=bottom, 1=top)
    trans = mtransforms.blended_transform_factory(ax1d.transData, ax1d.transAxes)
    ax1d.text(
        ev["ssw_peak"], 0.97,
        f" {ev['ssw_peak_label']}",
        transform=trans,
        color="red", fontweight="bold",
        verticalalignment="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85),
    )

    ann_text = f"slope a={slope:.5f}  intercept b={intercept:.5f}  corr(y,Ap)={r:.3f}"
    ax1d.text(0.015, 0.88, ann_text, transform=ax1d.transAxes,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85), fontsize=9)

    lines  = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1d.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.9)
    ax1d.grid(True, linestyle=":", alpha=0.5)
    ax1d.set_title(f"{ev['label']} — 1D Metrics & 2D Latitude-Time Density Distributions",
                   fontsize=12, fontweight="bold", pad=14)
    plt.setp(ax1d.get_xticklabels(), visible=False)

    # ── 2D Ap-Detrended Panel ─────────────────────────────────────────────────
    X_edges = mdates.date2num(date_edges)
    Y_edges = lat_edges

    mesh_detrend = ax2d_detrend.pcolormesh(
        X_edges, Y_edges, Z_detrend,
        cmap="RdBu_r", vmin=-0.20, vmax=+0.20, shading="flat"
    )
    cbar_detrend = plt.colorbar(mesh_detrend, ax=ax2d_detrend, pad=0.015, aspect=15)
    cbar_detrend.set_label(r"Residual $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$",
                           fontsize=9, fontweight="bold")

    # Contour lines
    X_centers = 0.5 * (X_edges[:-1] + X_edges[1:])
    Y_centers = 0.5 * (Y_edges[:-1] + Y_edges[1:])
    ax2d_detrend.contour(X_centers, Y_centers, Z_detrend,
                         levels=[0.0], colors="black", linewidths=0.8, linestyles="-", alpha=0.4)
    ax2d_detrend.contour(X_centers, Y_centers, Z_detrend,
                         levels=[-0.1, 0.1], colors="gray", linewidths=0.6, linestyles="--", alpha=0.4)

    ax2d_detrend.axvline(mdates.date2num(ev["ssw_peak"]), color="red", ls="--", lw=1.8)
    ax2d_detrend.axhline(0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax2d_detrend.set_ylabel("Latitude [deg]", fontweight="bold")
    ax2d_detrend.set_ylim(LAT_MIN, LAT_MAX)
    ax2d_detrend.grid(True, linestyle=":", alpha=0.4)
    ax2d_detrend.set_title(
        r"(b) Ap-Detrended Residual $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$",
        fontsize=10, fontweight="bold", loc="left"
    )

    # Date formatting
    ax2d_detrend.xaxis_date()
    ax2d_detrend.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2d_detrend.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax2d_detrend.set_xlabel("Date (MM/DD)", fontweight="bold")

    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
