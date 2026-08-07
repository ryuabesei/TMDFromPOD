"""
plot_6panel_ratio_temp_2024_2026_fixed.py

Purpose:
    Fix the data drop-out (linear line interpolation) issue for SWARM-B (and other satellites)
    caused by rigid Local Time (LT) filtering thresholds.

    Instead of applying strict static LT bounds (e.g. 22-05h and 11-17h) which fail as
    the satellite orbit precesses across 2024 and 2026, this script separates the satellite's
    two orbital passes using Ascending (Northbound) and Descending (Southbound) orbital nodes.

    This guarantees 100% complete daily coverage without artificial data gaps or straight lines.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr

# ============================================================
# Helper Functions
# ============================================================

def load_era5_temp(era5_dir: Path, lat_min: float = 60.0, lat_max: float = 90.0) -> pd.Series:
    """Load ERA5 10 hPa temperature NetCDF files and average over specified latitude band."""
    nc_files = sorted(list(era5_dir.glob("*.nc")))
    if not nc_files:
        return pd.Series(dtype=float)
    
    ds_list = []
    for f in nc_files:
        try:
            ds = xr.open_dataset(f)
            ds_list.append(ds)
        except Exception as e:
            print(f"    [WARN] Failed to read {f.name}: {e}")
    if not ds_list:
        return pd.Series(dtype=float)

    ds_all = xr.concat(ds_list, dim="valid_time") if "valid_time" in ds_list[0].dims else xr.concat(ds_list, dim="time")
    time_dim = "valid_time" if "valid_time" in ds_all.dims else "time"

    t_var = "t" if "t" in ds_all else ("temperature" if "temperature" in ds_all else None)
    if t_var is None:
        return pd.Series(dtype=float)

    lat_dim = "latitude" if "latitude" in ds_all.dims else ("lat" if "lat" in ds_all.dims else None)
    if lat_dim:
        lat_vals = ds_all[lat_dim].values
        if lat_vals[0] > lat_vals[-1]:  # descending
            ds_sel = ds_all.sel({lat_dim: slice(lat_max, lat_min)})
        else:
            ds_sel = ds_all.sel({lat_dim: slice(lat_min, lat_max)})
    else:
        ds_sel = ds_all

    spatial_dims = [d for d in ds_sel[t_var].dims if d != time_dim]
    t_mean = ds_sel[t_var].mean(dim=spatial_dims)

    df_t = t_mean.to_dataframe()
    df_t.index = pd.to_datetime(df_t.index, utc=True)
    s_daily = df_t[t_var].resample("D").mean()
    s_daily.name = "T10hPa"
    return s_daily


def load_swarm_ratio_and_split_nodes(parquet_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load Swarm parquet, filter dates, and split orbit into Ascending (dlat>0) and Descending (dlat<0) nodes."""
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    
    col = None
    for c in ["density_ratio_msis", "rho_ratio", "density_norm"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise KeyError(f"No density ratio column found in {parquet_path.name}")
    
    df["rho_ratio"] = df[col]
    df = df.dropna(subset=["datetime", "lat", "lst_h", "rho_ratio"]).sort_values("datetime").reset_index(drop=True)
    df = df[(df["datetime"] >= pd.Timestamp(start_date, tz="UTC")) & 
            (df["datetime"] <= pd.Timestamp(end_date, tz="UTC"))].copy()
    df["date"] = df["datetime"].dt.floor("D")

    # Determine orbit direction (dlat/dt)
    lat_diff = df["lat"].diff()
    # Fill first NaN with second value
    lat_diff = lat_diff.bfill()
    
    # 1: Ascending (Northbound, dlat >= 0), 2: Descending (Southbound, dlat < 0)
    df["node"] = np.where(lat_diff >= 0, 1, 2)
    return df


# ============================================================
# Main Plotter for Single Satellite / Event
# ============================================================

def plot_6panel_satellite_fixed(
    event_year: str,
    sat_label: str,
    df: pd.DataFrame,
    temp: pd.Series,
    x_start: pd.Timestamp,
    x_end: pd.Timestamp,
    ref1: tuple[pd.Timestamp, pd.Timestamp],
    ref2: tuple[pd.Timestamp, pd.Timestamp],
    ssw_start: pd.Timestamp,
    ssw_end: pd.Timestamp,
    out_png: Path,
) -> None:
    lat_bands = [
        ("High (40-60°)", 40.0, 60.0),
        ("Mid  (20-40°)", 20.0, 40.0),
        ("Low  ( 0-20°)",  0.0, 20.0),
    ]

    df_period = df[(df["datetime"] >= x_start) & (df["datetime"] <= x_end)]
    node1_lst = df_period[df_period["node"] == 1]["lst_h"]
    node2_lst = df_period[df_period["node"] == 2]["lst_h"]

    node1_min, node1_max = node1_lst.min(), node1_lst.max()
    node2_min, node2_max = node2_lst.min(), node2_lst.max()

    node1_label = f"Ascending Pass (LT {node1_min:.1f}–{node1_max:.1f}h)"
    node2_label = f"Descending Pass (LT {node2_min:.1f}–{node2_max:.1f}h)"

    nodes = [
        dict(node_id=1, label=node1_label, color="#1f77b4"),
        dict(node_id=2, label=node2_label, color="#c0392b"),
    ]

    n_rows = len(lat_bands)
    n_cols = len(nodes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.8 * n_cols, 3.4 * n_rows),
        sharex=True, sharey="row"
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.06)

    x_min = x_start - pd.Timedelta(hours=12)
    x_max = x_end + pd.Timedelta(hours=12)

    if not temp.empty:
        temp_sub = temp[(temp.index >= x_start) & (temp.index <= x_end)]
    else:
        temp_sub = pd.Series(dtype=float)

    for col_idx, nd in enumerate(nodes):
        node_id    = nd["node_id"]
        node_lbl   = nd["label"]
        line_color = nd["color"]

        df_node = df_period[df_period["node"] == node_id]

        for row_idx, (band_label, lat_lo, lat_hi) in enumerate(lat_bands):
            ax = axes[row_idx, col_idx]

            mask = (df_node["lat"].abs() >= lat_lo) & (df_node["lat"].abs() < lat_hi)
            sub  = df_node[mask]
            daily = sub.groupby("date")["rho_ratio"].median()

            # --- Background Shading ---
            ax.axvspan(ref1[0], ref1[1], color="lightblue", alpha=0.20)
            ax.axvspan(ref2[0], ref2[1], color="lightblue", alpha=0.20)
            ax.axvspan(ssw_start, ssw_end, color="lightyellow", alpha=0.40)

            # --- Left Y Axis: rho_ratio ---
            if len(daily) > 0:
                ax.plot(
                    daily.index, daily.values,
                    color=line_color, linewidth=2.0, marker="o", markersize=4,
                    zorder=4, label="rho_ratio"
                )

            # --- Right Y Axis: ERA5 T (10 hPa) ---
            if len(temp_sub) > 0:
                ax2 = ax.twinx()
                ax2.plot(
                    temp_sub.index, temp_sub.values,
                    color="hotpink", linewidth=1.6, linestyle="-",
                    marker="s", markersize=3, alpha=0.85, zorder=3,
                    label="ERA5 T (10 hPa, 60-90°N)"
                )
                t_vals = temp_sub.values[~np.isnan(temp_sub.values)]
                if len(t_vals) > 0:
                    t_margin = max((t_vals.max() - t_vals.min()) * 0.25, 4.0)
                    ax2.set_ylim(t_vals.min() - t_margin, t_vals.max() + t_margin)
                ax2.tick_params(axis="y", labelcolor="hotpink", labelsize=8)

                if col_idx == n_cols - 1:
                    ax2.set_ylabel("T (10 hPa) [K]", fontsize=9, color="hotpink")
                else:
                    ax2.set_yticklabels([])

            # Decoration
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(axis="y", alpha=0.3, linewidth=0.7)
            ax.tick_params(axis="both", labelsize=9)

            if col_idx == 0:
                ax.set_ylabel("rho_ratio\n(rho_obs / rho_MSIS)", fontsize=9)

            ax.text(
                0.01, 0.97, band_label.strip(),
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left"
            )

            if row_idx == 0:
                ax.set_title(node_lbl, fontsize=11, fontweight="bold", pad=6, color=line_color)

            if row_idx == n_rows - 1:
                ax.set_xlabel(f"Date ({event_year})", fontsize=10)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc="lightblue", alpha=0.4, label="Non-SSW ref"),
        plt.Rectangle((0, 0), 1, 1, fc="lightyellow", alpha=0.6, label="SSW period"),
        plt.Line2D([0], [0], color="hotpink", lw=1.6, marker="s", ms=4, label="ERA5 T (10 hPa, 60-90 deg N)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=3, fontsize=9,
        framealpha=0.85, bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(
        f"{sat_label}  density_ratio_msis ({event_year} SSW)\n{node1_label}   |   {node2_label}",
        fontsize=11, fontweight="bold", y=1.02
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================
# Execution Setup for 2024 and 2026
# ============================================================

def run_2024_fixed():
    print("\n=== Processing 2024 6-Panel Plots (Orbit Node Separation) ===")
    era5_dir = Path("data/SSW2024/ERA5")
    temp = load_era5_temp(era5_dir, 60.0, 90.0)

    x_start  = pd.Timestamp("2023-12-22", tz="UTC")
    x_end    = pd.Timestamp("2024-02-28", tz="UTC")
    ref1     = (pd.Timestamp("2023-12-22", tz="UTC"), pd.Timestamp("2024-01-04", tz="UTC"))
    ssw_s    = pd.Timestamp("2024-01-05", tz="UTC")
    ssw_e    = pd.Timestamp("2024-01-30", tz="UTC")
    ref2     = (pd.Timestamp("2024-01-31", tz="UTC"), pd.Timestamp("2024-02-28", tz="UTC"))

    sats = [
        dict(label="SWARM-A", parquet=Path("normalizeddata/2024/swarm_dnsapod_2024_normalized_with_LT.parquet"), out_png=Path("Figure/2024/1D_ratio_msis_2024_SWARM-A_by_LT.png")),
        dict(label="SWARM-B", parquet=Path("normalizeddata/2024/swarm_dnsbpod_2024_normalized_with_LT.parquet"), out_png=Path("Figure/2024/1D_ratio_msis_2024_SWARM-B_by_LT.png")),
        dict(label="SWARM-C", parquet=Path("normalizeddata/2024/swarm_dnscpod_2024_normalized_with_LT.parquet"), out_png=Path("Figure/2024/1D_ratio_msis_2024_SWARM-C_by_LT.png")),
    ]

    for sat in sats:
        print(f"  Generating {sat['label']} ...")
        df = load_swarm_ratio_and_split_nodes(sat["parquet"], "2023-12-15", "2024-03-01")
        plot_6panel_satellite_fixed(
            event_year="2024",
            sat_label=sat["label"],
            df=df,
            temp=temp,
            x_start=x_start, x_end=x_end,
            ref1=ref1, ref2=ref2,
            ssw_start=ssw_s, ssw_end=ssw_e,
            out_png=sat["out_png"]
        )


def run_2026_fixed():
    print("\n=== Processing 2026 6-Panel Plots (Orbit Node Separation) ===")
    era5_dir = Path("data/SSW2026/ERA5")
    temp = load_era5_temp(era5_dir, 60.0, 90.0)

    x_start  = pd.Timestamp("2025-12-20", tz="UTC")
    x_end    = pd.Timestamp("2026-03-20", tz="UTC")
    ref1     = (pd.Timestamp("2025-12-20", tz="UTC"), pd.Timestamp("2026-01-10", tz="UTC"))
    ssw_s    = pd.Timestamp("2026-01-25", tz="UTC")
    ssw_e    = pd.Timestamp("2026-02-25", tz="UTC")
    ref2     = (pd.Timestamp("2026-02-26", tz="UTC"), pd.Timestamp("2026-03-15", tz="UTC"))

    sats = [
        dict(label="SWARM-A", parquet=Path("normalizeddata/2026/swarm_dnsapod_2026_normalized_with_LT.parquet"), out_png=Path("Figure/2026/1D_ratio_msis_2026_SWARM-A_by_LT.png")),
        dict(label="SWARM-B", parquet=Path("normalizeddata/2026/swarm_dnsbpod_2026_normalized_with_LT.parquet"), out_png=Path("Figure/2026/1D_ratio_msis_2026_SWARM-B_by_LT.png")),
        dict(label="SWARM-C", parquet=Path("normalizeddata/2026/swarm_dnscpod_2026_normalized_with_LT.parquet"), out_png=Path("Figure/2026/1D_ratio_msis_2026_SWARM-C_by_LT.png")),
    ]

    for sat in sats:
        print(f"  Generating {sat['label']} ...")
        df = load_swarm_ratio_and_split_nodes(sat["parquet"], "2025-12-15", "2026-03-25")
        plot_6panel_satellite_fixed(
            event_year="2026",
            sat_label=sat["label"],
            df=df,
            temp=temp,
            x_start=x_start, x_end=x_end,
            ref1=ref1, ref2=ref2,
            ssw_start=ssw_s, ssw_end=ssw_e,
            out_png=sat["out_png"]
        )


def main():
    run_2024_fixed()
    run_2026_fixed()
    print("\nAll fixed 6-panel plots generated successfully!")

if __name__ == "__main__":
    main()
