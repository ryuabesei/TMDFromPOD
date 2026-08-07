"""
plot_6panel_ratio_temp_2024_2026.py

Purpose:
    Generate 6-panel (3 rows x 2 cols) 1D density ratio (rho_obs / rho_MSIS) plots
    with ERA5 10 hPa temperature overlay for SWARM-A, B, C during 2024 and 2026 SSW events.

    Matches the exact layout and aesthetics requested by the user:
    - 3 rows: Latitude bands (High 40-60°, Mid 20-40°, Low 0-20°)
    - 2 cols: LT sectors (e.g. Morning / Evening or Nightside / Dayside)
    - Left Y axis: rho_ratio = rho_obs / rho_MSIS (blue/red lines with markers)
    - Right Y axis: ERA5 T (10 hPa, 60-90°N) [hotpink line with square markers]
    - Background shading: Non-SSW ref (lightblue) / SSW period (lightyellow)
    - Bottom legend: Non-SSW ref, SSW period, ERA5 T (10 hPa, 60-90 deg N)
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

    # Select temperature variable
    t_var = "t" if "t" in ds_all else ("temperature" if "temperature" in ds_all else None)
    if t_var is None:
        return pd.Series(dtype=float)

    # Latitude slicing / filtering
    lat_dim = "latitude" if "latitude" in ds_all.dims else ("lat" if "lat" in ds_all.dims else None)
    if lat_dim:
        lat_vals = ds_all[lat_dim].values
        if lat_vals[0] > lat_vals[-1]:  # descending
            ds_sel = ds_all.sel({lat_dim: slice(lat_max, lat_min)})
        else:
            ds_sel = ds_all.sel({lat_dim: slice(lat_min, lat_max)})
    else:
        ds_sel = ds_all

    # Mean over spatial dimensions
    spatial_dims = [d for d in ds_sel[t_var].dims if d != time_dim]
    t_mean = ds_sel[t_var].mean(dim=spatial_dims)

    # Convert to pandas Series & resample daily
    df_t = t_mean.to_dataframe()
    df_t.index = pd.to_datetime(df_t.index, utc=True)
    s_daily = df_t[t_var].resample("D").mean()
    s_daily.name = "T10hPa"
    return s_daily


def load_swarm_ratio(parquet_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    
    # Pick target column
    col = None
    for c in ["density_ratio_msis", "rho_ratio", "density_norm"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise KeyError(f"No density ratio column found in {parquet_path.name}")
    
    df["rho_ratio"] = df[col]
    df = df.dropna(subset=["datetime", "lat", "lst_h", "rho_ratio"])
    df = df[(df["datetime"] >= pd.Timestamp(start_date, tz="UTC")) & 
            (df["datetime"] <= pd.Timestamp(end_date, tz="UTC"))]
    df["date"] = df["datetime"].dt.floor("D")
    return df


# ============================================================
# Main Plotter for Single Satellite / Event
# ============================================================

def plot_6panel_satellite(
    event_year: str,
    sat_label: str,
    df: pd.DataFrame,
    lt_sectors: list[dict],
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

    n_rows = len(lat_bands)
    n_cols = len(lt_sectors)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.8 * n_cols, 3.4 * n_rows),
        sharex=True, sharey="row"
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.06)

    x_min = x_start - pd.Timedelta(hours=12)
    x_max = x_end + pd.Timedelta(hours=12)

    # Filter temperature series
    if not temp.empty:
        temp_sub = temp[(temp.index >= x_start) & (temp.index <= x_end)]
    else:
        temp_sub = pd.Series(dtype=float)

    for col_idx, lt in enumerate(lt_sectors):
        lt_label = lt["label"]
        lt_min   = lt["lt_min"]
        lt_max   = lt["lt_max"]
        wrap     = lt.get("wrap", False)
        line_color = lt["color"]

        if wrap:
            df_lt = df[(df["lst_h"] >= lt_min) | (df["lst_h"] < lt_max)]
        else:
            df_lt = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)]

        for row_idx, (band_label, lat_lo, lat_hi) in enumerate(lat_bands):
            ax = axes[row_idx, col_idx]

            # Subset data for latitude band
            mask = (df_lt["lat"].abs() >= lat_lo) & (df_lt["lat"].abs() < lat_hi)
            sub  = df_lt[mask]
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

            # Lat band label top-left
            ax.text(
                0.01, 0.97, band_label.strip(),
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left"
            )

            # Column title (top row only)
            if row_idx == 0:
                ax.set_title(lt_label, fontsize=11, fontweight="bold", pad=6, color=line_color if col_idx == 1 and line_color != "#1f77b4" else "black")

            # X label (bottom row only)
            if row_idx == n_rows - 1:
                ax.set_xlabel(f"Date ({event_year})", fontsize=10)

    # --- Common Legend at Bottom ---
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

    lt_title_str = "   |   ".join(lt["label"] for lt in lt_sectors)
    fig.suptitle(
        f"{sat_label}  density_ratio_msis ({event_year} SSW)\n{lt_title_str}",
        fontsize=11, fontweight="bold", y=1.02
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================
# Execution Setup for 2024 and 2026
# ============================================================

def run_2024_6panel():
    print("\n=== Processing 2024 6-Panel Plots ===")
    era5_dir = Path("data/SSW2024/ERA5")
    temp = load_era5_temp(era5_dir, 60.0, 90.0)

    x_start  = pd.Timestamp("2023-12-22", tz="UTC")
    x_end    = pd.Timestamp("2024-02-28", tz="UTC")
    ref1     = (pd.Timestamp("2023-12-22", tz="UTC"), pd.Timestamp("2024-01-04", tz="UTC"))
    ssw_s    = pd.Timestamp("2024-01-05", tz="UTC")
    ssw_e    = pd.Timestamp("2024-01-30", tz="UTC")
    ref2     = (pd.Timestamp("2024-01-31", tz="UTC"), pd.Timestamp("2024-02-28", tz="UTC"))

    sats = [
        dict(
            label="SWARM-A",
            parquet=Path("normalizeddata/2024/swarm_dnsapod_2024_normalized_with_LT.parquet"),
            out_png=Path("Figure/2024/1D_ratio_msis_2024_SWARM-A_by_LT.png"),
            lt_sectors=[
                dict(label="Morning (LT 6-12h)", lt_min=4, lt_max=11, color="#1f77b4"),
                dict(label="Evening (LT 18-24h)", lt_min=16, lt_max=23, color="#c0392b"),
            ]
        ),
        dict(
            label="SWARM-B",
            parquet=Path("normalizeddata/2024/swarm_dnsbpod_2024_normalized_with_LT.parquet"),
            out_png=Path("Figure/2024/1D_ratio_msis_2024_SWARM-B_by_LT.png"),
            lt_sectors=[
                dict(label="Nightside (LT 22-05h)", lt_min=22, lt_max=5, wrap=True, color="#6a0dad"),
                dict(label="Dayside (LT 11-17h)", lt_min=11, lt_max=17, color="#c0392b"),
            ]
        ),
        dict(
            label="SWARM-C",
            parquet=Path("normalizeddata/2024/swarm_dnscpod_2024_normalized_with_LT.parquet"),
            out_png=Path("Figure/2024/1D_ratio_msis_2024_SWARM-C_by_LT.png"),
            lt_sectors=[
                dict(label="Morning (LT 6-12h)", lt_min=4, lt_max=11, color="#1f77b4"),
                dict(label="Evening (LT 18-24h)", lt_min=16, lt_max=23, color="#c0392b"),
            ]
        ),
    ]

    for sat in sats:
        print(f"  Generating {sat['label']} ...")
        df = load_swarm_ratio(sat["parquet"], "2023-12-15", "2024-03-01")
        plot_6panel_satellite(
            event_year="2024",
            sat_label=sat["label"],
            df=df,
            lt_sectors=sat["lt_sectors"],
            temp=temp,
            x_start=x_start, x_end=x_end,
            ref1=ref1, ref2=ref2,
            ssw_start=ssw_s, ssw_end=ssw_e,
            out_png=sat["out_png"]
        )


def run_2026_6panel():
    print("\n=== Processing 2026 6-Panel Plots ===")
    era5_dir = Path("data/SSW2026/ERA5")
    temp = load_era5_temp(era5_dir, 60.0, 90.0)

    x_start  = pd.Timestamp("2025-12-20", tz="UTC")
    x_end    = pd.Timestamp("2026-03-20", tz="UTC")
    ref1     = (pd.Timestamp("2025-12-20", tz="UTC"), pd.Timestamp("2026-01-10", tz="UTC"))
    ssw_s    = pd.Timestamp("2026-01-25", tz="UTC")
    ssw_e    = pd.Timestamp("2026-02-25", tz="UTC")
    ref2     = (pd.Timestamp("2026-02-26", tz="UTC"), pd.Timestamp("2026-03-15", tz="UTC"))

    sats = [
        dict(
            label="SWARM-A",
            parquet=Path("normalizeddata/2026/swarm_dnsapod_2026_normalized_with_LT.parquet"),
            out_png=Path("Figure/2026/1D_ratio_msis_2026_SWARM-A_by_LT.png"),
            lt_sectors=[
                dict(label="Morning (LT 6-12h)", lt_min=4, lt_max=11, color="#1f77b4"),
                dict(label="Evening (LT 18-24h)", lt_min=16, lt_max=23, color="#c0392b"),
            ]
        ),
        dict(
            label="SWARM-B",
            parquet=Path("normalizeddata/2026/swarm_dnsbpod_2026_normalized_with_LT.parquet"),
            out_png=Path("Figure/2026/1D_ratio_msis_2026_SWARM-B_by_LT.png"),
            lt_sectors=[
                dict(label="Nightside (LT 22-05h)", lt_min=22, lt_max=5, wrap=True, color="#6a0dad"),
                dict(label="Dayside (LT 11-17h)", lt_min=11, lt_max=17, color="#c0392b"),
            ]
        ),
        dict(
            label="SWARM-C",
            parquet=Path("normalizeddata/2026/swarm_dnscpod_2026_normalized_with_LT.parquet"),
            out_png=Path("Figure/2026/1D_ratio_msis_2026_SWARM-C_by_LT.png"),
            lt_sectors=[
                dict(label="Morning (LT 6-12h)", lt_min=4, lt_max=11, color="#1f77b4"),
                dict(label="Evening (LT 18-24h)", lt_min=16, lt_max=23, color="#c0392b"),
            ]
        ),
    ]

    for sat in sats:
        print(f"  Generating {sat['label']} ...")
        df = load_swarm_ratio(sat["parquet"], "2025-12-15", "2026-03-25")
        plot_6panel_satellite(
            event_year="2026",
            sat_label=sat["label"],
            df=df,
            lt_sectors=sat["lt_sectors"],
            temp=temp,
            x_start=x_start, x_end=x_end,
            ref1=ref1, ref2=ref2,
            ssw_start=ssw_s, ssw_end=ssw_e,
            out_png=sat["out_png"]
        )


def main():
    run_2024_6panel()
    run_2026_6panel()
    print("\nAll 2024 and 2026 6-panel plots generated successfully!")

if __name__ == "__main__":
    main()
