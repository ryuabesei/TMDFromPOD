# plot_1D_ratio_temp_Kp_SH_2024.py
"""
Plot Southern Hemisphere (SH) 2024 Sudden Stratospheric Warming (SSW)
using SWARM density ratio (rho_ratio), ERA5 10 hPa temperature (-60°–-90° latitude)
and daily‑mean Ap (proxy for Kp).
The script mirrors `plot_1D_ratio_temp_Kp_3events.py` but focuses on the SH
SSW events of July–August 2024.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# ─── Constants ───────────────────────────────────────────────────────────────
VALUE_COL_CANDIDATES = ["density_ratio_msis", "rho_ratio", "density_norm"]
AP_KP3 = 15.0  # Kp=3 ↔ Ap=15

# Latitude bands – absolute latitude is used, so the same bands apply to SH
LAT_BANDS = [
    ("High (40–60°)", 40.0, 60.0, "#1f77b4"),
    ("Mid  (20–40°)", 20.0, 40.0, "#2ca02c"),
    ("Low  ( 0–20°)", 0.0, 20.0, "#ff7f0e"),
]

OUT_BASE = Path("Figure/2024/sh")

# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_kp(kp_csv: Path) -> pd.Series:
    df = pd.read_csv(kp_csv, parse_dates=["DATE"])
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
    df = df.set_index("DATE").sort_index()
    return df["AP_AVG"].astype(float)


def load_era5_temp(era5_dir: Path, lat_min: float, lat_max: float) -> pd.Series:
    """Load all *.nc in `era5_dir`, area‑average T10 hPa over [lat_min, lat_max]."""
    files = sorted(era5_dir.glob("*.nc"))
    series_list = []
    for fp in files:
        with xr.open_dataset(fp) as ds:
            lo, hi = min(lat_min, lat_max), max(lat_min, lat_max)
            ds_sub = ds.sel(latitude=slice(hi, lo))  # ERA5 latitude decreasing
            weights = np.cos(np.deg2rad(ds_sub["latitude"]))
            w_temp = ds_sub["t"].weighted(weights)
            t_avg = w_temp.mean(dim=["latitude", "longitude"]).squeeze()
            times = pd.to_datetime(t_avg["valid_time"].values)
            vals = t_avg.values.ravel()
            s = pd.Series(vals, index=times)
            s.index = s.index.tz_localize("UTC")
            series_list.append(s)
    combined = pd.concat(series_list).sort_index()
    combined = combined.resample("D").mean()
    return combined


def load_ratio(parquet: Path, t_start: str, t_end: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    # Normalise latitude column name
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col and lat_col != "lat":
        df = df.rename(columns={lat_col: "lat"})
    # Resolve ratio column
    value_col = next((c for c in VALUE_COL_CANDIDATES if c in df.columns), None)
    if value_col is None:
        raise KeyError(f"No ratio column found in {parquet}. Available: {list(df.columns)}")
    if value_col != "density_ratio_msis":
        df["density_ratio_msis"] = df[value_col]
    df = df.dropna(subset=["datetime", "lat", "density_ratio_msis", "lst_h"])
    df = df[(df["datetime"] >= pd.Timestamp(t_start, tz="UTC")) &
            (df["datetime"] <= pd.Timestamp(t_end, tz="UTC"))]
    df["date"] = df["datetime"].dt.normalize()
    return df

# ─── Plot function ─────────────────────────────────────────────────────────────

def plot_event_satellite(
    event_label: str,
    sat_label: str,
    df: pd.DataFrame,
    lt_sectors: list,
    temp: pd.Series,
    kp: pd.Series,
    x_start: pd.Timestamp,
    x_end: pd.Timestamp,
    ssw_starts: list[pd.Timestamp],
    ssw_ends: list[pd.Timestamp],
    ssw_peaks: list[pd.Timestamp],
    ref_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    temp_region_label: str,
    out_png: Path,
) -> None:
    n_lt = len(lt_sectors)
    fig, axes = plt.subplots(1, n_lt, figsize=(8 * n_lt, 6), gridspec_kw={"wspace": 0.30})
    if n_lt == 1:
        axes = [axes]

    for ci, lt in enumerate(lt_sectors):
        lo_lt, hi_lt = lt["lt_min"], lt["lt_max"]
        wrap = lt.get("wrap", False)
        if wrap:
            df_lt = df[(df["lst_h"] >= lo_lt) | (df["lst_h"] < hi_lt)].copy()
        elif lo_lt == 0 and hi_lt == 24:
            df_lt = df.copy()
        else:
            df_lt = df[(df["lst_h"] >= lo_lt) & (df["lst_h"] < hi_lt)].copy()

        ax_r = axes[ci]
        ax_t = ax_r.twinx()
        ax_a = ax_r.twinx()
        ax_a.spines["right"].set_position(("axes", 1.14))
        ax_a.spines["right"].set_visible(True)

        # background shading
        ax_r.axvspan(ref_periods[0][0], ref_periods[0][1], color="lightblue", alpha=0.25, lw=0)
        ax_r.axvspan(ref_periods[1][0], ref_periods[1][1], color="lightblue", alpha=0.25, lw=0)
        for s_start, s_end in zip(ssw_starts, ssw_ends):
            ax_r.axvspan(s_start, s_end, color="lightyellow", alpha=0.40, lw=0)
        for peak in ssw_peaks:
            ax_r.axvline(peak, color="red", lw=1.8, ls="--", zorder=6)

        # rho_ratio
        for (band_label, lat_lo, lat_hi, color) in LAT_BANDS:
            mask = (df_lt["lat"].abs() >= lat_lo) & (df_lt["lat"].abs() < lat_hi)
            daily = df_lt[mask].groupby("date")["density_ratio_msis"].median()
            ax_r.plot(daily.index, daily.values, color=color, lw=2.0, marker="o",
                      markersize=4, label=band_label, zorder=5)
        ax_r.set_ylabel("ρ_ratio (obs / MSIS)", fontsize=11, fontweight="bold")
        ax_r.set_xlabel("Date", fontsize=10)
        ax_r.grid(alpha=0.22)

        # temperature
        t_sub = temp[(temp.index >= x_start) & (temp.index <= x_end)]
        ax_t.plot(t_sub.index, t_sub.values, color="crimson", lw=2.4, marker="s",
                  markersize=3.5, ls="-", zorder=4,
                  label=f"T(10 hPa) {temp_region_label}")
        ax_t.set_ylabel(f"T (10 hPa) [K] {temp_region_label}", fontsize=10,
                        fontweight="bold", color="crimson")
        ax_t.tick_params(axis="y", labelcolor="crimson")
        ax_t.spines["right"].set_edgecolor("crimson")

        # Ap index
        kp_sub = kp[(kp.index >= x_start) & (kp.index <= x_end)]
        ax_a.bar(kp_sub.index, kp_sub.values, width=pd.Timedelta(days=1),
                 color="slategray", alpha=0.45, align="center", zorder=2,
                 label="Ap (daily mean)")
        ax_a.axhline(AP_KP3, color="darkgray", lw=1.2, ls=":", zorder=3,
                     label=f"Kp=3 (Ap={AP_KP3:.0f})")
        ax_a.set_ylabel("Ap index", fontsize=10, fontweight="bold", color="slategray")
        ax_a.tick_params(axis="y", labelcolor="slategray")
        ax_a.spines["right"].set_edgecolor("slategray")

        # x‑axis formatting
        ax_r.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_r.set_xlim(x_start, x_end)
        plt.setp(ax_r.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
        ax_r.set_title(lt["label"], fontsize=12, fontweight="bold", pad=8)

        # combined legend
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        lines_t, labels_t = ax_t.get_legend_handles_labels()
        lines_a, labels_a = ax_a.get_legend_handles_labels()
        ax_r.legend(lines_r + lines_t + lines_a,
                    labels_r + labels_t + labels_a,
                    fontsize=8, loc="upper left", framealpha=0.85, ncol=2)

    # figure‑level legend
    legend_elems = [
        mpatches.Patch(facecolor="lightblue", alpha=0.5, label="Non‑SSW ref period"),
        mpatches.Patch(facecolor="lightyellow", alpha=0.7, label="SSW period"),
        plt.Line2D([0], [0], color="red", lw=1.5, ls="--", label="SSW peak"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=10,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{sat_label}  —  {event_label}", fontsize=14, fontweight="bold", y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)

# ─── Event definition for SH 2024 ───────────────────────────────────────────────

def run_2024_SH():
    print("\n=== 2024 SH SSW ===")
    kp = load_kp(Path("data/SSW2024/Kpindex/SW-20231201_20240331.csv"))
    era5_dir = Path("data/SSW2024/ERA5")
    if era5_dir.exists() and any(era5_dir.glob("*.nc")):
        temp = load_era5_temp(era5_dir, lat_min=-90, lat_max=-60)
    else:
        idx = pd.date_range("2024-06-20", "2024-08-20", freq="D", tz="UTC")
        temp = pd.Series(np.nan, index=idx, name="T10hPa_placeholder")
        print("  [WARN] ERA5 data not found — temperature panel will be empty")

    x_start = pd.Timestamp("2024-06-20", tz="UTC")
    x_end   = pd.Timestamp("2024-08-20", tz="UTC")
    # Two SSW windows around the July and August peaks
    ssw_starts = [pd.Timestamp("2024-07-05", tz="UTC"), pd.Timestamp("2024-08-03", tz="UTC")]
    ssw_ends   = [pd.Timestamp("2024-07-09", tz="UTC"), pd.Timestamp("2024-08-07", tz="UTC")]
    ssw_peaks  = [pd.Timestamp("2024-07-07", tz="UTC"), pd.Timestamp("2024-08-05", tz="UTC")]
    ref1 = (pd.Timestamp("2024-06-20", tz="UTC"), pd.Timestamp("2024-07-04", tz="UTC"))
    ref2 = (pd.Timestamp("2024-08-10", tz="UTC"), pd.Timestamp("2024-08-20", tz="UTC"))
    ref_periods = [ref1, ref2]

    satellites = [
        dict(label="SWARM-A",
             parquet="normalizeddata/2024/swarm_dnsapod_2024_normalized_with_LT.parquet",
             lt_sectors=[
                 dict(label="Morning (04–11 LT)", lt_min=4, lt_max=11, color="#1a6faf"),
                 dict(label="Evening (16–23 LT)", lt_min=16, lt_max=23, color="#e07b39"),
             ]),
        dict(label="SWARM-B",
             parquet="normalizeddata/2024/swarm_dnsbpod_2024_normalized_with_LT.parquet",
             lt_sectors=[
                 dict(label="Nightside (22–05 LT)", lt_min=22, lt_max=5, color="#6a0dad", wrap=True),
                 dict(label="Dayside (11–17 LT)",   lt_min=11, lt_max=17, color="#c0392b"),
             ]),
        dict(label="SWARM-C",
             parquet="normalizeddata/2024/swarm_dnscpod_2024_normalized_with_LT.parquet",
             lt_sectors=[
                 dict(label="Morning (04–11 LT)", lt_min=4, lt_max=11, color="#1a6faf"),
                 dict(label="Evening (16–23 LT)", lt_min=16, lt_max=23, color="#e07b39"),
             ]),
    ]

    for sat in satellites:
        print(f"  {sat['label']} ...", end=" ")
        df = load_ratio(Path(sat["parquet"]), "2024-06-15", "2024-08-25")
        plot_event_satellite(
            event_label="2024 SH SSW",
            sat_label=sat["label"],
            df=df,
            lt_sectors=sat["lt_sectors"],
            temp=temp,
            kp=kp,
            x_start=x_start,
            x_end=x_end,
            ssw_starts=ssw_starts,
            ssw_ends=ssw_ends,
            ssw_peaks=ssw_peaks,
            ref_periods=ref_periods,
            temp_region_label="ERA5 60–90°S",
            out_png=OUT_BASE / f"ratio_temp_Kp_2024_SH_{sat['label'].replace('-','')}.png",
        )

if __name__ == "__main__":
    run_2024_SH()
    print("\nAll SH 2024 plots done.")
