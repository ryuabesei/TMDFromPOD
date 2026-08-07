"""
plot_1D_ratio_temp_Kp_3events.py

Purpose:
    For each SSW event (2018 NH / 2019 SH / 2021 NH) and each satellite,
    plot a 3-panel time series:
        Panel 1 (top)   : rho_ratio (density_ratio_msis, NO Kp filter) by LT sector
        Panel 2 (middle): Stratospheric temperature T(10 hPa) [K]
                          2018 → COSMIC (60–90°N avg)
                          2019 → ERA5   (60–90°S avg)
                          2021 → ERA5   (60–90°N avg)
        Panel 3 (bottom): Daily-mean Ap index (proxy for Kp)
                          Kp=3 threshold (Ap=15) marked as horizontal dashed line

Layout per figure: 3 panels stacked, shared x-axis (date)
Latitude bands (High/Mid/Low) are colour-coded within Panel 1.
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
# Column name for rho_ratio — fall back across different pipeline versions
VALUE_COL_CANDIDATES = ["density_ratio_msis", "rho_ratio", "density_norm"]
AP_KP3    = 15.0          # Kp=3 ↔ Ap=15

LAT_BANDS = [
    ("High (40–60°)", 40.0, 60.0, "#1f77b4"),
    ("Mid  (20–40°)", 20.0, 40.0, "#2ca02c"),
    ("Low  ( 0–20°)",  0.0, 20.0, "#ff7f0e"),
]

OUT_BASE = Path("Figure/summary")


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_kp(kp_csv: Path) -> pd.Series:
    df = pd.read_csv(kp_csv, parse_dates=["DATE"])
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
    df = df.set_index("DATE").sort_index()
    return df["AP_AVG"].astype(float)


def load_cosmic_temp(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("datetime")["T10_K"].sort_index()


def load_era5_temp(era5_dir: Path, lat_min: float, lat_max: float) -> pd.Series:
    """Load all *.nc in era5_dir, area-average T10hPa over [lat_min, lat_max]."""
    files = sorted(era5_dir.glob("*.nc"))
    series_list = []
    for fp in files:
        with xr.open_dataset(fp) as ds:
            # slice latitude (ERA5 lat is decreasing: 90 → -90)
            lo, hi = min(lat_min, lat_max), max(lat_min, lat_max)
            ds_sub  = ds.sel(latitude=slice(hi, lo))   # ERA5 order: high→low
            weights = np.cos(np.deg2rad(ds_sub["latitude"]))
            w_temp  = ds_sub["t"].weighted(weights)
            t_avg   = w_temp.mean(dim=["latitude", "longitude"]).squeeze()
            # daily mean
            times = pd.to_datetime(t_avg["valid_time"].values)
            vals  = t_avg.values.ravel()
            s = pd.Series(vals, index=times)
            s.index = s.index.tz_localize("UTC")
            series_list.append(s)
    combined = pd.concat(series_list).sort_index()
    # daily mean
    combined = combined.resample("D").mean()
    return combined


def load_ratio(parquet: Path, t_start: str, t_end: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col and lat_col != "lat":
        df = df.rename(columns={lat_col: "lat"})
    # Resolve VALUE_COL: use whichever ratio column is available
    value_col = next((c for c in VALUE_COL_CANDIDATES if c in df.columns), None)
    if value_col is None:
        raise KeyError(f"No ratio column found in {parquet}. Available: {list(df.columns)}")
    if value_col != "density_ratio_msis":
        df["density_ratio_msis"] = df[value_col]   # normalise to expected name
    global VALUE_COL
    VALUE_COL = "density_ratio_msis"
    df = df.dropna(subset=["datetime", "lat", VALUE_COL, "lst_h"])
    df = df[(df["datetime"] >= pd.Timestamp(t_start, tz="UTC")) &
            (df["datetime"] <= pd.Timestamp(t_end,   tz="UTC"))]
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
    ssw_start: pd.Timestamp,
    ssw_end: pd.Timestamp,
    ssw_peak: pd.Timestamp,
    ref1: tuple,
    ref2: tuple,
    temp_region_label: str,
    out_png: Path,
) -> None:

    n_lt = len(lt_sectors)
    fig, axes = plt.subplots(
        1, n_lt,
        figsize=(8 * n_lt, 6),
        gridspec_kw={"wspace": 0.30}
    )
    if n_lt == 1:
        axes = [axes]

    for ci, lt in enumerate(lt_sectors):
        lo_lt, hi_lt = lt["lt_min"], lt["lt_max"]
        wrap = lt.get("wrap", False)

        # ── Filter by LT ──────────────────────────────────────────────────
        if wrap:
            df_lt = df[(df["lst_h"] >= lo_lt) | (df["lst_h"] < hi_lt)].copy()
        elif lo_lt == 0 and hi_lt == 24:
            df_lt = df.copy()
        else:
            df_lt = df[(df["lst_h"] >= lo_lt) & (df["lst_h"] < hi_lt)].copy()

        ax_r = axes[ci]          # left  axis  : rho_ratio
        ax_t = ax_r.twinx()      # right axis 1: T(10 hPa)
        ax_a = ax_r.twinx()      # right axis 2: Ap bars

        # Offset the Ap axis to the far right so it doesn't overlap T axis
        ax_a.spines["right"].set_position(("axes", 1.14))
        ax_a.spines["right"].set_visible(True)

        # ── Background shading ────────────────────────────────────────────
        ax_r.patch.set_visible(True)
        ax_r.axvspan(ref1[0], ref1[1], color="lightblue",   alpha=0.25, lw=0)
        ax_r.axvspan(ref2[0], ref2[1], color="lightblue",   alpha=0.25, lw=0)
        ax_r.axvspan(ssw_start, ssw_end, color="lightyellow", alpha=0.40, lw=0)
        ax_r.axvline(ssw_peak, color="red", lw=1.8, ls="--", zorder=6, label="SSW peak")

        # ── rho_ratio (left axis) ─────────────────────────────────────────
        for (band_label, lat_lo, lat_hi, color) in LAT_BANDS:
            mask  = (df_lt["lat"].abs() >= lat_lo) & (df_lt["lat"].abs() < lat_hi)
            daily = df_lt[mask].groupby("date")[VALUE_COL].median()
            ax_r.plot(daily.index, daily.values,
                      color=color, lw=2.0, marker="o", markersize=4,
                      label=band_label, zorder=5)

        ax_r.set_ylabel("ρ_ratio  (obs / MSIS)", fontsize=11, fontweight="bold")
        ax_r.set_xlabel("Date", fontsize=10)
        ax_r.grid(alpha=0.22)

        # ── T(10 hPa) (right axis 1) ──────────────────────────────────────
        t_sub = temp[(temp.index >= x_start) & (temp.index <= x_end)]
        ax_t.plot(t_sub.index, t_sub.values,
                  color="crimson", lw=2.4, marker="s", markersize=3.5,
                  ls="-", zorder=4, label=f"T(10 hPa) {temp_region_label}")
        ax_t.set_ylabel(f"T (10 hPa) [K]  {temp_region_label}",
                        fontsize=10, fontweight="bold", color="crimson")
        ax_t.tick_params(axis="y", labelcolor="crimson")
        ax_t.spines["right"].set_edgecolor("crimson")

        # ── Ap index (right axis 2, far right) ────────────────────────────
        kp_sub = kp[(kp.index >= x_start) & (kp.index <= x_end)]
        ax_a.bar(kp_sub.index, kp_sub.values,
                 width=pd.Timedelta(days=1), color="slategray", alpha=0.45,
                 align="center", zorder=2, label="Ap (daily mean)")
        ax_a.axhline(AP_KP3, color="darkgray", lw=1.2, ls=":",
                     zorder=3, label=f"Kp=3 (Ap={AP_KP3:.0f})")
        ax_a.set_ylabel("Ap index", fontsize=10, fontweight="bold", color="slategray")
        ax_a.tick_params(axis="y", labelcolor="slategray")
        ax_a.spines["right"].set_edgecolor("slategray")

        # ── x-axis format ─────────────────────────────────────────────────
        ax_r.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_r.set_xlim(x_start, x_end)
        plt.setp(ax_r.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

        ax_r.set_title(lt["label"], fontsize=12, fontweight="bold", pad=8)

        # ── Combined legend ───────────────────────────────────────────────
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        lines_t, labels_t = ax_t.get_legend_handles_labels()
        lines_a, labels_a = ax_a.get_legend_handles_labels()
        ax_r.legend(lines_r + lines_t + lines_a,
                    labels_r + labels_t + labels_a,
                    fontsize=8, loc="upper left", framealpha=0.85,
                    ncol=2)

    # ── Figure title ──────────────────────────────────────────────────────────
    legend_elems = [
        mpatches.Patch(facecolor="lightblue",   alpha=0.5, label="Non-SSW ref period"),
        mpatches.Patch(facecolor="lightyellow", alpha=0.7, label="SSW period"),
        plt.Line2D([0], [0], color="red", lw=1.5, ls="--", label="SSW peak"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.85, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"{sat_label}  —  {event_label}",
        fontsize=14, fontweight="bold", y=1.02
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ─── Event definitions ────────────────────────────────────────────────────────

def run_2018():
    print("\n=== 2018 NH SSW ===")
    kp = load_kp(Path("data/SSW2018/Kpindex/SW-20180120_20180320.csv"))
    temp = load_cosmic_temp(Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv"))

    x_start  = pd.Timestamp("2018-01-29", tz="UTC")
    x_end    = pd.Timestamp("2018-03-07", tz="UTC")
    ssw_s    = pd.Timestamp("2018-02-06", tz="UTC")
    ssw_e    = pd.Timestamp("2018-02-24", tz="UTC")
    ssw_peak = pd.Timestamp("2018-02-12", tz="UTC")
    ref1     = (pd.Timestamp("2018-01-29", tz="UTC"), pd.Timestamp("2018-02-04", tz="UTC"))
    ref2     = (pd.Timestamp("2018-02-25", tz="UTC"), pd.Timestamp("2018-03-07", tz="UTC"))

    satellites = [
        dict(label="SWARM-A",
             parquet="normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet",
             lt_sectors=[
                 dict(label="Morning (04–11 LT)", lt_min=4,  lt_max=11, color="#1a6faf"),
                 dict(label="Evening (16–23 LT)", lt_min=16, lt_max=23, color="#e07b39"),
             ]),
        dict(label="SWARM-B",
             parquet="normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet",
             lt_sectors=[
                 dict(label="Nightside (22–05 LT)", lt_min=22, lt_max=5,  color="#6a0dad", wrap=True),
                 dict(label="Dayside (11–17 LT)",   lt_min=11, lt_max=17, color="#c0392b"),
             ]),
        dict(label="SWARM-C",
             parquet="normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet",
             lt_sectors=[
                 dict(label="Morning (04–11 LT)", lt_min=4,  lt_max=11, color="#1a6faf"),
                 dict(label="Evening (16–23 LT)", lt_min=16, lt_max=23, color="#e07b39"),
             ]),
    ]

    for sat in satellites:
        print(f"  {sat['label']} ...", end=" ")
        df = load_ratio(Path(sat["parquet"]), "2018-01-20", "2018-03-21")
        plot_event_satellite(
            event_label="2018 NH SSW",
            sat_label=sat["label"],
            df=df,
            lt_sectors=sat["lt_sectors"],
            temp=temp,
            kp=kp,
            x_start=x_start, x_end=x_end,
            ssw_start=ssw_s, ssw_end=ssw_e, ssw_peak=ssw_peak,
            ref1=ref1, ref2=ref2,
            temp_region_label="COSMIC 60–90°N",
            out_png=OUT_BASE / f"ratio_temp_Kp_2018_{sat['label'].replace('-','')}.png",
        )


def run_2019():
    print("\n=== 2019 SH SSW ===")
    kp = load_kp(Path("data/SSW2019/Kpindex/SW-All_2019-08-14_to_2019-09-24.csv"))
    temp = load_era5_temp(Path("data/SSW2019/ERA5"), lat_min=-90, lat_max=-60)

    x_start  = pd.Timestamp("2019-08-20", tz="UTC")
    x_end    = pd.Timestamp("2019-09-23", tz="UTC")
    ssw_s    = pd.Timestamp("2019-08-27", tz="UTC")
    ssw_e    = pd.Timestamp("2019-09-19", tz="UTC")
    ssw_peak = pd.Timestamp("2019-09-19", tz="UTC")
    ref1     = (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC"))
    ref2     = (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC"))

    satellites = [
        dict(label="SWARM-A",
             parquet="normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet",
             lt_sectors=[
                 dict(label="Dawn (02.5–08.5 LT)", lt_min=2.5, lt_max=8.5,  color="#1a6faf"),
                 dict(label="Dusk (14.5–20.5 LT)", lt_min=14.5,lt_max=20.5, color="#e07b39"),
             ]),
        dict(label="SWARM-B",
             parquet="normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet",
             lt_sectors=[
                 dict(label="Midnight (00–04 LT)", lt_min=0,  lt_max=4,  color="#6a0dad"),
                 dict(label="Noon (12–16 LT)",     lt_min=12, lt_max=16, color="#c0392b"),
             ]),
        dict(label="SWARM-C",
             parquet="normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet",
             lt_sectors=[
                 dict(label="Dawn (02.5–08.5 LT)", lt_min=2.5, lt_max=8.5,  color="#1a6faf"),
                 dict(label="Dusk (14.5–20.5 LT)", lt_min=14.5,lt_max=20.5, color="#e07b39"),
             ]),
    ]

    for sat in satellites:
        print(f"  {sat['label']} ...", end=" ")
        df = load_ratio(Path(sat["parquet"]), "2019-08-20", "2019-09-23")
        plot_event_satellite(
            event_label="2019 SH SSW",
            sat_label=sat["label"],
            df=df,
            lt_sectors=sat["lt_sectors"],
            temp=temp,
            kp=kp,
            x_start=x_start, x_end=x_end,
            ssw_start=ssw_s, ssw_end=ssw_e, ssw_peak=ssw_peak,
            ref1=ref1, ref2=ref2,
            temp_region_label="ERA5 60–90°S",
            out_png=OUT_BASE / f"ratio_temp_Kp_2019_{sat['label'].replace('-','')}.png",
        )


def run_2021():
    print("\n=== 2021 NH SSW ===")
    kp = load_kp(Path("data/SSW2021/Kpindex/SW-20201220_20210228.csv"))
    temp = load_era5_temp(Path("data/SSW2021/ERA5"), lat_min=60, lat_max=90)

    x_start  = pd.Timestamp("2020-12-25", tz="UTC")
    x_end    = pd.Timestamp("2021-02-05", tz="UTC")
    ssw_s    = pd.Timestamp("2020-12-30", tz="UTC")
    ssw_e    = pd.Timestamp("2021-01-31", tz="UTC")
    ssw_peak = pd.Timestamp("2021-01-04", tz="UTC")
    ref1     = (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29", tz="UTC"))
    ref2     = (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05", tz="UTC"))

    satellites = [
        dict(label="SWARM-C",
             parquet="normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet",
             lt_sectors=[
                 dict(label="Dawn (02.5–08.5 LT)", lt_min=2.5, lt_max=8.5,  color="#1a6faf"),
                 dict(label="Dusk (14.5–20.5 LT)", lt_min=14.5,lt_max=20.5, color="#e07b39"),
             ]),
        dict(label="GRACE-FO",
             parquet="normalizeddata/2021/grace_fo_dns_2021_normalized_with_LT_removed.parquet",
             lt_sectors=[
                 dict(label="All LT", lt_min=0, lt_max=24, color="#2ca02c"),
             ]),
    ]

    for sat in satellites:
        print(f"  {sat['label']} ...", end=" ")
        df = load_ratio(Path(sat["parquet"]), "2020-12-20", "2021-02-10")
        plot_event_satellite(
            event_label="2021 NH SSW",
            sat_label=sat["label"],
            df=df,
            lt_sectors=sat["lt_sectors"],
            temp=temp,
            kp=kp,
            x_start=x_start, x_end=x_end,
            ssw_start=ssw_s, ssw_end=ssw_e, ssw_peak=ssw_peak,
            ref1=ref1, ref2=ref2,
            temp_region_label="ERA5 60–90°N",
            out_png=OUT_BASE / f"ratio_temp_Kp_2021_{sat['label'].replace('-','').replace('/', '')}.png",
        )


def run_2024():
    print("\n=== 2024 NH SSW ===")
    kp   = load_kp(Path("data/SSW2024/Kpindex/SW-20231201_20240331.csv"))

    # ERA5 temperature: use if available, else flat NaN series
    era5_dir = Path("data/SSW2024/ERA5")
    if era5_dir.exists() and any(era5_dir.glob("*.nc")):
        temp = load_era5_temp(era5_dir, lat_min=60, lat_max=90)
    else:
        # Placeholder until ERA5 is downloaded
        idx  = pd.date_range("2023-12-01", "2024-03-31", freq="D", tz="UTC")
        temp = pd.Series(np.nan, index=idx, name="T10hPa_placeholder")
        print("  [WARN] ERA5 data not found — temperature panel will be empty")

    x_start  = pd.Timestamp("2023-12-22", tz="UTC")
    x_end    = pd.Timestamp("2024-02-28", tz="UTC")
    ssw_s    = pd.Timestamp("2024-01-05", tz="UTC")
    ssw_e    = pd.Timestamp("2024-01-30", tz="UTC")
    ssw_peak = pd.Timestamp("2024-01-16", tz="UTC")   # major SSW peak
    ref1     = (pd.Timestamp("2023-12-22", tz="UTC"), pd.Timestamp("2024-01-04", tz="UTC"))
    ref2     = (pd.Timestamp("2024-01-31", tz="UTC"), pd.Timestamp("2024-02-28", tz="UTC"))

    satellites = [
        dict(label="SWARM-A",
             parquet="normalizeddata/2024/swarm_dnsapod_2024_normalized_with_LT.parquet",
             lt_sectors=[
                 dict(label="Morning (04–11 LT)", lt_min=4,  lt_max=11, color="#1a6faf"),
                 dict(label="Evening (16–23 LT)", lt_min=16, lt_max=23, color="#e07b39"),
             ]),
        dict(label="SWARM-B",
             parquet="normalizeddata/2024/swarm_dnsbpod_2024_normalized_with_LT.parquet",
             lt_sectors=[
                 dict(label="Nightside (22–05 LT)", lt_min=22, lt_max=5,  color="#6a0dad", wrap=True),
                 dict(label="Dayside (11–17 LT)",   lt_min=11, lt_max=17, color="#c0392b"),
             ]),
        dict(label="SWARM-C",
             parquet="normalizeddata/2024/swarm_dnscpod_2024_normalized_with_LT.parquet",
             lt_sectors=[
                 dict(label="Morning (04–11 LT)", lt_min=4,  lt_max=11, color="#1a6faf"),
                 dict(label="Evening (16–23 LT)", lt_min=16, lt_max=23, color="#e07b39"),
             ]),
    ]

    for sat in satellites:
        print(f"  {sat['label']} ...", end=" ")
        df = load_ratio(Path(sat["parquet"]), "2023-12-15", "2024-03-01")
        plot_event_satellite(
            event_label="2024 NH SSW",
            sat_label=sat["label"],
            df=df,
            lt_sectors=sat["lt_sectors"],
            temp=temp,
            kp=kp,
            x_start=x_start, x_end=x_end,
            ssw_start=ssw_s, ssw_end=ssw_e, ssw_peak=ssw_peak,
            ref1=ref1, ref2=ref2,
            temp_region_label="ERA5 60–90°N",
            out_png=OUT_BASE / f"ratio_temp_Kp_2024_{sat['label'].replace('-','')}.png",
        )


def run_2026():
    print("\n=== 2026 NH SSW ===")
    kp   = load_kp(Path("data/SSW2026/Kpindex/SW-20251201_20260430.csv"))

    # ERA5 temperature: use if available, else flat NaN series
    era5_dir = Path("data/SSW2026/ERA5")
    if era5_dir.exists() and any(era5_dir.glob("*.nc")):
        temp = load_era5_temp(era5_dir, lat_min=60, lat_max=90)
    else:
        idx  = pd.date_range("2025-12-01", "2026-04-30", freq="D", tz="UTC")
        temp = pd.Series(np.nan, index=idx, name="T10hPa_placeholder")
        print("  [WARN] ERA5 data not found — temperature panel will be empty")

    x_start  = pd.Timestamp("2025-12-20", tz="UTC")
    x_end    = pd.Timestamp("2026-03-20", tz="UTC")
    ssw_s    = pd.Timestamp("2026-01-25", tz="UTC")
    ssw_e    = pd.Timestamp("2026-02-25", tz="UTC")
    ssw_peak = pd.Timestamp("2026-02-10", tz="UTC")   # provisional peak
    ref1     = (pd.Timestamp("2025-12-20", tz="UTC"), pd.Timestamp("2026-01-10", tz="UTC"))
    ref2     = (pd.Timestamp("2026-02-26", tz="UTC"), pd.Timestamp("2026-03-15", tz="UTC"))

    satellites = [
        dict(label="SWARM-A",
             parquet="normalizeddata/2026/swarm_dnsapod_2026_normalized_with_LT.parquet",
             lt_sectors=[
                 dict(label="Morning (04–11 LT)", lt_min=4,  lt_max=11, color="#1a6faf"),
                 dict(label="Evening (16–23 LT)", lt_min=16, lt_max=23, color="#e07b39"),
             ]),
        dict(label="SWARM-B",
             parquet="normalizeddata/2026/swarm_dnsbpod_2026_normalized_with_LT.parquet",
             lt_sectors=[
                 dict(label="Nightside (22–05 LT)", lt_min=22, lt_max=5,  color="#6a0dad", wrap=True),
                 dict(label="Dayside (11–17 LT)",   lt_min=11, lt_max=17, color="#c0392b"),
             ]),
        dict(label="SWARM-C",
             parquet="normalizeddata/2026/swarm_dnscpod_2026_normalized_with_LT.parquet",
             lt_sectors=[
                 dict(label="Morning (04–11 LT)", lt_min=4,  lt_max=11, color="#1a6faf"),
                 dict(label="Evening (16–23 LT)", lt_min=16, lt_max=23, color="#e07b39"),
             ]),
    ]

    for sat in satellites:
        print(f"  {sat['label']} ...", end=" ")
        df = load_ratio(Path(sat["parquet"]), "2025-12-01", "2026-04-01")
        plot_event_satellite(
            event_label="2026 NH SSW",
            sat_label=sat["label"],
            df=df,
            lt_sectors=sat["lt_sectors"],
            temp=temp,
            kp=kp,
            x_start=x_start, x_end=x_end,
            ssw_start=ssw_s, ssw_end=ssw_e, ssw_peak=ssw_peak,
            ref1=ref1, ref2=ref2,
            temp_region_label="ERA5 60–90°N",
            out_png=OUT_BASE / f"ratio_temp_Kp_2026_{sat['label'].replace('-','')}.png",
        )


if __name__ == "__main__":
    run_2018()
    run_2019()
    run_2021()
    run_2024()
    run_2026()
    print("\nAll done.")
