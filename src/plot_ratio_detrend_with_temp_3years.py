"""
plot_ratio_detrend_with_temp_3years.py

Purpose:
    Plot the Ap-detrended density ratio residual
        Delta y_Ap = y_i - y_pred(Ap_i) = y_i - (a * Ap_i + b)
    together with stratospheric temperature T(10 hPa) [K] for 3 major SSW events:
      - 2018 NH SSW (SWARM-A, left column)
      - 2019 SH SSW (SWARM-A, center column)
      - 2021 NH SSW (SWARM-C, right column)

    Original ratio is excluded as requested.

Layout:
    3 columns (events) x 2 rows:
      Top Row   : Delta y_Ap (detrended density ratio residual, centered at 0)
      Bottom Row: Stratospheric Temperature T(10 hPa) [K]

Output:
    Figure/Ap_removal/ratio_detrend_with_temp_3years.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# ─── Paths ───────────────────────────────────────────────────────────────────
P2018 = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")

COSMIC_2018 = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
ERA5_2019_DIR = Path("data/SSW2019/ERA5")
ERA5_2021_DIR = Path("data/SSW2021/ERA5")

OUT_DIR = Path("Figure/Ap_removal")
OUT_PNG = OUT_DIR / "ratio_detrend_with_temp_3years.png"

# ─── Event Settings ──────────────────────────────────────────────────────────
EVENTS = [
    dict(
        label="2018 NH SSW", sat="SWARM-A",
        parquet=P2018, mode="doy",
        doy_start=30, doy_end=65,
        ref_doy=[(30, 40), (61, 65)],
        ssw_peak_doy=43, year=2018,
        temp_type="cosmic",
        temp_file=COSMIC_2018,
        temp_label="COSMIC T(10 hPa) 60–90°N",
    ),
    dict(
        label="2019 SH SSW", sat="SWARM-A",
        parquet=P2019, mode="date",
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"), year=2019,
        temp_type="era5",
        temp_dir=ERA5_2019_DIR, lat_range=(-90.0, -60.0),
        temp_label="ERA5 T(10 hPa) 60–90°S",
    ),
    dict(
        label="2021 NH SSW", sat="SWARM-C",
        parquet=P2021, mode="date",
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end=pd.Timestamp("2021-02-05", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29", tz="UTC")),
            (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2021-01-04", tz="UTC"), year=2021,
        temp_type="era5",
        temp_dir=ERA5_2021_DIR, lat_range=(60.0, 90.0),
        temp_label="ERA5 T(10 hPa) 60–90°N",
    ),
]


# ─── Loaders ─────────────────────────────────────────────────────────────────
def load_event_df(ev: dict) -> pd.DataFrame:
    df = pd.read_parquet(ev["parquet"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for cname in ["lat", "latitude", "geod_lat"]:
        if cname in df.columns and cname != "lat":
            df = df.rename(columns={cname: "lat"})
            break
    df = df.dropna(subset=["datetime", "density_ratio_msis", "AP_AVG"])

    if ev["mode"] == "doy":
        df["key"] = df["datetime"].dt.dayofyear
        df = df[(df["key"] >= ev["doy_start"]) & (df["key"] <= ev["doy_end"])]
    else:
        df["key"] = df["datetime"].dt.normalize()
        df = df[(df["datetime"] >= ev["date_start"]) &
                (df["datetime"] <= ev["date_end"] + pd.Timedelta(hours=23, minutes=59))]
    return df


def load_temp(ev: dict) -> pd.Series:
    if ev["temp_type"] == "cosmic":
        df = pd.read_csv(ev["temp_file"], parse_dates=["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        if ev["mode"] == "doy":
            df["key"] = df["datetime"].dt.dayofyear
            df = df[(df["key"] >= ev["doy_start"]) & (df["key"] <= ev["doy_end"])]
            return df.set_index("key")["T10_K"].sort_index()
        else:
            return df.set_index("datetime")["T10_K"].sort_index()
    else: # era5
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
        if ev["mode"] == "date":
            combined.index = combined.index.normalize()
            combined = combined[(combined.index >= ev["date_start"]) &
                                (combined.index <= ev["date_end"])]
        return combined


# ─── Main Processing & Plotting ──────────────────────────────────────────────
def main() -> None:
    all_data = []

    for ev in EVENTS:
        print(f"Processing {ev['label']} ({ev['sat']})...")
        df = load_event_df(ev)

        # Calculate daily median ratio and mean Ap
        daily = df.groupby("key").agg(
            ratio=("density_ratio_msis", "median"),
            ap=("AP_AVG", "mean"),
        ).reset_index()

        # Fit linear regression: y = a * Ap + b
        x = daily["ap"].values
        y = daily["ratio"].values
        p = np.polyfit(x, y, 1)
        slope, intercept = p[0], p[1]
        r = float(np.corrcoef(x, y)[0, 1])

        # Compute pure Ap-detrended residual: Delta y_Ap = y_i - (a * Ap_i + b)
        delta_ap = daily["ratio"] - (slope * daily["ap"] + intercept)
        s_delta = pd.Series(delta_ap.values, index=daily["key"])

        # Load Temperature
        s_temp = load_temp(ev)

        all_data.append({
            "ev": ev,
            "daily": daily,
            "s_delta": s_delta,
            "s_temp": s_temp,
            "slope": slope,
            "intercept": intercept,
            "r": r,
        })

    # Plotting: 2 rows x 3 columns
    fig, axes = plt.subplots(
        2, 3, figsize=(21, 7.5),
        gridspec_kw={"height_ratios": [2.5, 2.0], "hspace": 0.12, "wspace": 0.20},
    )
    fig.suptitle(
        r"Ap-Detrended Density Ratio Residual ($\Delta y_{Ap} = y_i - y_{pred}(Ap_i)$) & Stratospheric Temperature T(10 hPa)" "\n"
        r"Orange dashed: $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$  |  Crimson solid: Stratospheric Temperature T(10 hPa) [K]",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for col, data in enumerate(all_data):
        ev = data["ev"]
        ax_r = axes[0, col]
        ax_t = axes[1, col]

        s_delta = data["s_delta"]
        s_temp  = data["s_temp"]
        slope   = data["slope"]
        intercept = data["intercept"]
        r       = data["r"]

        # ── Top Row: Delta y_Ap ──────────────────────────────────────────────
        ax_r.plot(s_delta.index, s_delta.values,
                  color="#ff7f0e", lw=2.2, ls="--", marker="^", ms=4.5,
                  zorder=6, label=r"$\Delta y_{Ap} = y_i - y_{pred}(Ap_i)$")
        ax_r.axhline(0, color="gray", lw=0.8, ls=":", zorder=3)

        # SSW Peak
        pk = ev.get("ssw_peak_doy") or ev.get("ssw_peak")
        if pk is not None:
            ax_r.axvline(pk, color="red", lw=1.8, ls="--", alpha=0.8, zorder=7, label="SSW Peak")
            ax_t.axvline(pk, color="red", lw=1.8, ls="--", alpha=0.8, zorder=7)
            ax_r.text(pk, 0.94, " SSW Peak", transform=ax_r.get_xaxis_transform(),
                      fontsize=8.5, color="red", fontweight="bold", alpha=0.9, va="top")

        # Ref period shading
        if ev["mode"] == "doy":
            for lo, hi in ev["ref_doy"]:
                ax_r.axvspan(lo, hi, color="lightblue", alpha=0.25, lw=0)
                ax_t.axvspan(lo, hi, color="lightblue", alpha=0.25, lw=0)
        else:
            for s, e in ev["ref_dates"]:
                ax_r.axvspan(s, e, color="lightblue", alpha=0.25, lw=0)
                ax_t.axvspan(s, e, color="lightblue", alpha=0.25, lw=0)

        # Reg params annotation
        ax_r.text(
            0.02, 0.97,
            f"a={slope:.5f}\nb={intercept:.4f}\nr(y,Ap)={r:.3f}",
            transform=ax_r.transAxes,
            fontsize=8.0, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

        ax_r.set_title(f"{ev['label']}  ({ev['sat']})", fontsize=11, fontweight="bold")
        ax_r.set_ylabel(r"$\Delta y_{Ap}$ (Detrended Residual)", fontsize=10)
        ax_r.grid(True, ls=":", alpha=0.45)
        ax_r.tick_params(labelbottom=False)
        ax_r.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        # ── Bottom Row: Stratospheric Temperature ────────────────────────────
        if not s_temp.empty:
            ax_t.plot(s_temp.index, s_temp.values,
                      color="crimson", lw=2.2, marker="s", ms=3.5,
                      zorder=5, label=ev["temp_label"])
            ax_t.set_ylabel("T (10 hPa) [K]", fontsize=10, color="crimson")
            ax_t.tick_params(axis="y", labelcolor="crimson")
            ax_t.grid(True, ls=":", alpha=0.45)
            ax_t.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

        if ev["mode"] == "date":
            ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax_t.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            plt.setp(ax_t.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax_t.set_xlabel(f"{'DOY' if ev['mode']=='doy' else 'Date'} ({ev['year']})", fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure: {OUT_PNG}")


if __name__ == "__main__":
    main()
