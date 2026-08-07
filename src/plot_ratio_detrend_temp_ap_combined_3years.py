"""
plot_ratio_detrend_temp_ap_combined_3years.py

Purpose:
    Plot Delta y_Ap = y_i - y_pred(Ap_i), Stratospheric Temperature T(10 hPa),
    and Ap index SIMULTANEOUSLY on the SAME subplot panel using multiple Y-axes.

Layout:
    3 columns for 3 major SSW events:
      - 2018 NH SSW (SWARM-A, left panel)
      - 2019 SH SSW (SWARM-A, center panel)
      - 2021 NH SSW (SWARM-C, right panel)

Axes per panel:
    - Left Y-axis (primary)  : Delta y_Ap (detrended density ratio residual, centered at 0)
    - Right Y-axis 1         : Stratospheric Temperature T(10 hPa) [K] (Crimson line)
    - Right Y-axis 2 (offset): Daily-mean Ap index (Gray bar chart)

Output:
    Figure/Ap_removal/ratio_detrend_temp_ap_combined_3years.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─── Paths ───────────────────────────────────────────────────────────────────
P2018 = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")

COSMIC_2018 = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")
ERA5_2019_DIR = Path("data/SSW2019/ERA5")
ERA5_2021_DIR = Path("data/SSW2021/ERA5")

OUT_DIR = Path("Figure/Ap_removal")
OUT_PNG = OUT_DIR / "ratio_detrend_temp_ap_combined_3years.png"

AP_KP3 = 15.0

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
        temp_label="T(10 hPa) 60–90°N",
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
        temp_label="T(10 hPa) 60–90°S",
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
        temp_label="T(10 hPa) 60–90°N",
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


# ─── Main Script ─────────────────────────────────────────────────────────────
def main() -> None:
    all_data = []

    for ev in EVENTS:
        print(f"Processing {ev['label']} ({ev['sat']})...")
        df = load_event_df(ev)

        daily = df.groupby("key").agg(
            ratio=("density_ratio_msis", "median"),
            ap=("AP_AVG", "mean"),
        ).reset_index()

        x = daily["ap"].values
        y = daily["ratio"].values
        p = np.polyfit(x, y, 1)
        slope, intercept = p[0], p[1]
        r = float(np.corrcoef(x, y)[0, 1])

        # Delta y_Ap = y_i - (a * Ap_i + b)
        delta_ap = daily["ratio"] - (slope * daily["ap"] + intercept)
        s_delta = pd.Series(delta_ap.values, index=daily["key"])
        s_ap = pd.Series(daily["ap"].values, index=daily["key"])

        s_temp = load_temp(ev)

        all_data.append({
            "ev": ev,
            "daily": daily,
            "s_delta": s_delta,
            "s_temp": s_temp,
            "s_ap": s_ap,
            "slope": slope,
            "intercept": intercept,
            "r": r,
        })

    # Create figure: 3 rows x 1 column (wide aspect ratio stacked vertically)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    fig.suptitle(
        r"Thermospheric Density Ratio Residual ($\Delta y_{Ap}$), Stratospheric Temp T(10 hPa), & Ap Index" "\n"
        r"Blue line: $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$ (Left Y-axis)  |  Crimson line: T(10 hPa) (Right Y-axis 1)  |  Orange line: Ap index (Right Y-axis 2)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    for row, data in enumerate(all_data):
        ev = data["ev"]
        ax_r = axes[row]
        ax_t = ax_r.twinx()
        ax_a = ax_r.twinx()

        # Offset right spine for Ap index (3rd Y-axis)
        ax_a.spines["right"].set_position(("axes", 1.10))
        ax_a.spines["right"].set_visible(True)

        s_delta = data["s_delta"]
        s_temp  = data["s_temp"]
        s_ap    = data["s_ap"]
        slope   = data["slope"]
        intercept = data["intercept"]
        r       = data["r"]

        # Ref period shading
        if ev["mode"] == "doy":
            for lo, hi in ev["ref_doy"]:
                ax_r.axvspan(lo, hi, color="lightblue", alpha=0.22, lw=0)
        else:
            for s, e in ev["ref_dates"]:
                ax_r.axvspan(s, e, color="lightblue", alpha=0.22, lw=0)

        # ── 1. Ap Line Plot (3rd Y-axis, Orange/Brown line with square markers) ───
        ax_a.plot(s_ap.index, s_ap.values,
                  color="#e07b39", lw=1.8, marker="s", ms=4.5, alpha=0.9,
                  zorder=3, label="Daily mean Ap")
        ax_a.axhline(AP_KP3, color="#e07b39", lw=1.0, ls=":", alpha=0.7, zorder=2)
        ax_a.set_ylabel("Ap Index", fontsize=10, fontweight="bold", color="#e07b39")
        ax_a.tick_params(axis="y", labelcolor="#e07b39", labelsize=8.5)
        ap_max = s_ap.max() if not s_ap.empty else AP_KP3
        ax_a.set_ylim(0, max(ap_max * 1.25, AP_KP3 * 1.3))
        ax_a.spines["right"].set_edgecolor("#e07b39")

        # ── 2. Temperature Line (2nd Y-axis, Crimson line) ───────────────────
        if not s_temp.empty:
            ax_t.plot(s_temp.index, s_temp.values,
                      color="crimson", lw=2.0, marker="^", ms=4.5,
                      zorder=4, label=ev["temp_label"])
            ax_t.set_ylabel("T (10 hPa) [K]", fontsize=10, fontweight="bold", color="crimson")
            ax_t.tick_params(axis="y", labelcolor="crimson", labelsize=8.5)
            ax_t.spines["right"].set_edgecolor("crimson")

        # ── 3. Delta y_Ap Line (Left Y-axis, Blue line) ──────────────────────
        ax_r.plot(s_delta.index, s_delta.values,
                  color="#1f77b4", lw=2.2, marker="o", ms=5.0,
                  zorder=5, label=r"$\Delta y_{Ap} = y_i - y_{pred}(Ap_i)$")
        ax_r.axhline(0, color="black", lw=0.8, ls=":", zorder=3)
        ax_r.set_ylabel(r"$\Delta y_{Ap}$ (Density Ratio Residual)", fontsize=10, fontweight="bold", color="#1f77b4")
        ax_r.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8.5)

        # SSW Peak line
        pk = ev.get("ssw_peak_doy") or ev.get("ssw_peak")
        if pk is not None:
            ax_r.axvline(pk, color="red", lw=1.8, ls="--", alpha=0.8, zorder=7)
            ax_r.text(pk, 0.95, " SSW Peak", transform=ax_r.get_xaxis_transform(),
                      fontsize=8.5, color="red", fontweight="bold", alpha=0.9, va="top")

        # Reg parameters box
        ax_r.text(
            0.015, 0.95,
            f"slope a={slope:.5f}  intercept b={intercept:.4f}  corr(y,Ap)={r:.3f}",
            transform=ax_r.transAxes,
            fontsize=8.0, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

        ax_r.set_title(f"{ev['label']}  ({ev['sat']})", fontsize=11, fontweight="bold")
        ax_r.grid(True, ls=":", alpha=0.40)

        if ev["mode"] == "date":
            ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax_r.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            plt.setp(ax_r.xaxis.get_majorticklabels(), rotation=15, ha="right")
        ax_r.set_xlabel(f"{'DOY' if ev['mode']=='doy' else 'Date'} ({ev['year']})", fontsize=9.5)

        # Combine legends for each row panel
        lines1, labels1 = ax_r.get_legend_handles_labels()
        lines2, labels2 = ax_t.get_legend_handles_labels()
        lines3, labels3 = ax_a.get_legend_handles_labels()
        ax_r.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                    loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved combined figure: {OUT_PNG}")


if __name__ == "__main__":
    main()
