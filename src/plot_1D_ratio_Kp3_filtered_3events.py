"""
plot_1D_ratio_Kp3_filtered_3events.py

Purpose:
    Plot daily-median density_ratio_msis (rho_obs / rho_MSIS_real) split by
    LT sector and latitude band.

    Each panel overlays TWO lines:
        - Solid bold line  : Kp < 3 only (days with daily-mean Ap >= 15 removed)
        - Light dashed line: all data (no filter)

    Filter criterion:
        Remove entire days when daily-mean Ap (AP_AVG column) >= 15
        (Kp = 3 corresponds to Ap = 15 by the official conversion table).

    For each event, produce one figure per satellite:
        - 3 rows = latitude bands (High 40-60°, Mid 20-40°, Low 0-20°)
        - 2 columns = LT sectors (Dawn/Dusk for SWARM-A/C, Midnight/Noon for SWARM-B)

    Kp-removed days are shaded in light red.

Events:
    2018 NH SSW  (SWARM-A, B, C)   DOY 30–65
    2019 SH SSW  (SWARM-A, B, C)   2019-08-20 to 2019-09-23
    2021 NH SSW  (SWARM-C, GRACE-FO)  2020-12-25 to 2021-02-05

Output:
    Figure/Kp3_filtered/1D_ratio_Kp3_<year>_<satellite>.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# ─── Kp < 3 threshold (Ap units) ─────────────────────────────────────────────
AP_KP3 = 15.0

# ─── LT sector definitions ────────────────────────────────────────────────────
LT_DAWN_DUSK = [
    dict(label="Dawn  (LT 2.5–8.5 h)",  lt_min=2.5,  lt_max=8.5,  color="#1a6faf"),
    dict(label="Dusk  (LT 14.5–20.5 h)", lt_min=14.5, lt_max=20.5, color="#e07b39"),
]
LT_MIDNIGHT_NOON = [
    dict(label="Midnight (LT 0–4 h)",  lt_min=0,  lt_max=4,  color="#6a0dad"),
    dict(label="Noon     (LT 12–16 h)", lt_min=12, lt_max=16, color="#c0392b"),
]
LT_ALL = [
    dict(label="All LT", lt_min=0, lt_max=24, color="#2ca02c"),
]

# ─── Latitude bands ──────────────────────────────────────────────────────────
LAT_BANDS = [
    ("High  (40–60°)", 40.0, 60.0),
    ("Mid   (20–40°)", 20.0, 40.0),
    ("Low   ( 0–20°)",  0.0, 20.0),
]

VALUE_COL = "density_ratio_msis"

# ─── Event configurations ─────────────────────────────────────────────────────
EVENTS = [
    # ── 2018 NH SSW ──────────────────────────────────────────────────────────
    dict(
        year=2018, title="2018 NH SSW",
        date_start=pd.Timestamp("2018-01-29", tz="UTC"),
        date_end  =pd.Timestamp("2018-03-07", tz="UTC"),
        ref1_start=pd.Timestamp("2018-01-29", tz="UTC"),
        ref1_end  =pd.Timestamp("2018-02-05", tz="UTC"),
        ref2_start=pd.Timestamp("2018-02-25", tz="UTC"),
        ref2_end  =pd.Timestamp("2018-03-07", tz="UTC"),
        ssw_start =pd.Timestamp("2018-02-06", tz="UTC"),
        ssw_end   =pd.Timestamp("2018-02-24", tz="UTC"),
        ssw_peak  =pd.Timestamp("2018-02-12", tz="UTC"),  # COSMIC T10hPa peak DOY43
        peak_label="COSMIC T10hPa peak",
        xlabel="Date (2018)",
        satellites=[
            dict(label="SWARM-A", lt_sectors=LT_DAWN_DUSK,
                 parquet="normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
            dict(label="SWARM-B", lt_sectors=LT_MIDNIGHT_NOON,
                 parquet="normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
            dict(label="SWARM-C", lt_sectors=LT_DAWN_DUSK,
                 parquet="normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        ],
    ),
    # ── 2019 SH SSW ──────────────────────────────────────────────────────────
    dict(
        year=2019, title="2019 SH SSW",
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end  =pd.Timestamp("2019-09-23", tz="UTC"),
        ref1_start=pd.Timestamp("2019-08-20", tz="UTC"),
        ref1_end  =pd.Timestamp("2019-08-26", tz="UTC"),
        ref2_start=pd.Timestamp("2019-09-20", tz="UTC"),
        ref2_end  =pd.Timestamp("2019-09-23", tz="UTC"),
        ssw_start =pd.Timestamp("2019-08-27", tz="UTC"),
        ssw_end   =pd.Timestamp("2019-09-19", tz="UTC"),
        ssw_peak  =pd.Timestamp("2019-09-19", tz="UTC"),
        peak_label="ERA5 T10hPa peak",
        xlabel="Date (2019)",
        satellites=[
            dict(label="SWARM-A", lt_sectors=LT_DAWN_DUSK,
                 parquet="normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
            dict(label="SWARM-B", lt_sectors=LT_MIDNIGHT_NOON,
                 parquet="normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
            dict(label="SWARM-C", lt_sectors=LT_DAWN_DUSK,
                 parquet="normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        ],
    ),
    # ── 2021 NH SSW ──────────────────────────────────────────────────────────
    dict(
        year=2021, title="2021 NH SSW",
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end  =pd.Timestamp("2021-02-05", tz="UTC"),
        ref1_start=pd.Timestamp("2020-12-25", tz="UTC"),
        ref1_end  =pd.Timestamp("2020-12-29", tz="UTC"),
        ref2_start=pd.Timestamp("2021-02-01", tz="UTC"),
        ref2_end  =pd.Timestamp("2021-02-05", tz="UTC"),
        ssw_start =pd.Timestamp("2020-12-30", tz="UTC"),
        ssw_end   =pd.Timestamp("2021-01-31", tz="UTC"),
        ssw_peak  =pd.Timestamp("2021-01-04", tz="UTC"),
        peak_label="T10hPa Peak (01/04)",
        xlabel="Date (2020–2021)",
        satellites=[
            dict(label="SWARM-C", lt_sectors=LT_DAWN_DUSK,
                 parquet="normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet"),
            dict(label="GRACE-FO", lt_sectors=LT_ALL,
                 parquet="normalizeddata/2021/grace_fo_dns_2021_normalized_with_LT_removed.parquet"),
        ],
    ),
]

OUT_BASE = Path("Figure/Kp3_filtered")


def apply_kp3_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Remove days with daily-mean AP_AVG >= 15 (Kp >= 3).
    Returns filtered df and set of removed dates."""
    if "AP_AVG" not in df.columns:
        print("  [WARNING] AP_AVG column not found — skipping Kp filter")
        return df, set()
    if "date" not in df.columns:
        df = df.copy()
        df["date"] = df["datetime"].dt.normalize()
    daily_ap = df.groupby("date")["AP_AVG"].mean()
    removed = set(daily_ap[daily_ap >= AP_KP3].index)
    df_filt = df[~df["date"].isin(removed)].copy()
    print(f"  Kp<3 filter: {len(removed)} days removed "
          f"({[str(d.date()) for d in sorted(removed)]})")
    return df_filt, removed


def plot_event_satellite(event: dict, sat: dict) -> None:
    parquet_path = Path(sat["parquet"])
    if not parquet_path.exists():
        print(f"  [SKIP] File not found: {parquet_path}")
        return

    print(f"\n=== {event['title']} | {sat['label']} ===")
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        print("  [SKIP] Latitude column not found")
        return

    df = df.dropna(subset=["datetime", lat_col, VALUE_COL])
    df = df[(df["datetime"] >= event["date_start"]) & (df["datetime"] <= event["date_end"])]
    df["date"] = df["datetime"].dt.normalize()

    # Kp < 3 filter
    df_filt, removed_days = apply_kp3_filter(df)

    lt_sectors = sat["lt_sectors"]
    n_bands = len(LAT_BANDS)
    n_lt    = len(lt_sectors)

    x_min = event["date_start"] - pd.Timedelta(hours=12)
    x_max = event["date_end"]   + pd.Timedelta(hours=12)

    fig, axes = plt.subplots(
        n_bands, n_lt,
        figsize=(6.5 * n_lt, 3.2 * n_bands),
        sharex=True, sharey="row"
    )
    if n_lt == 1:
        axes = axes[:, np.newaxis]
    fig.subplots_adjust(hspace=0.10, wspace=0.06)

    for col_idx, lt in enumerate(lt_sectors):
        lt_label = lt["label"]
        color    = lt["color"]

        if lt["lt_max"] > lt["lt_min"]:
            df_lt = df_filt[(df_filt["lst_h"] >= lt["lt_min"]) & (df_filt["lst_h"] < lt["lt_max"])]
        else:
            # wrap-around (e.g. 22–5)
            df_lt = df_filt[(df_filt["lst_h"] >= lt["lt_min"]) | (df_filt["lst_h"] < lt["lt_max"])]
        print(f"  {lt_label}: {len(df_lt):,} obs after Kp filter")

        for bi, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
            ax = axes[bi, col_idx]

            # ── All-data (unfiltered) daily series ──────────────────────────
            if lt["lt_max"] > lt["lt_min"]:
                df_lt_all = df[(df["lst_h"] >= lt["lt_min"]) & (df["lst_h"] < lt["lt_max"])]
            else:
                df_lt_all = df[(df["lst_h"] >= lt["lt_min"]) | (df["lst_h"] < lt["lt_max"])]
            mask_all = (df_lt_all[lat_col].abs() >= lat_lo) & (df_lt_all[lat_col].abs() < lat_hi)
            daily_all = df_lt_all[mask_all].groupby("date")[VALUE_COL].median()

            # ── Kp<3 filtered daily series ──────────────────────────────────
            mask  = (df_lt[lat_col].abs() >= lat_lo) & (df_lt[lat_col].abs() < lat_hi)
            sub   = df_lt[mask]
            daily = sub.groupby("date")[VALUE_COL].median()

            # ── Reference median (based on filtered data) ────────────────────
            ref_mask = (
                ((daily.index >= event["ref1_start"]) & (daily.index <= event["ref1_end"])) |
                ((daily.index >= event["ref2_start"]) & (daily.index <= event["ref2_end"]))
            )
            ref_val = float(daily[ref_mask].median()) if ref_mask.any() else float("nan")

            # ── Shaded regions ──────────────────────────────────────────────
            ax.axvspan(event["ref1_start"], event["ref1_end"], color="lightblue",  alpha=0.25)
            ax.axvspan(event["ref2_start"], event["ref2_end"], color="lightblue",  alpha=0.25)
            ax.axvspan(event["ssw_start"],  event["ssw_end"],  color="lightyellow", alpha=0.40)

            # ── Removed (high Kp) days — shade in red ───────────────────────
            for rd in sorted(removed_days):
                ax.axvspan(rd, rd + pd.Timedelta(days=1),
                           color="#ff4444", alpha=0.18, zorder=1)

            # ── Reference line ───────────────────────────────────────────────
            if not np.isnan(ref_val):
                ax.axhline(ref_val, color="gray", linewidth=0.8, linestyle="--", zorder=2)

            # ── SSW peak line ────────────────────────────────────────────────
            ax.axvline(event["ssw_peak"], color="red", linewidth=1.5,
                       linestyle="--", zorder=5)

            # ── All-data line (light dashed, behind) ─────────────────────────
            ax.plot(daily_all.index, daily_all.values,
                    color=color, linewidth=1.2, linestyle="--",
                    alpha=0.40, marker="o", markersize=3,
                    zorder=3, label="All data")

            # ── Kp<3 filtered line (solid bold, front) ───────────────────────
            ax.plot(daily.index, daily.values,
                    color=color, linewidth=2.2, marker="o", markersize=4,
                    zorder=4, label="Kp < 3 only")

            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(axis="y", alpha=0.3, linewidth=0.7)
            ax.tick_params(axis="y", labelleft=True)

            if col_idx == 0:
                ax.set_ylabel("ρ_ratio\n(ρ_obs / ρ_MSIS)", fontsize=9)

            label_txt = band_label.strip()
            ax.text(0.01, 0.97, label_txt, transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="left")
            if not np.isnan(ref_val):
                ax.text(0.99, 0.97, f"ref = {ref_val:.3f}",
                        transform=ax.transAxes, fontsize=8, va="top", ha="right",
                        color="gray")

            if bi == 0:
                ax.set_title(lt_label, fontsize=11, fontweight="bold", pad=6, color=color)
            if bi == n_bands - 1:
                ax.set_xlabel(event["xlabel"], fontsize=10)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_elems = [
        mpatches.Patch(facecolor="lightblue",   alpha=0.4, label="Non-SSW ref"),
        mpatches.Patch(facecolor="lightyellow", alpha=0.7, label="SSW period"),
        mpatches.Patch(facecolor="#ff4444",     alpha=0.3, label=f"Kp≥3 removed (Ap≥{AP_KP3:.0f})"),
        plt.Line2D([0], [0], color="steelblue", lw=2.2, ls="-",  marker="o", markersize=4, label="Kp < 3 only (filtered)"),
        plt.Line2D([0], [0], color="steelblue", lw=1.2, ls="--", marker="o", markersize=3, alpha=0.5, label="All data (unfiltered)"),
        plt.Line2D([0], [0], color="red",  lw=1.5, ls="--", label=event["peak_label"]),
        plt.Line2D([0], [0], color="gray", lw=0.8, ls="--", label="Non-SSW ref median (Kp<3)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.04))

    removed_str = ", ".join(str(d.date()) for d in sorted(removed_days)) or "none"
    fig.suptitle(
        f"{sat['label']}  ρ_ratio  ({event['title']})  |  Kp < 3 filter\n"
        f"Removed days (Kp≥3): {removed_str}",
        fontsize=11, fontweight="bold", y=1.01
    )

    out_png = OUT_BASE / f"1D_ratio_Kp3_{event['year']}_{sat['label'].replace('-', '')}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


def main() -> None:
    for event in EVENTS:
        for sat in event["satellites"]:
            plot_event_satellite(event, sat)
    print("\nAll done.")


if __name__ == "__main__":
    main()
