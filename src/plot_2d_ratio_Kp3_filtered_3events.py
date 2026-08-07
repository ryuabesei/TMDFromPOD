"""
plot_2d_ratio_Kp3_filtered_3events.py

Purpose:
    Plot 2D (DOY/Date vs Latitude) maps of:
        Row 0: rho_ratio  (density_ratio_msis = rho_obs / rho_MSIS_real)
        Row 1: delta_ratio (residual vs non-SSW reference)
    with Kp < 3 filtering applied (days with daily-mean Ap >= 15 are masked to NaN).

    Kp < 3 threshold: AP_AVG daily mean < 15 (Kp=3 ≡ Ap=15 by official table).

Events:
    2018 NH SSW  (SWARM-A, B, C)   DOY 20-80
    2019 SH SSW  (SWARM-A, B, C)   DOY 232-266  (Aug 20 – Sep 23)
    2021 NH SSW  (SWARM-C, GRACE-FO)

Output:
    Figure/Kp3_filtered/2D_ratio_Kp3_<year>_<satellite>.png
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ─── Kp < 3 threshold ────────────────────────────────────────────────────────
AP_KP3 = 15.0

# ─── Global grid settings ────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -60, 60
LAT_BIN  = 3.0
DOY_BIN  = 1.0
N_LEVELS = 21

VALUE_COL = "density_ratio_msis"

# ─── Event configurations ─────────────────────────────────────────────────────
EVENTS = [
    # ── 2018 NH SSW ──────────────────────────────────────────────────────────
    dict(
        year=2018, title="2018 NH SSW",
        t_start="2018-01-20", t_end="2018-03-21",
        doy_min=20.0, doy_max=80.0,
        doy_ref1=(20, 40), doy_ref2=(61, 80),
        doy_ssw=(41, 60),
        ssw_peak_doy=43,
        x_label="Day of Year 2018 (DOY 20–80)",
        use_doy=True,
        satellites=[
            dict(label="SWARM-A",
                 parquet="normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet",
                 sec1=(4, 11),   sec1_wrap=False, sec1_title="Morning (04–11 LT)",
                 sec2=(16, 23),  sec2_wrap=False, sec2_title="Evening (16–23 LT)"),
            dict(label="SWARM-B",
                 parquet="normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet",
                 sec1=(22, 5),   sec1_wrap=True,  sec1_title="Nightside (22–05 LT)",
                 sec2=(11, 17),  sec2_wrap=False, sec2_title="Dayside (11–17 LT)"),
            dict(label="SWARM-C",
                 parquet="normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet",
                 sec1=(4, 11),   sec1_wrap=False, sec1_title="Morning (04–11 LT)",
                 sec2=(16, 23),  sec2_wrap=False, sec2_title="Evening (16–23 LT)"),
        ],
    ),
    # ── 2019 SH SSW ──────────────────────────────────────────────────────────
    dict(
        year=2019, title="2019 SH SSW",
        t_start="2019-08-20", t_end="2019-09-23",
        doy_min=232.0, doy_max=266.0,
        doy_ref1=(232, 238), doy_ref2=(263, 266),
        doy_ssw=(239, 262),
        ssw_peak_doy=262,   # Sep 19 = DOY 262
        x_label="Day of Year 2019 (DOY 232–266)",
        use_doy=True,
        satellites=[
            dict(label="SWARM-A",
                 parquet="normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet",
                 sec1=(2.5, 8.5),  sec1_wrap=False, sec1_title="Dawn (02.5–08.5 LT)",
                 sec2=(14.5, 20.5),sec2_wrap=False, sec2_title="Dusk (14.5–20.5 LT)"),
            dict(label="SWARM-B",
                 parquet="normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet",
                 sec1=(0, 4),   sec1_wrap=False, sec1_title="Midnight (00–04 LT)",
                 sec2=(12, 16), sec2_wrap=False, sec2_title="Noon (12–16 LT)"),
            dict(label="SWARM-C",
                 parquet="normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet",
                 sec1=(2.5, 8.5),  sec1_wrap=False, sec1_title="Dawn (02.5–08.5 LT)",
                 sec2=(14.5, 20.5),sec2_wrap=False, sec2_title="Dusk (14.5–20.5 LT)"),
        ],
    ),
    # ── 2021 NH SSW ──────────────────────────────────────────────────────────
    dict(
        year=2021, title="2021 NH SSW",
        t_start="2020-12-25", t_end="2021-02-05",
        doy_min=360.0, doy_max=402.0,   # DOY 360 = Dec 26, DOY 402 ≈ Feb 5
        doy_ref1=(360, 364), doy_ref2=(397, 402),
        doy_ssw=(365, 396),
        ssw_peak_doy=369,   # Jan 4 = DOY 4 → offset 365 → 369
        x_label="Day of Year (2020→2021)",
        use_doy=True,
        satellites=[
            dict(label="SWARM-C",
                 parquet="normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet",
                 sec1=(2.5, 8.5),  sec1_wrap=False, sec1_title="Dawn (02.5–08.5 LT)",
                 sec2=(14.5, 20.5),sec2_wrap=False, sec2_title="Dusk (14.5–20.5 LT)"),
            dict(label="GRACE-FO",
                 parquet="normalizeddata/2021/grace_fo_dns_2021_normalized_with_LT_removed.parquet",
                 sec1=(0, 24),  sec1_wrap=False, sec1_title="All LT",
                 sec2=None,     sec2_wrap=False, sec2_title=None),
        ],
    ),
]

OUT_BASE = Path("Figure/Kp3_filtered")


# ─── Utilities ────────────────────────────────────────────────────────────────

def add_doy_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fractional DOY, with 2021 continuing past Dec 31 as DOY 366+."""
    dt = df["datetime"]
    doy_frac = (
        dt.dt.dayofyear
        + dt.dt.hour / 24.0
        + dt.dt.minute / 1440.0
    )
    # For 2021 event: treat 2020-days as DOY 0-366, 2021-days as +366
    year = dt.dt.year
    doy_frac = doy_frac.where(year == 2021, doy_frac - 366 + 360)
    # For years that are just themselves (2018, 2019), keep as-is
    # (2020 Dec dates will come out as small numbers; handled by caller)
    df = df.copy()
    df["DOY"] = doy_frac
    return df


def add_doy_simple(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    df = df.copy()
    df["DOY"] = dt.dt.dayofyear + dt.dt.hour / 24.0 + dt.dt.minute / 1440.0
    return df


def apply_kp3_mask(df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Zero out (mask) entire days where daily-mean AP_AVG >= AP_KP3."""
    if "AP_AVG" not in df.columns:
        print("  [WARNING] AP_AVG not found — skipping Kp filter")
        return df, set()
    date_col = df["datetime"].dt.normalize()
    daily_ap = df.groupby(date_col)["AP_AVG"].mean()
    removed_dates = set(daily_ap[daily_ap >= AP_KP3].index)
    df_filt = df[~date_col.isin(removed_dates)].copy()
    print(f"  Kp<3: removed {len(removed_dates)} days "
          f"({[str(d.date()) for d in sorted(removed_dates)]})")
    return df_filt, removed_dates


def grid_median(
    df: pd.DataFrame,
    doy_bins: np.ndarray,
    lat_bins: np.ndarray,
) -> np.ndarray:
    Z = np.full((len(lat_bins) - 1, len(doy_bins) - 1), np.nan)
    if len(df) == 0:
        return Z
    doy = df["DOY"].to_numpy()
    lat = df["lat"].to_numpy()
    val = df[VALUE_COL].to_numpy()
    ok = np.isfinite(doy) & np.isfinite(lat) & np.isfinite(val)
    doy, lat, val = doy[ok], lat[ok], val[ok]
    doy_i = np.digitize(doy, doy_bins) - 1
    lat_i = np.digitize(lat, lat_bins) - 1
    ok2 = (
        (doy_i >= 0) & (doy_i < len(doy_bins) - 1)
        & (lat_i >= 0) & (lat_i < len(lat_bins) - 1)
    )
    doy_i, lat_i, val = doy_i[ok2], lat_i[ok2], val[ok2]
    bucket: dict = defaultdict(list)
    for i, j, v in zip(lat_i, doy_i, val):
        bucket[(i, j)].append(float(v))
    for (i, j), arr in bucket.items():
        Z[i, j] = float(np.median(arr))
    return Z


def compute_residual(Z: np.ndarray, doy_bins: np.ndarray,
                     ref1: tuple, ref2: tuple) -> np.ndarray:
    centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    ref_mask = (
        ((centers >= ref1[0]) & (centers <= ref1[1])) |
        ((centers >= ref2[0]) & (centers <= ref2[1]))
    )
    ref_profile = np.nanmedian(Z[:, ref_mask], axis=1)
    return Z - ref_profile[:, np.newaxis]


def filter_lt(df: pd.DataFrame, sec: tuple, wrap: bool) -> pd.DataFrame:
    if sec is None:
        return df.iloc[:0]  # empty
    lo, hi = sec
    if wrap:
        return df[(df["lst_h"] >= lo) | (df["lst_h"] < hi)].copy()
    elif lo == 0 and hi == 24:
        return df.copy()
    else:
        return df[(df["lst_h"] >= lo) & (df["lst_h"] < hi)].copy()


def mark_removed_doy_cols(ax, removed_dates, year, is_2021=False):
    """Shade the DOY columns corresponding to removed dates."""
    for rd in sorted(removed_dates):
        d = pd.Timestamp(rd)
        doy = d.dayofyear
        if is_2021:
            # map 2021 dates to continuous DOY axis
            if d.year == 2021:
                doy_plot = doy + 365 - 365 + 360   # DOY 1 → 360+0? redo
                # Jan 1 = DOY 1 → 360 + 1 - 1 = 360? Let me just do:
                # 2020 Dec 25 = DOY 360 (continuous 0-based: 359 → 360 in our mapping)
                # 2021 Jan 4 = DOY 4 → 360 + 4 - 1 = 363? Let's use offset
                doy_plot = doy + 360 - 1   # Jan 1 → 360, Jan 4 → 363
            else:
                doy_plot = doy - 366 + 360  # Dec 25 = DOY 360
        else:
            doy_plot = doy
        ax.axvspan(doy_plot, doy_plot + 1, color="#ff4444", alpha=0.25, zorder=3)


# ─── Main plot function ────────────────────────────────────────────────────────

def plot_satellite(event: dict, sat: dict) -> None:
    parquet_path = Path(sat["parquet"])
    if not parquet_path.exists():
        print(f"  [SKIP] Not found: {parquet_path}")
        return

    print(f"\n=== {event['title']} | {sat['label']} ===")
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    # Unify latitude column name
    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        print("  [SKIP] No lat column")
        return
    if lat_col != "lat":
        df = df.rename(columns={lat_col: "lat"})

    df = df.dropna(subset=["datetime", "lat", VALUE_COL]).copy()

    # Time filter
    t0 = pd.Timestamp(event["t_start"], tz="UTC")
    t1 = pd.Timestamp(event["t_end"], tz="UTC")
    df = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].copy()
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)].copy()

    # DOY axis
    if event["year"] == 2021:
        # continuous DOY across year boundary: Dec 25 = 360, Jan 4 = 369
        def to_doy_2021(dt_series):
            doy = dt_series.dt.dayofyear.astype(float)
            yr  = dt_series.dt.year
            # 2020 dates → DOY_2020 - 366 + 360  (Dec 25=360, Dec 31=365)
            # 2021 dates → DOY_2021 + 365         (Jan 1=366, Feb 5=401)
            result = doy.copy()
            result[yr == 2020] = doy[yr == 2020] - 366 + 360
            result[yr == 2021] = doy[yr == 2021] + 365
            return result + df["datetime"].dt.hour / 24.0
        df["DOY"] = to_doy_2021(df["datetime"])
    else:
        df = add_doy_simple(df)

    print(f"  Total rows: {len(df):,}  DOY: {df['DOY'].min():.1f}–{df['DOY'].max():.1f}")

    # Kp < 3 filter
    df_filt, removed_dates = apply_kp3_mask(df)

    doy_min = event["doy_min"]
    doy_max = event["doy_max"]
    doy_bins = np.arange(doy_min, doy_max + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    X, Y = np.meshgrid(
        0.5 * (doy_bins[:-1] + doy_bins[1:]),
        0.5 * (lat_bins[:-1] + lat_bins[1:])
    )

    has_sec2 = sat["sec2"] is not None

    # Grids: filtered
    df1_filt = filter_lt(df_filt, sat["sec1"], sat["sec1_wrap"])
    Z_r1 = grid_median(df1_filt, doy_bins, lat_bins)
    Z_d1 = compute_residual(Z_r1, doy_bins, event["doy_ref1"], event["doy_ref2"])

    if has_sec2:
        df2_filt = filter_lt(df_filt, sat["sec2"], sat["sec2_wrap"])
        Z_r2 = grid_median(df2_filt, doy_bins, lat_bins)
        Z_d2 = compute_residual(Z_r2, doy_bins, event["doy_ref1"], event["doy_ref2"])

    # Color scale from filtered data
    all_ratio = Z_r1[np.isfinite(Z_r1)]
    if has_sec2:
        all_ratio = np.concatenate([all_ratio, Z_r2[np.isfinite(Z_r2)]])
    vmin_r = float(np.nanpercentile(all_ratio, 1))
    vmax_r = float(np.nanpercentile(all_ratio, 99))
    levels_ratio = np.linspace(vmin_r, vmax_r, N_LEVELS + 1)

    all_resid = Z_d1[np.isfinite(Z_d1)]
    if has_sec2:
        all_resid = np.concatenate([all_resid, Z_d2[np.isfinite(Z_d2)]])
    if len(all_resid) > 0:
        vmax_d = float(np.nanpercentile(np.abs(all_resid), 98))
    else:
        vmax_d = 0.1
    levels_resid = np.linspace(-vmax_d, vmax_d, N_LEVELS + 1)

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_cols = 2 if has_sec2 else 1
    fig = plt.figure(figsize=(8 * n_cols + 1, 10))
    gs  = gridspec.GridSpec(2, n_cols + 1,
                            width_ratios=[1] * n_cols + [0.045],
                            height_ratios=[1, 1],
                            wspace=0.08, hspace=0.20)

    ref1 = event["doy_ref1"]
    ref2 = event["doy_ref2"]
    ssw  = event["doy_ssw"]
    peak = event["ssw_peak_doy"]
    is21 = (event["year"] == 2021)

    def decorate(ax, title, ylabel=False):
        ax.axvspan(*ref1, color="lightblue",  alpha=0.18, lw=0)
        ax.axvspan(*ref2, color="lightblue",  alpha=0.18, lw=0)
        ax.axvspan(*ssw,  color="lightyellow", alpha=0.35, lw=0)
        # Removed days are left as NaN (white) — no extra shading needed
        ax.axvline(peak, color="red", lw=1.5, ls="--", alpha=0.85)
        ax.set_xlim(doy_min, doy_max)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.grid(alpha=0.15, color="white", lw=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        if ylabel:
            ax.set_ylabel("Geographic Latitude (°)", fontsize=11, fontweight="bold")
        else:
            ax.tick_params(axis="y", labelleft=False)

    # ── Row 0: rho_ratio ──────────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    cf00 = ax00.contourf(X, Y, Z_r1, levels=levels_ratio, cmap="plasma", extend="both")
    decorate(ax00, f"ρ_ratio — {sat['sec1_title']}", ylabel=True)

    if has_sec2:
        ax01 = fig.add_subplot(gs[0, 1])
        cf01 = ax01.contourf(X, Y, Z_r2, levels=levels_ratio, cmap="plasma", extend="both")
        decorate(ax01, f"ρ_ratio — {sat['sec2_title']}")
        cb_src = cf01
    else:
        cb_src = cf00

    cb_ax0 = fig.add_subplot(gs[0, -1])
    cbar0  = fig.colorbar(cb_src, cax=cb_ax0)
    cbar0.set_label("ρ_ratio  (obs / MSIS)", fontsize=10, fontweight="bold")

    # ── Row 1: delta_ratio ────────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    cf10 = ax10.contourf(X, Y, Z_d1, levels=levels_resid, cmap="RdBu_r", extend="both")
    decorate(ax10, f"Δρ_ratio — {sat['sec1_title']}", ylabel=True)
    ax10.set_xlabel(event["x_label"], fontsize=11, fontweight="bold")

    if has_sec2:
        ax11 = fig.add_subplot(gs[1, 1])
        cf11 = ax11.contourf(X, Y, Z_d2, levels=levels_resid, cmap="RdBu_r", extend="both")
        decorate(ax11, f"Δρ_ratio — {sat['sec2_title']}")
        ax11.set_xlabel(event["x_label"], fontsize=11, fontweight="bold")
        cb_src1 = cf11
    else:
        cb_src1 = cf10

    cb_ax1 = fig.add_subplot(gs[1, -1])
    cbar1  = fig.colorbar(cb_src1, cax=cb_ax1)
    cbar1.set_label("Δρ_ratio  (residual vs non-SSW ref)", fontsize=10, fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────────
    removed_str = ", ".join(str(pd.Timestamp(d).date()) for d in sorted(removed_dates)) or "none"
    legend_elems = [
        mpatches.Patch(facecolor="lightblue",   alpha=0.4,  label="Non-SSW ref"),
        mpatches.Patch(facecolor="lightyellow", alpha=0.7,  label="SSW period"),
        plt.Line2D([0], [0], color="red", lw=1.5, ls="--", label="SSW peak"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        f"{sat['label']}  ρ_ratio  ({event['title']})  |  Kp < 3 filter\n"
        f"Top: ρ_ratio  |  Bottom: Δρ_ratio (residual vs non-SSW ref)\n"
        f"Removed days (Kp≥3): {removed_str}",
        fontsize=12, fontweight="bold", y=1.01
    )

    out_png = OUT_BASE / f"2D_ratio_Kp3_{event['year']}_{sat['label'].replace('-', '').replace('/', '')}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


def main() -> None:
    for event in EVENTS:
        for sat in event["satellites"]:
            plot_satellite(event, sat)
    print("\nAll done.")


if __name__ == "__main__":
    main()
