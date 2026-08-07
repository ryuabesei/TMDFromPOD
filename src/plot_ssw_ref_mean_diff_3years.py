r"""
plot_ssw_ref_mean_diff_3years.py

Purpose:
    For each of the 3 SSW events, compute the latitudinal profile of:
        mean(Delta y_Ap during SSW period) - mean(Delta y_Ap during Reference period)

    where:
        Delta y_Ap = density_ratio_msis - (a * AP_AVG + b)
        Reference period = quiet periods before and after the SSW (ref_dates)
        SSW period = the active period between the two reference windows

    Plot:
        Left  : 3-event overlay on one panel (latitude vs mean difference)
        Right : 3 separate panels stacked vertically (one per event)

Output:
    Figure/Ap_removal/ssw_ref_mean_diff_3years.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Paths ────────────────────────────────────────────────────────────────────
P2018   = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019   = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021   = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")
OUT_DIR = Path("Figure/Ap_removal")
OUT_PNG = OUT_DIR / "ssw_ref_mean_diff_3years.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Grid Settings ────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -60.0, 60.0
LAT_BIN = 5.0   # 5-deg bins for smooth latitudinal profile

# ─── Event Configurations ─────────────────────────────────────────────────────
EVENTS = [
    dict(
        year=2018,
        label="2018 NH SSW (SWARM-A)",
        hemisphere="NH",
        color="#1f77b4",
        parquet=P2018,
        date_start=pd.Timestamp("2018-01-30", tz="UTC"),
        date_end=pd.Timestamp("2018-03-06 23:59:59", tz="UTC"),
        # Reference: quiet periods at start and end
        ref_dates=[
            (pd.Timestamp("2018-01-30", tz="UTC"), pd.Timestamp("2018-02-09 23:59:59", tz="UTC")),
            (pd.Timestamp("2018-03-02", tz="UTC"), pd.Timestamp("2018-03-06 23:59:59", tz="UTC")),
        ],
        # SSW active: between the two reference windows
        ssw_dates=[
            (pd.Timestamp("2018-02-10", tz="UTC"), pd.Timestamp("2018-03-01 23:59:59", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2018-02-12", tz="UTC"),
        ssw_peak_label="SSW Peak\n(DOY 43)",
    ),
    dict(
        year=2019,
        label="2019 SH SSW (SWARM-A)",
        hemisphere="SH",
        color="#d62728",
        parquet=P2019,
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26 23:59:59", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23 23:59:59", tz="UTC")),
        ],
        ssw_dates=[
            (pd.Timestamp("2019-08-27", tz="UTC"), pd.Timestamp("2019-09-19 23:59:59", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"),
        ssw_peak_label="SSW Peak\n(Sep 19)",
    ),
    dict(
        year=2021,
        label="2021 NH SSW (SWARM-C)",
        hemisphere="NH",
        color="#2ca02c",
        parquet=P2021,
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end=pd.Timestamp("2021-02-05 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29 23:59:59", tz="UTC")),
            (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05 23:59:59", tz="UTC")),
        ],
        ssw_dates=[
            (pd.Timestamp("2020-12-30", tz="UTC"), pd.Timestamp("2021-01-31 23:59:59", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2021-01-04", tz="UTC"),
        ssw_peak_label="SSW Peak\n(Jan 04)",
    ),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_density(ev: dict) -> pd.DataFrame:
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


def apply_detrend(df: pd.DataFrame) -> pd.DataFrame:
    """Fit linear Ap detrend on all data in the event window and compute delta_y_ap."""
    daily = df.groupby("date").agg(
        ratio=("density_ratio_msis", "median"),
        ap=("AP_AVG", "mean"),
    ).reset_index()
    x, y = daily["ap"].values, daily["ratio"].values
    p = np.polyfit(x, y, 1)
    slope, intercept = p[0], p[1]
    df = df.copy()
    df["delta_y_ap"] = df["density_ratio_msis"] - (slope * df["AP_AVG"] + intercept)
    return df, slope, intercept


def period_mask(df: pd.DataFrame, periods: list[tuple]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for s, e in periods:
        mask |= (df["datetime"] >= s) & (df["datetime"] <= e)
    return mask


def lat_profile(df: pd.DataFrame, lat_bins: np.ndarray, col: str = "delta_y_ap") -> pd.Series:
    """Compute median of col in each latitude bin. Returns Series indexed by lat bin centre."""
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    df = df.copy()
    df["lat_i"] = np.digitize(df["lat"].values, lat_bins) - 1
    valid = (df["lat_i"] >= 0) & (df["lat_i"] < len(lat_centers)) & np.isfinite(df[col].values)
    result = df[valid].groupby("lat_i")[col].median()
    # reindex to all bins
    s = pd.Series(np.nan, index=range(len(lat_centers)))
    s.update(result)
    s.index = lat_centers
    return s


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    results = []
    for ev in EVENTS:
        print(f"Processing {ev['label']}...")
        df = load_density(ev)
        df, slope, intercept = apply_detrend(df)

        df_ref = df[period_mask(df, ev["ref_dates"])]
        df_ssw = df[period_mask(df, ev["ssw_dates"])]

        prof_ref = lat_profile(df_ref, lat_bins)
        prof_ssw = lat_profile(df_ssw, lat_bins)
        diff     = prof_ssw - prof_ref

        results.append({
            "ev":       ev,
            "prof_ref": prof_ref,
            "prof_ssw": prof_ssw,
            "diff":     diff,
            "slope":    slope,
            "intercept": intercept,
        })
        print(f"  ref n={len(df_ref):,}  ssw n={len(df_ssw):,}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        r"SSW − Reference Period Mean Difference: $\langle\Delta y_{Ap}\rangle_{\rm SSW} - \langle\Delta y_{Ap}\rangle_{\rm Ref}$"
        "\n(Latitudinal Profile, Ap-Detrended Density Ratio Residual)",
        fontsize=13, fontweight="bold", y=1.00,
    )

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08,
                           width_ratios=[1.6, 1, 1, 1])

    # ── Left: all 3 events overlay ────────────────────────────────────────────
    ax_all = fig.add_subplot(gs[0, 0])

    for res in results:
        ev   = res["ev"]
        diff = res["diff"]
        ax_all.plot(
            diff.values, diff.index,
            color=ev["color"], lw=2.2, marker="o", ms=5,
            label=ev["label"],
        )

    ax_all.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax_all.axhline(0, color="gray",  lw=0.8, ls=":", alpha=0.5)
    ax_all.set_xlabel(
        r"$\langle\Delta y_{Ap}\rangle_{\rm SSW} - \langle\Delta y_{Ap}\rangle_{\rm Ref}$",
        fontsize=11, fontweight="bold",
    )
    ax_all.set_ylabel("Geographic Latitude (°)", fontsize=11, fontweight="bold")
    ax_all.set_ylim(LAT_MIN, LAT_MAX)
    ax_all.set_yticks(range(int(LAT_MIN), int(LAT_MAX) + 1, 20))
    ax_all.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax_all.grid(True, linestyle=":", alpha=0.5)
    ax_all.set_title("All 3 Events", fontsize=11, fontweight="bold")
    # shade NH / EQ / SH regions
    ax_all.axhspan(  0,  60, color="lightyellow", alpha=0.25, zorder=0)
    ax_all.axhspan(-60,   0, color="lightcyan",   alpha=0.25, zorder=0)
    ax_all.text(ax_all.get_xlim()[0] if ax_all.get_xlim()[0] > -1 else -0.25,
                30, "NH", color="goldenrod", fontsize=9, fontstyle="italic", va="center")
    ax_all.text(ax_all.get_xlim()[0] if ax_all.get_xlim()[0] > -1 else -0.25,
               -30, "SH", color="steelblue",  fontsize=9, fontstyle="italic", va="center")

    # ── Right: individual panels ───────────────────────────────────────────────
    axes_ind = []
    for col_i, res in enumerate(results):
        ax = fig.add_subplot(gs[0, col_i + 1], sharey=ax_all if col_i == 0 else axes_ind[0])
        axes_ind.append(ax)
        ev   = res["ev"]
        diff = res["diff"]
        prof_ref = res["prof_ref"]
        prof_ssw = res["prof_ssw"]

        # Fill positive/negative difference
        ax.fill_betweenx(
            diff.index,
            0, diff.values,
            where=(diff.values >= 0),
            color="#d62728", alpha=0.35, label="SSW > Ref",
        )
        ax.fill_betweenx(
            diff.index,
            0, diff.values,
            where=(diff.values < 0),
            color="#1f77b4", alpha=0.35, label="SSW < Ref",
        )
        ax.plot(diff.values, diff.index, color=ev["color"], lw=2.0, marker="o", ms=4.5)

        ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
        ax.axhline(0, color="gray",  lw=0.8, ls=":", alpha=0.5)
        ax.set_xlabel(
            r"$\langle\Delta y_{Ap}\rangle_{\rm SSW} - \langle\Delta y_{Ap}\rangle_{\rm Ref}$",
            fontsize=9, fontweight="bold",
        )
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.tick_params(axis="y", labelleft=False)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85)
        ax.set_title(f"{ev['year']} ({ev['hemisphere']})", fontsize=11, fontweight="bold",
                     color=ev["color"])

        # Annotation: slope & intercept
        ann = f"a={res['slope']:.4f}\nb={res['intercept']:.4f}"
        ax.text(0.04, 0.97, ann, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

        ax.axhspan(  0,  60, color="lightyellow", alpha=0.20, zorder=0)
        ax.axhspan(-60,   0, color="lightcyan",   alpha=0.20, zorder=0)

    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
