r"""
plot_1d_delta_yap_3years.py

Purpose:
    Simple 1D time series plots of Ap-detrended density ratio residual
        Delta y_Ap = density_ratio_msis - (a * AP_AVG + b)
    for each of the 3 SSW events.
    Reference periods and SSW period are highlighted.

Output:
    Figure/Ap_removal/1D_delta_yap_2018.png
    Figure/Ap_removal/1D_delta_yap_2019.png
    Figure/Ap_removal/1D_delta_yap_2021.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms

# ─── Paths ────────────────────────────────────────────────────────────────────
P2018   = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019   = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021   = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")
OUT_DIR = Path("Figure/Ap_removal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Event Configurations ─────────────────────────────────────────────────────
EVENTS = [
    dict(
        year=2018,
        label="2018 NH SSW (SWARM-A)",
        parquet=P2018,
        date_start=pd.Timestamp("2018-01-30", tz="UTC"),
        date_end=pd.Timestamp("2018-03-06 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2018-01-30", tz="UTC"), pd.Timestamp("2018-02-09 23:59:59", tz="UTC")),
            (pd.Timestamp("2018-03-02", tz="UTC"), pd.Timestamp("2018-03-06 23:59:59", tz="UTC")),
        ],
        ssw_dates=(pd.Timestamp("2018-02-10", tz="UTC"), pd.Timestamp("2018-03-01 23:59:59", tz="UTC")),
        ssw_peak=pd.Timestamp("2018-02-12", tz="UTC"),
        ssw_peak_label="SSW Peak (DOY 43)",
        x_interval=4,
        color="#1f77b4",
    ),
    dict(
        year=2019,
        label="2019 SH SSW (SWARM-A)",
        parquet=P2019,
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26 23:59:59", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23 23:59:59", tz="UTC")),
        ],
        ssw_dates=(pd.Timestamp("2019-08-27", tz="UTC"), pd.Timestamp("2019-09-19 23:59:59", tz="UTC")),
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"),
        ssw_peak_label="SSW Peak (Sep 19)",
        x_interval=4,
        color="#d62728",
    ),
    dict(
        year=2021,
        label="2021 NH SSW (SWARM-C)",
        parquet=P2021,
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end=pd.Timestamp("2021-02-05 23:59:59", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29 23:59:59", tz="UTC")),
            (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05 23:59:59", tz="UTC")),
        ],
        ssw_dates=(pd.Timestamp("2020-12-30", tz="UTC"), pd.Timestamp("2021-01-31 23:59:59", tz="UTC")),
        ssw_peak=pd.Timestamp("2021-01-04", tz="UTC"),
        ssw_peak_label="SSW Peak (Jan 04)",
        x_interval=5,
        color="#2ca02c",
    ),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_and_detrend(ev: dict) -> pd.DataFrame:
    df = pd.read_parquet(ev["parquet"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for cname in ["lat", "latitude", "geod_lat"]:
        if cname in df.columns and cname != "lat":
            df = df.rename(columns={cname: "lat"})
            break
    df = df.dropna(subset=["datetime", "lat", "density_ratio_msis", "AP_AVG"])
    df = df[(df["datetime"] >= ev["date_start"]) & (df["datetime"] <= ev["date_end"])].copy()
    df["date"] = df["datetime"].dt.normalize()

    # Linear detrend: fit on all data in the event window
    daily = df.groupby("date").agg(
        ratio=("density_ratio_msis", "median"),
        ap=("AP_AVG", "mean"),
    ).reset_index()
    x, y = daily["ap"].values, daily["ratio"].values
    p = np.polyfit(x, y, 1)
    slope, intercept = p[0], p[1]
    r = float(np.corrcoef(x, y)[0, 1])

    df["delta_y_ap"] = df["density_ratio_msis"] - (slope * df["AP_AVG"] + intercept)
    daily["delta_ap"] = daily["ratio"] - (slope * daily["ap"] + intercept)

    return daily, slope, intercept, r


def period_mask_daily(daily: pd.DataFrame, periods: list[tuple]) -> pd.Series:
    mask = pd.Series(False, index=daily.index)
    for s, e in periods:
        mask |= (daily["date"] >= s.normalize()) & (daily["date"] <= e.normalize())
    return mask


# ─── Single-event plot ────────────────────────────────────────────────────────
def plot_event(ev: dict) -> None:
    year    = ev["year"]
    out_png = OUT_DIR / f"1D_delta_yap_{year}.png"
    print(f"Processing {ev['label']}...")

    daily, slope, intercept, r = load_and_detrend(ev)

    # Reference period stats
    ref_mask = period_mask_daily(daily, ev["ref_dates"])
    ssw_s, ssw_e = ev["ssw_dates"]
    ssw_mask = (daily["date"] >= ssw_s.normalize()) & (daily["date"] <= ssw_e.normalize())

    ref_mean = daily.loc[ref_mask, "delta_ap"].mean()
    ssw_mean = daily.loc[ssw_mask, "delta_ap"].mean()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Reference period shading (teal)
    for r_start, r_end in ev["ref_dates"]:
        ax.axvspan(r_start, r_end, color="#008080", alpha=0.15, zorder=0,
                   label="Reference period" if r_start == ev["ref_dates"][0][0] else "")

    # SSW period shading (yellow)
    ax.axvspan(ssw_s, ssw_e, color="#f0c060", alpha=0.20, zorder=0, label="SSW period")

    # Horizontal reference lines
    ax.axhline(0.0,      color="gray",       lw=1.0, ls=":",  alpha=0.7)
    ax.axhline(ref_mean, color="#008080",     lw=1.5, ls="--", alpha=0.85,
               label=f"Ref mean = {ref_mean:+.4f}")
    ax.axhline(ssw_mean, color="#d45500",     lw=1.5, ls="--", alpha=0.85,
               label=f"SSW mean = {ssw_mean:+.4f}")

    # Main line: daily Δy_Ap
    ax.plot(daily["date"], daily["delta_ap"],
            "o-", color=ev["color"], lw=2.0, ms=4.5,
            label=r"Daily median $\Delta y_{Ap}$", zorder=3)

    # Colour-code markers by period
    ax.plot(daily.loc[ref_mask, "date"], daily.loc[ref_mask, "delta_ap"],
            "o", color="#008080", ms=5.5, zorder=4, label="Ref data points")
    ax.plot(daily.loc[ssw_mask, "date"], daily.loc[ssw_mask, "delta_ap"],
            "o", color="#d45500", ms=5.5, zorder=4, label="SSW data points")

    # SSW peak vertical line (pinned label at top)
    ax.axvline(ev["ssw_peak"], color="red", ls="--", lw=1.8)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(ev["ssw_peak"], 0.97, f" {ev['ssw_peak_label']}",
            transform=trans, color="red", fontweight="bold",
            verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85))

    # Difference annotation
    diff = ssw_mean - ref_mean
    ax.annotate(
        f"SSW − Ref = {diff:+.4f}",
        xy=(ev["ssw_peak"], ssw_mean),
        xytext=(0.55, 0.12), textcoords="axes fraction",
        fontsize=10, fontweight="bold", color="#d45500",
        arrowprops=dict(arrowstyle="->", color="#d45500", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d45500", alpha=0.9),
    )

    # Regression info
    ax.text(0.01, 0.97,
            f"slope a = {slope:.5f}   intercept b = {intercept:.5f}   corr(y, Ap) = {r:.3f}",
            transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85))

    ax.set_xlabel("Date (MM/DD)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\Delta y_{Ap} = \rho_{\rm ratio} - (a \cdot Ap + b)$",
                  fontsize=11, fontweight="bold")
    ax.set_title(
        f"{ev['label']}  —  Ap-Detrended Density Ratio Residual vs Time",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(ev["date_start"], ev["date_end"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=ev["x_interval"]))
    ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", fontsize=8.5,
              framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {out_png}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    for ev in EVENTS:
        plot_event(ev)
    print("\nAll 3 events completed!")


if __name__ == "__main__":
    main()
