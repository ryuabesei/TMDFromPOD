"""
plot_ref_medians_comparison_2021.py

Purpose:
    Compute and plot the median of density_ratio_msis during the non-SSW
    reference periods for 2021 NH SSW (SWARM-C and GRACE-FO).

    Non-SSW reference:
        REF1: 2020-12-25 to 2020-12-29
        REF2: 2021-02-01 to 2021-02-05

    LT sectors:
        SWARM-C:  Fixed windows — Dawn (LT 4.5-10.5h) / Dusk (LT 16.5-22.5h)
        GRACE-FO: Dynamic orbital-plane split (Plane A lower-LT / Plane B higher-LT)
                  using circular-distance continuity tracking across 0/24 h boundary.

Output:
    Figure/2021/reference_medians_comparison_2021.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Settings
# ============================================================
DATE_START = pd.Timestamp("2020-12-25", tz="UTC")
DATE_END   = pd.Timestamp("2021-02-05", tz="UTC")

DATE_REF1_START = pd.Timestamp("2020-12-25", tz="UTC")
DATE_REF1_END   = pd.Timestamp("2020-12-29", tz="UTC")
DATE_REF2_START = pd.Timestamp("2021-02-01", tz="UTC")
DATE_REF2_END   = pd.Timestamp("2021-02-05", tz="UTC")

LAT_BANDS = [
    ("High\n(40-60°)", 40.0, 60.0),
    ("Mid\n(20-40°)",  20.0, 40.0),
    ("Low\n(0-20°)",    0.0, 20.0),
]

VALUE_COL = "density_ratio_msis"


# ============================================================
# Helpers
# ============================================================
def get_ref_median(daily: pd.Series) -> float:
    idx = daily.index
    mask = (
        ((idx >= DATE_REF1_START) & (idx <= DATE_REF1_END)) |
        ((idx >= DATE_REF2_START) & (idx <= DATE_REF2_END))
    )
    if mask.sum() == 0 or daily[mask].isna().all():
        return float("nan")
    return float(daily[mask].median())


def circ_dist(a: float, b: float, period: float = 24.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def assign_orbital_plane(df: pd.DataFrame, lat_col: str) -> pd.DataFrame:
    df = df.copy()
    df["orbital_plane"] = "A"
    for date, grp in df.groupby("date"):
        p25 = grp["lst_h"].quantile(0.25)
        p75 = grp["lst_h"].quantile(0.75)
        midpoint = (p25 + p75) / 2.0
        if abs(p75 - p25) < 3.0:
            midpoint = 12.0
        df.loc[grp.index[grp["lst_h"] >= midpoint], "orbital_plane"] = "B"

    dates_sorted = sorted(df["date"].unique())
    init_a = df[df["orbital_plane"] == "A"].groupby("date")["lst_h"].median()
    init_b = df[df["orbital_plane"] == "B"].groupby("date")["lst_h"].median()
    corrected_a_prev = init_a.get(dates_sorted[0], np.nan)

    for i in range(1, len(dates_sorted)):
        curr_date = dates_sorted[i]
        curr_a0 = init_a.get(curr_date, np.nan)
        curr_b0 = init_b.get(curr_date, np.nan)
        if np.isnan(corrected_a_prev) or np.isnan(curr_a0) or np.isnan(curr_b0):
            corrected_a_prev = curr_a0
            continue
        if circ_dist(corrected_a_prev, curr_b0) < circ_dist(corrected_a_prev, curr_a0):
            mask = df["date"] == curr_date
            df.loc[mask & (df["orbital_plane"] == "A"), "orbital_plane"] = "_tmp"
            df.loc[mask & (df["orbital_plane"] == "B"), "orbital_plane"] = "A"
            df.loc[mask & (df["orbital_plane"] == "_tmp"), "orbital_plane"] = "B"
            corrected_a_prev = curr_b0
        else:
            corrected_a_prev = curr_a0
    return df


# ============================================================
# Compute ref medians
# ============================================================
def compute_swarm_c() -> dict[str, list[float]]:
    fp = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")
    df = pd.read_parquet(fp)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next(c for c in ["lat", "latitude", "geod_lat"] if c in df.columns)
    df = df.dropna(subset=["datetime", lat_col, "lst_h", VALUE_COL])
    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] <= DATE_END + pd.Timedelta(hours=23, minutes=59))]
    df["date"] = df["datetime"].dt.normalize()

    sectors = {
        "Dawn (LT 4.5-10.5h)":  df[(df["lst_h"] >= 4.5)  & (df["lst_h"] < 10.5)],
        "Dusk (LT 16.5-22.5h)": df[(df["lst_h"] >= 16.5) & (df["lst_h"] < 22.5)],
    }
    result = {}
    for lt_label, df_lt in sectors.items():
        medians = []
        for _, lat_lo, lat_hi in LAT_BANDS:
            mask = (df_lt[lat_col].abs() >= lat_lo) & (df_lt[lat_col].abs() < lat_hi)
            daily = df_lt[mask].groupby("date")[VALUE_COL].median()
            medians.append(get_ref_median(daily))
        result[lt_label] = medians
        print(f"  SWARM-C {lt_label}: {[f'{v:.3f}' for v in medians]}")
    return result


def compute_grace_fo() -> dict[str, list[float]]:
    fp = Path("normalizeddata/2021/grace_fo_dns_2021_normalized_with_LT_removed.parquet")
    df = pd.read_parquet(fp)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    lat_col = next(c for c in ["lat", "latitude", "geod_lat"] if c in df.columns)
    df = df.dropna(subset=["datetime", lat_col, "lst_h", VALUE_COL])
    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] <= DATE_END + pd.Timedelta(hours=23, minutes=59))]
    df["date"] = df["datetime"].dt.normalize()

    print("  [GRACE-FO] Assigning orbital planes ...")
    df = assign_orbital_plane(df, lat_col)

    lt_a_med = df[df["orbital_plane"] == "A"].groupby("date")["lst_h"].median()
    lt_b_med = df[df["orbital_plane"] == "B"].groupby("date")["lst_h"].median()
    label_a = f"Plane A (LT {lt_a_med.iloc[0]:.1f}h\u2192{lt_a_med.iloc[-1]:.1f}h)"
    label_b = f"Plane B (LT {lt_b_med.iloc[0]:.1f}h\u2192{lt_b_med.iloc[-1]:.1f}h)"

    result = {}
    for plane, label in [("A", label_a), ("B", label_b)]:
        df_lt = df[df["orbital_plane"] == plane]
        medians = []
        for _, lat_lo, lat_hi in LAT_BANDS:
            mask = (df_lt[lat_col].abs() >= lat_lo) & (df_lt[lat_col].abs() < lat_hi)
            daily = df_lt[mask].groupby("date")[VALUE_COL].median()
            medians.append(get_ref_median(daily))
        result[label] = medians
        print(f"  GRACE-FO {label}: {[f'{v:.3f}' for v in medians]}")
    return result


# ============================================================
# Plot
# ============================================================
def make_bar_chart(sat_results: dict[str, dict[str, list[float]]]) -> None:
    categories = [b[0] for b in LAT_BANDS]
    x = np.arange(len(categories))
    width = 0.35

    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"

    n_sats = len(sat_results)
    fig, axes = plt.subplots(1, n_sats, figsize=(7 * n_sats, 6), sharey=True)
    if n_sats == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    colors = ["#1a6faf", "#e07b39"]

    for ax, (sat_name, lt_data) in zip(axes, sat_results.items()):
        lt_keys = list(lt_data.keys())
        lt1, lt2 = lt_keys[0], lt_keys[1]

        rects1 = ax.bar(x - width / 2, lt_data[lt1], width, label=lt1,
                        color=colors[0], alpha=0.9, edgecolor="none")
        rects2 = ax.bar(x + width / 2, lt_data[lt2], width, label=lt2,
                        color=colors[1], alpha=0.9, edgecolor="none")

        def autolabel(rects, ax=ax):
            for rect in rects:
                h = rect.get_height()
                if not np.isnan(h):
                    ax.annotate(f"{h:.3f}",
                                xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=10, fontweight="bold")

        autolabel(rects1)
        autolabel(rects2)

        ax.set_title(sat_name, fontsize=15, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.legend(loc="lower center", fontsize=10, framealpha=0.9,
                  bbox_to_anchor=(0.5, -0.25), ncol=1)

    axes[0].set_ylabel("Reference Median of \u03c1_ratio\n(rho_obs / rho_MSIS)", fontsize=13, fontweight="bold")

    ref_str = "Non-SSW ref: 2020-12-25\u201312-29  &  2021-02-01\u201302-05"
    plt.suptitle(f"2021 non-SSW Reference Medians (NH SSW)\n{ref_str}",
                 fontsize=16, fontweight="bold", y=1.03)

    out_png = Path("Figure/2021/reference_medians_comparison_2021.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_png}")
    plt.close()


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=== SWARM-C ===")
    swarm_c_data = compute_swarm_c()
    print("\n=== GRACE-FO ===")
    grace_fo_data = compute_grace_fo()
    sat_results = {
        "SWARM-C":  swarm_c_data,
        "GRACE-FO": grace_fo_data,
    }
    make_bar_chart(sat_results)
    print("Done.")


if __name__ == "__main__":
    main()
