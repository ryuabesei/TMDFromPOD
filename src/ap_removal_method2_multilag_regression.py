"""
ap_removal_method2_multilag_regression.py

手法②: タイムラグ付き多変量回帰 (Multi-lag Regression)

アルゴリズム:
    1. 説明変数:
       - AP_AVG      (当日平均 Ap)
       - AP_AVG_prev (前日平均 Ap)
       - AP_prev2    (前々日平均 Ap = AP1_prev2...AP8_prev2 の平均)
    2. 静穏期（ref 期間）の daily median で OLS 回帰
       rho_ratio ~ a0 + a1*Ap(t) + a2*Ap(t-1) + a3*Ap(t-2)
    3. 補正:
       rho_corr = rho_ratio - (predicted - intercept)
       → Ap 由来の予測成分を差し引き、intercept 水準に揃える
    4. R² Before / After を比較パネルで表示

出力:
    Figure/Ap_removal/method2_multilag_regression.png
    Figure/Ap_removal/method2_regression_diagnostics.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# ─── Paths ───────────────────────────────────────────────────────────────────
PARQUET_2018 = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet")
PARQUET_2019 = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet")
PARQUET_2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")

OUT_DIR = Path("Figure/Ap_removal")
OUT_MAIN = OUT_DIR / "method2_multilag_regression.png"
OUT_DIAG = OUT_DIR / "method2_regression_diagnostics.png"

AP_KP3 = 15.0

LAT_BANDS = [
    ("High (40-60deg)", 40.0, 60.0),
    ("Mid  (20-40deg)", 20.0, 40.0),
    ("Low  ( 0-20deg)",  0.0, 20.0),
]

EVENTS = [
    dict(
        label="2018 NH SSW", parquet=PARQUET_2018, mode="doy",
        doy_start=30, doy_end=65,
        ref_doy=[(30, 40), (61, 65)],
        ssw_peak_doy=43, year=2018,
    ),
    dict(
        label="2019 SH SSW", parquet=PARQUET_2019, mode="date",
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"), year=2019,
    ),
    dict(
        label="2021 NH SSW", parquet=PARQUET_2021, mode="date",
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end=pd.Timestamp("2021-02-05", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29", tz="UTC")),
            (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2021-01-04", tz="UTC"), year=2021,
    ),
]


# ─── データロード ─────────────────────────────────────────────────────────────
def load_event_df(ev: dict) -> pd.DataFrame:
    df = pd.read_parquet(ev["parquet"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for cname in ["lat", "latitude", "geod_lat"]:
        if cname in df.columns and cname != "lat":
            df = df.rename(columns={cname: "lat"})
            break
    # 前々日 Ap 平均を計算 (AP1_prev2 ~ AP8_prev2)
    prev2_cols = [c for c in df.columns if "prev2" in c and c.startswith("AP")]
    if prev2_cols:
        df["AP_AVG_prev2"] = df[prev2_cols].mean(axis=1)
    else:
        df["AP_AVG_prev2"] = np.nan

    required = ["density_ratio_msis", "AP_AVG", "AP_AVG_prev"]
    df = df.dropna(subset=["datetime", "lat"] + required)

    if ev["mode"] == "doy":
        df["key"] = df["datetime"].dt.dayofyear
        df = df[(df["key"] >= ev["doy_start"]) & (df["key"] <= ev["doy_end"])]
    else:
        df["key"] = df["datetime"].dt.normalize()
        df = df[(df["datetime"] >= ev["date_start"]) &
                (df["datetime"] <= ev["date_end"] + pd.Timedelta(hours=23, minutes=59))]
    return df


def get_ref_mask_on_df(df: pd.DataFrame, ev: dict) -> pd.Series:
    if ev["mode"] == "doy":
        mask = pd.Series(False, index=df.index)
        for lo, hi in ev["ref_doy"]:
            mask |= (df["key"] >= lo) & (df["key"] <= hi)
    else:
        mask = pd.Series(False, index=df.index)
        for s, e in ev["ref_dates"]:
            mask |= (df["key"] >= s.normalize()) & (df["key"] <= e.normalize())
    return mask


# ─── 多変量 OLS 回帰 ──────────────────────────────────────────────────────────
def fit_multilag(df_ref_band: pd.DataFrame) -> dict:
    """
    daily median 集計 -> OLS で [intercept, a1, a2, a3] を推定。
    features: AP_AVG, AP_AVG_prev, AP_AVG_prev2
    """
    daily = df_ref_band.groupby("key").agg(
        ratio_med=("density_ratio_msis", "median"),
        ap0=("AP_AVG", "mean"),
        ap1=("AP_AVG_prev", "mean"),
        ap2=("AP_AVG_prev2", "mean"),
    ).dropna()

    if len(daily) < 4:
        return {"coeffs": np.array([1.0, 0.0, 0.0, 0.0]), "r2_ref": np.nan, "feature_names": ["intercept", "AP_t", "AP_t-1", "AP_t-2"]}

    X = np.column_stack([np.ones(len(daily)), daily["ap0"].values,
                         daily["ap1"].values, daily["ap2"].values])
    y = daily["ratio_med"].values
    result = np.linalg.lstsq(X, y, rcond=None)
    coeffs = result[0]

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"coeffs": coeffs, "r2_ref": r2,
            "feature_names": ["intercept", "AP_t", "AP_t-1", "AP_t-2"]}


def predict_multilag(df: pd.DataFrame, fit: dict) -> pd.Series:
    """全データに対して回帰予測値を計算する (行ごと)"""
    c = fit["coeffs"]
    pred = c[0] + c[1]*df["AP_AVG"] + c[2]*df["AP_AVG_prev"] + c[3]*df["AP_AVG_prev2"].fillna(0)
    return pred


def apply_multilag_correction(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """rho_corr = rho_ratio - (pred - intercept)"""
    df = df.copy()
    pred = predict_multilag(df, fit)
    intercept = fit["coeffs"][0]
    df["rho_multilag_corr"] = df["density_ratio_msis"] - (pred - intercept)
    df["ap_pred_multilag"] = pred
    return df


# ─── daily 集計 + 残差 ───────────────────────────────────────────────────────
def daily_stats(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("key")[col].median()


def compute_delta(series: pd.Series, ev: dict) -> pd.Series:
    if ev["mode"] == "doy":
        ref_keys = []
        for lo, hi in ev["ref_doy"]:
            ref_keys.extend(range(lo, hi + 1))
        mask = series.index.isin(ref_keys)
    else:
        ref_keys = []
        for s, e in ev["ref_dates"]:
            dates = pd.date_range(s.normalize(), e.normalize(), freq="D")
            if series.index.tzinfo is not None:
                dates = dates.tz_localize("UTC") if dates.tzinfo is None else dates.tz_convert("UTC")
            ref_keys.extend(dates)
        mask = series.index.isin(ref_keys)
    if mask.sum() == 0 or series[mask].isna().all():
        return series * np.nan
    return series - float(series[mask].median())


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# ─── メインプロット ───────────────────────────────────────────────────────────
def plot_main(all_data: list[dict]) -> None:
    n_bands = len(LAT_BANDS)
    n_events = len(EVENTS)
    fig = plt.figure(figsize=(20, 4.5 * n_bands))
    fig.suptitle(
        "Method 2 — Multi-lag Ap Regression Correction of rho_ratio (SWARM-C)\n"
        "Blue: original  |  Green dashed: multi-lag corrected  |  Gray bars: Ap",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(n_bands, n_events, figure=fig, hspace=0.35, wspace=0.20)

    for row, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
        for col, (ev_data, ev_cfg) in enumerate(zip(all_data, EVENTS)):
            ax = fig.add_subplot(gs[row, col])
            ax2 = ax.twinx()

            delta_raw  = ev_data["delta_raw"][band_label]
            delta_corr = ev_data["delta_corr"][band_label]
            daily_ap   = ev_data["daily_ap"][band_label]

            if ev_cfg["mode"] == "doy":
                ax2.bar(daily_ap.index, daily_ap.values, width=0.8,
                        color="slategray", alpha=0.30, zorder=1)
            else:
                ax2.bar(daily_ap.index, daily_ap.values, width=pd.Timedelta(days=1),
                        color="slategray", alpha=0.30, zorder=1)
            ax2.axhline(AP_KP3, color="darkgray", lw=1.0, ls=":", zorder=2)
            ax2.set_ylabel("Ap", fontsize=8, color="slategray")
            ax2.tick_params(axis="y", labelcolor="slategray", labelsize=7)
            ap_max = daily_ap.max() if not daily_ap.empty else AP_KP3
            ax2.set_ylim(0, max(ap_max * 2.5, AP_KP3 * 3))

            ax.plot(delta_raw.index, delta_raw.values,
                    color="#1f77b4", lw=2.0, zorder=5, label="Original")
            ax.plot(delta_corr.index, delta_corr.values,
                    color="#2ca02c", lw=2.0, ls="--", zorder=6, label="Multi-lag corrected")
            ax.axhline(0, color="black", lw=0.8, ls=":", zorder=3)

            pk = ev_cfg.get("ssw_peak_doy") or ev_cfg.get("ssw_peak")
            if pk is not None:
                ax.axvline(pk, color="red", lw=1.5, ls="--", alpha=0.8, zorder=7, label="SSW Peak")
                ax.text(pk, 0.95, " SSW Peak", transform=ax.get_xaxis_transform(),
                        fontsize=7.5, color="red", fontweight="bold", alpha=0.9, va="top")

            if ev_cfg["mode"] == "doy":
                for lo, hi in ev_cfg["ref_doy"]:
                    ax.axvspan(lo, hi, color="lightblue", alpha=0.25, lw=0)
            else:
                for s, e in ev_cfg["ref_dates"]:
                    ax.axvspan(s, e, color="lightblue", alpha=0.25, lw=0)

            if row == 0:
                ax.set_title(ev_cfg["label"], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{band_label}\nDelta Ratio", fontsize=9)
            if row == n_bands - 1:
                if ev_cfg["mode"] == "date":
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
                ax.set_xlabel(f"{'DOY' if ev_cfg['mode']=='doy' else 'Date'} ({ev_cfg['year']})", fontsize=9)

            ax.grid(True, linestyle=":", alpha=0.5)

            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

            fp = ev_data["fit_params"][band_label]
            c = fp["coeffs"]
            r2b = fp["r2_before"]
            r2a = fp["r2_after"]
            ax.text(0.98, 0.97,
                    f"a0={c[0]:.4f}, a1={c[1]:.5f}\na2={c[2]:.5f}, a3={c[3]:.5f}\n"
                    f"R2_before={r2b:.3f}  R2_after={r2a:.3f}",
                    transform=ax.transAxes,
                    fontsize=6.5, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_MAIN, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_MAIN}")


def plot_diagnostics(all_data: list[dict]) -> None:
    """R² before / after の棒グラフ比較"""
    n_events = len(EVENTS)
    n_bands = len(LAT_BANDS)
    fig, axes = plt.subplots(1, n_events, figsize=(5 * n_events, 5))
    fig.suptitle("Method 2 — R² Improvement: Before vs After Multi-lag Correction",
                 fontsize=13, fontweight="bold")

    x = np.arange(n_bands)
    width = 0.35
    band_labels = [b[0] for b in LAT_BANDS]

    for col, (ev_data, ev_cfg) in enumerate(zip(all_data, EVENTS)):
        ax = axes[col]
        r2_before = [ev_data["fit_params"][bl]["r2_before"] for bl in band_labels]
        r2_after  = [ev_data["fit_params"][bl]["r2_after"]  for bl in band_labels]

        bars1 = ax.bar(x - width/2, r2_before, width, label="Before correction", color="#1f77b4", alpha=0.75)
        bars2 = ax.bar(x + width/2, r2_after,  width, label="After correction",  color="#2ca02c", alpha=0.75)

        ax.set_title(ev_cfg["label"], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("R² (rho vs AP_AVG)", fontsize=9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.4, axis="y")
        ax.legend(fontsize=8)

        # 値ラベル
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02 if h >= 0 else h - 0.07,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02 if h >= 0 else h - 0.07,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, color="#2ca02c")

    plt.tight_layout()
    plt.savefig(OUT_DIAG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIAG}")


# ─── メイン ──────────────────────────────────────────────────────────────────
def main() -> None:
    all_data = []

    for ev in EVENTS:
        print(f"\n{'='*60}")
        print(f"Processing: {ev['label']}")
        df = load_event_df(ev)
        ref_mask_row = get_ref_mask_on_df(df, ev)
        df_ref = df[ref_mask_row]

        ev_data = dict(
            label=ev["label"],
            daily_ratio_raw={},
            daily_ratio_corr={},
            daily_ap={},
            delta_raw={},
            delta_corr={},
            fit_params={},
        )

        for band_label, lat_lo, lat_hi in LAT_BANDS:
            band_mask  = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
            df_band    = df[band_mask]
            ref_mask_b = (df_ref["lat"].abs() >= lat_lo) & (df_ref["lat"].abs() < lat_hi)
            df_ref_band = df_ref[ref_mask_b]

            fit = fit_multilag(df_ref_band)
            df_band_corr = apply_multilag_correction(df_band, fit)

            daily_raw  = daily_stats(df_band, "density_ratio_msis")
            daily_corr = daily_stats(df_band_corr, "rho_multilag_corr")
            daily_ap   = daily_stats(df_band, "AP_AVG")

            # R² before correction: rho_ratio vs AP_AVG
            daily_all = df_band.groupby("key").agg(
                r=("density_ratio_msis", "median"),
                ap=("AP_AVG", "mean"),
            ).dropna()
            if len(daily_all) >= 2:
                slope_b = np.polyfit(daily_all["ap"].values, daily_all["r"].values, 1)
                r2_before = calc_r2(daily_all["r"].values,
                                    np.polyval(slope_b, daily_all["ap"].values))
            else:
                r2_before = np.nan

            # R² after: rho_corr vs AP_AVG
            daily_corr_all = df_band_corr.groupby("key").agg(
                r=("rho_multilag_corr", "median"),
                ap=("AP_AVG", "mean"),
            ).dropna()
            if len(daily_corr_all) >= 2:
                slope_a = np.polyfit(daily_corr_all["ap"].values, daily_corr_all["r"].values, 1)
                r2_after = calc_r2(daily_corr_all["r"].values,
                                   np.polyval(slope_a, daily_corr_all["ap"].values))
            else:
                r2_after = np.nan

            print(f"  {band_label}: R2 before={r2_before:.3f}, after={r2_after:.3f}")
            print(f"    coeffs: a0={fit['coeffs'][0]:.5f}, a1={fit['coeffs'][1]:.5f}, "
                  f"a2={fit['coeffs'][2]:.5f}, a3={fit['coeffs'][3]:.5f}")

            delta_raw  = compute_delta(daily_raw,  ev)
            delta_corr = compute_delta(daily_corr, ev)

            ev_data["daily_ratio_raw"][band_label]  = daily_raw
            ev_data["daily_ratio_corr"][band_label] = daily_corr
            ev_data["daily_ap"][band_label]          = daily_ap
            ev_data["delta_raw"][band_label]         = delta_raw
            ev_data["delta_corr"][band_label]        = delta_corr
            ev_data["fit_params"][band_label]        = {
                "coeffs": fit["coeffs"], "r2_before": r2_before, "r2_after": r2_after,
            }

        all_data.append(ev_data)

    print("\nPlotting main figure...")
    plot_main(all_data)
    print("Plotting diagnostics figure...")
    plot_diagnostics(all_data)
    print("\nMethod 2 complete.")


if __name__ == "__main__":
    main()
