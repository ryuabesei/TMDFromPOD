"""
ap_removal_method1_linear_detrend.py

手法①: 線形回帰による Ap 依存性除去 (Linear Detrending)

アルゴリズム:
    1. 各イベントの静穏期（ref期間）のデータで
       daily median density_ratio_msis ~ AP_AVG を線形フィット
       (slope a, intercept b)
    2. 全期間に補正を適用:
       rho_detrended = rho_ratio - (a * AP_AVG + b) + 1.0
       → Ap に線形に相関する成分を除去し、1.0 を基準にシフト
    3. 補正前後 + Ap 時系列を 3イベント × 3緯度帯 の 3×3 パネルで可視化

出力:
    Figure/Ap_removal/method1_linear_detrend.png
    Figure/Ap_removal/method1_regression_diagnostics.png
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
OUT_MAIN = OUT_DIR / "method1_linear_detrend.png"
OUT_DIAG = OUT_DIR / "method1_regression_diagnostics.png"

AP_KP3 = 15.0  # Kp=3 に相当する Ap 値

# ─── 緯度帯 ──────────────────────────────────────────────────────────────────
LAT_BANDS = [
    ("High (40-60deg)", 40.0, 60.0),
    ("Mid  (20-40deg)", 20.0, 40.0),
    ("Low  ( 0-20deg)",  0.0, 20.0),
]

# ─── イベント設定 ─────────────────────────────────────────────────────────────
EVENTS = [
    dict(
        label="2018 NH SSW",
        parquet=PARQUET_2018,
        mode="doy",
        doy_start=30, doy_end=65,
        ref_doy=[(30, 40), (61, 65)],
        ssw_peak_doy=43,
        year=2018,
    ),
    dict(
        label="2019 SH SSW",
        parquet=PARQUET_2019,
        mode="date",
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"),
        year=2019,
    ),
    dict(
        label="2021 NH SSW",
        parquet=PARQUET_2021,
        mode="date",
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end=pd.Timestamp("2021-02-05", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29", tz="UTC")),
            (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2021-01-04", tz="UTC"),
        year=2021,
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
    df = df.dropna(subset=["datetime", "lat", "density_ratio_msis", "AP_AVG"])

    if ev["mode"] == "doy":
        df["key"] = df["datetime"].dt.dayofyear
        df = df[(df["key"] >= ev["doy_start"]) & (df["key"] <= ev["doy_end"])]
    else:
        df["key"] = df["datetime"].dt.normalize()
        df = df[(df["datetime"] >= ev["date_start"]) &
                (df["datetime"] <= ev["date_end"] + pd.Timedelta(hours=23, minutes=59))]

    return df


def get_ref_mask(df: pd.DataFrame, ev: dict) -> pd.Series:
    if ev["mode"] == "doy":
        mask = pd.Series(False, index=df.index)
        for lo, hi in ev["ref_doy"]:
            mask |= (df["key"] >= lo) & (df["key"] <= hi)
    else:
        mask = pd.Series(False, index=df.index)
        for s, e in ev["ref_dates"]:
            mask |= (df["key"] >= s.normalize()) & (df["key"] <= e.normalize())
    return mask


# ─── 線形回帰 + 補正 ──────────────────────────────────────────────────────────
def fit_linear_ap(df_ref_band: pd.DataFrame) -> tuple[float, float]:
    """静穏期 daily median で AP_AVG ~ rho_ratio を線形フィット"""
    daily_ref = df_ref_band.groupby("key").agg(
        ratio_med=("density_ratio_msis", "median"),
        ap_mean=("AP_AVG", "mean"),
    ).dropna()

    if len(daily_ref) < 3:
        return 0.0, 1.0

    x = daily_ref["ap_mean"].values
    y = daily_ref["ratio_med"].values
    coeffs = np.polyfit(x, y, deg=1)
    return float(coeffs[0]), float(coeffs[1])


def apply_linear_correction(df: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """rho_detrended = rho_ratio - (a*Ap + b) + 1.0"""
    df = df.copy()
    df["ap_pred"] = a * df["AP_AVG"] + b
    df["rho_detrended"] = df["density_ratio_msis"] - (df["ap_pred"] - 1.0)
    return df


# ─── 日次集計・残差計算 ───────────────────────────────────────────────────────
def daily_band(df: pd.DataFrame, col: str = "density_ratio_msis") -> tuple[pd.Series, pd.Series]:
    daily_ratio = df.groupby("key")[col].median()
    daily_ap = df.groupby("key")["AP_AVG"].mean()
    return daily_ratio, daily_ap


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
            # Handle tz-aware vs tz-naive
            if series.index.tzinfo is not None:
                if dates.tzinfo is None:
                    dates = dates.tz_localize("UTC")
                else:
                    dates = dates.tz_convert("UTC")
            ref_keys.extend(dates)
        mask = series.index.isin(ref_keys)

    if mask.sum() == 0 or series[mask].isna().all():
        return series * np.nan
    ref_val = float(series[mask].median())
    return series - ref_val


# ─── メインプロット (3×3 パネル) ─────────────────────────────────────────────
def plot_main(all_data: list[dict]) -> None:
    n_bands = len(LAT_BANDS)
    n_events = len(EVENTS)

    fig = plt.figure(figsize=(20, 4.5 * n_bands))
    fig.suptitle(
        "Method 1 — Linear Ap Detrending of rho_ratio (SWARM-C)\n"
        "Blue solid: original  |  Orange dashed: Ap-corrected  |  Gray bars: Ap index",
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

            # Ap バー
            if ev_cfg["mode"] == "doy":
                ax2.bar(daily_ap.index, daily_ap.values, width=0.8,
                        color="slategray", alpha=0.30, zorder=1)
            else:
                ax2.bar(daily_ap.index, daily_ap.values,
                        width=pd.Timedelta(days=1),
                        color="slategray", alpha=0.30, zorder=1)
            ax2.axhline(AP_KP3, color="darkgray", lw=1.0, ls=":", zorder=2)
            ax2.set_ylabel("Ap", fontsize=8, color="slategray")
            ax2.tick_params(axis="y", labelcolor="slategray", labelsize=7)
            ap_max = daily_ap.max() if not daily_ap.empty else AP_KP3
            ax2.set_ylim(0, max(ap_max * 2.5, AP_KP3 * 3))

            # 補正前 (青)
            ax.plot(delta_raw.index, delta_raw.values,
                    color="#1f77b4", lw=2.0, zorder=5, label="Original")
            # 補正後 (橙 dashed)
            ax.plot(delta_corr.index, delta_corr.values,
                    color="#ff7f0e", lw=2.0, ls="--", zorder=6, label="Ap-corrected")
            ax.axhline(0, color="black", lw=0.8, ls=":", zorder=3)

            # SSW ピーク
            pk = ev_cfg.get("ssw_peak_doy") or ev_cfg.get("ssw_peak")
            if pk is not None:
                ax.axvline(pk, color="red", lw=1.5, ls="--", alpha=0.8, zorder=7, label="SSW Peak")
                ax.text(pk, 0.95, " SSW Peak", transform=ax.get_xaxis_transform(),
                        fontsize=7.5, color="red", fontweight="bold", alpha=0.9, va="top")

            # Ref 期間シェード
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

            a = ev_data["fit_params"][band_label]["a"]
            b = ev_data["fit_params"][band_label]["b"]
            r2 = ev_data["fit_params"][band_label]["r2"]
            ax.text(0.98, 0.97,
                    f"a={a:.5f}\nb={b:.4f}\nR2={r2:.3f}",
                    transform=ax.transAxes,
                    fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_MAIN, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_MAIN}")


# ─── 回帰診断プロット ─────────────────────────────────────────────────────────
def plot_diagnostics(all_data: list[dict]) -> None:
    n_bands = len(LAT_BANDS)
    n_events = len(EVENTS)
    fig, axes = plt.subplots(n_bands, n_events, figsize=(18, 4.0 * n_bands))
    fig.suptitle(
        "Method 1 — Regression Diagnostics: AP_AVG vs rho_ratio (daily median)",
        fontsize=13, fontweight="bold",
    )

    for row, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
        for col, (ev_data, ev_cfg) in enumerate(zip(all_data, EVENTS)):
            ax = axes[row, col]
            a = ev_data["fit_params"][band_label]["a"]
            b = ev_data["fit_params"][band_label]["b"]
            r2 = ev_data["fit_params"][band_label]["r2"]
            daily_raw = ev_data["daily_ratio_raw"][band_label]
            daily_corr = ev_data["daily_ratio_corr"][band_label]
            daily_ap  = ev_data["daily_ap"][band_label]
            ref_mask_d = ev_data["ref_mask_daily"][band_label]

            ax.scatter(daily_ap.values, daily_raw.values,
                       c="steelblue", s=20, alpha=0.6, zorder=4, label="All days (raw)")
            ref_ap  = daily_ap[ref_mask_d]
            ref_raw = daily_raw[ref_mask_d]
            ax.scatter(ref_ap.values, ref_raw.values,
                       c="red", s=40, zorder=5, label="Ref period")
            ax.scatter(daily_ap.values, daily_corr.values,
                       c="#ff7f0e", s=15, marker="^", alpha=0.6, zorder=4, label="Corrected")

            ap_line = np.linspace(0, max(daily_ap.max() * 1.3, AP_KP3 * 2), 100)
            ax.plot(ap_line, a * ap_line + b, color="red", lw=1.5, ls="--",
                    label=f"y={a:.5f}x+{b:.4f}")
            ax.axhline(1.0, color="gray", lw=0.8, ls=":")
            ax.axvline(AP_KP3, color="darkgray", lw=1.0, ls=":", alpha=0.7)

            ax.set_title(f"{ev_cfg['label']} / {band_label}\nR2={r2:.3f}", fontsize=9)
            ax.set_xlabel("AP_AVG (daily mean)", fontsize=8)
            if col == 0:
                ax.set_ylabel("density_ratio_msis\n(daily median)", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.4)

            if row == 0 and col == n_events - 1:
                ax.legend(fontsize=7, loc="upper left")

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
        ref_mask_row = get_ref_mask(df, ev)
        df_ref = df[ref_mask_row]

        ev_data = dict(
            label=ev["label"],
            daily_ratio_raw={},
            daily_ratio_corr={},
            daily_ap={},
            delta_raw={},
            delta_corr={},
            fit_params={},
            ref_mask_daily={},
        )

        for band_label, lat_lo, lat_hi in LAT_BANDS:
            band_mask = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
            df_band = df[band_mask]
            # df_ref is a subset of df; apply band_mask using reindex-safe approach
            ref_band_mask = (df_ref["lat"].abs() >= lat_lo) & (df_ref["lat"].abs() < lat_hi)
            df_ref_band = df_ref[ref_band_mask]

            a, b = fit_linear_ap(df_ref_band)

            # R2 計算 (全期間で評価)
            daily_all = df_band.groupby("key").agg(
                ratio_med=("density_ratio_msis", "median"),
                ap_mean=("AP_AVG", "mean"),
            ).dropna()
            if len(daily_all) >= 2:
                y_pred = a * daily_all["ap_mean"].values + b
                ss_res = np.sum((daily_all["ratio_med"].values - y_pred) ** 2)
                ss_tot = np.sum((daily_all["ratio_med"].values - daily_all["ratio_med"].mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            else:
                r2 = np.nan

            print(f"  {band_label}: a={a:.6f}, b={b:.5f}, R2={r2:.3f}")

            df_band_corr = apply_linear_correction(df_band, a, b)

            daily_raw, daily_ap = daily_band(df_band)
            daily_corr, _ = daily_band(df_band_corr, col="rho_detrended")

            delta_raw  = compute_delta(daily_raw,  ev)
            delta_corr = compute_delta(daily_corr, ev)

            if ev["mode"] == "doy":
                ref_keys = []
                for lo, hi in ev["ref_doy"]:
                    ref_keys.extend(range(lo, hi + 1))
                ref_mask_d = daily_raw.index.isin(ref_keys)
            else:
                ref_keys = []
                for s, e in ev["ref_dates"]:
                    dates = pd.date_range(s.normalize(), e.normalize(), freq="D")
                    if daily_raw.index.tzinfo is not None:
                        if dates.tzinfo is None:
                            dates = dates.tz_localize("UTC")
                        else:
                            dates = dates.tz_convert("UTC")
                    ref_keys.extend(dates)
                ref_mask_d = daily_raw.index.isin(ref_keys)

            ev_data["daily_ratio_raw"][band_label]  = daily_raw
            ev_data["daily_ratio_corr"][band_label] = daily_corr
            ev_data["daily_ap"][band_label]          = daily_ap
            ev_data["delta_raw"][band_label]         = delta_raw
            ev_data["delta_corr"][band_label]        = delta_corr
            ev_data["fit_params"][band_label]        = {"a": a, "b": b, "r2": r2}
            ev_data["ref_mask_daily"][band_label]    = ref_mask_d

        all_data.append(ev_data)

    print("\nPlotting main figure...")
    plot_main(all_data)
    print("Plotting diagnostics figure...")
    plot_diagnostics(all_data)
    print("\nMethod 1 complete.")


if __name__ == "__main__":
    main()
