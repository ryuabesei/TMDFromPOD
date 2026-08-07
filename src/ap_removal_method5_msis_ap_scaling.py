"""
ap_removal_method5_msis_ap_scaling.py

手法⑤: MSIS Ap ゲイン補正 (MSIS Ap Gain Tuning)

アルゴリズム:
    MSIS が実際の密度変動に対して「何倍の Ap 感度でスケールしているか」を診断し、
    ゲイン係数 k を推定して補正する。

    定式化:
        rho_MSIS_tuned = rho_MSIS_real * (1 + k * (AP_AVG - AP_ref) / AP_ref)
        rho_ratio_tuned = rho_obs / rho_MSIS_tuned

    ゲイン係数 k の推定:
        静穏期の daily median を用いて
        rho_obs / rho_MSIS_real ~ f(AP_AVG) を線形フィット。
        傾き a, 切片 b から:
            k = a * AP_ref / b
        (AP_ref = 静穏期の平均 Ap を使用)

    解釈:
        k > 0: MSIS が実際より地磁気応答を過小評価している
        k < 0: MSIS が実際より地磁気応答を過大評価している

出力:
    Figure/Ap_removal/method5_msis_ap_scaling.png
    Figure/Ap_removal/method5_gain_diagnostics.png
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

OUT_DIR  = Path("Figure/Ap_removal")
OUT_MAIN = OUT_DIR / "method5_msis_ap_scaling.png"
OUT_DIAG = OUT_DIR / "method5_gain_diagnostics.png"

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
    df = df.dropna(subset=["datetime", "lat", "density_ratio_msis", "AP_AVG",
                            "rho_model_real", "density"])

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


# ─── ゲイン係数推定 ───────────────────────────────────────────────────────────
def estimate_gain(df_ref_band: pd.DataFrame) -> dict:
    """
    静穏期 daily median で AP_AVG ~ rho_ratio を線形フィット
    → MSIS の Ap 感度ゲイン補正係数 k を推定
    """
    daily = df_ref_band.groupby("key").agg(
        ratio=("density_ratio_msis", "median"),
        ap=("AP_AVG", "mean"),
    ).dropna()

    if len(daily) < 3:
        return {"a": 0.0, "b": 1.0, "k": 0.0, "ap_ref": 5.0, "r2": np.nan}

    x = daily["ap"].values
    y = daily["ratio"].values
    p = np.polyfit(x, y, 1)
    a, b = p[0], p[1]
    ap_ref = float(daily["ap"].mean())

    # k: rho_ratio - 1 = a * (Ap - ap_ref) / b
    # よって MSIS に必要な追加ゲイン k = a * ap_ref / b
    k = a * ap_ref / b if abs(b) > 1e-10 else 0.0

    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"a": a, "b": b, "k": k, "ap_ref": ap_ref, "r2": r2}


def apply_gain_correction(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """
    rho_MSIS_tuned = rho_model_real * (1 + k * (AP_AVG - ap_ref) / ap_ref)
    rho_ratio_tuned = density / rho_MSIS_tuned
    """
    df = df.copy()
    k     = fit["k"]
    ap_ref = fit["ap_ref"]
    scale  = 1.0 + k * (df["AP_AVG"] - ap_ref) / ap_ref
    scale  = scale.clip(lower=0.1)  # 極端な値を防止
    df["rho_model_tuned"] = df["rho_model_real"] * scale
    df["rho_ratio_tuned"] = df["density"] / df["rho_model_tuned"]
    return df


# ─── 残差計算 ─────────────────────────────────────────────────────────────────
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


# ─── メインプロット ───────────────────────────────────────────────────────────
def plot_main(all_data: list[dict]) -> None:
    n_bands = len(LAT_BANDS)
    n_events = len(EVENTS)
    fig = plt.figure(figsize=(20, 4.5 * n_bands))
    fig.suptitle(
        "Method 5 — MSIS Ap Gain Tuning (SWARM-C)\n"
        "Blue solid: original ratio  |  Teal dashed: gain-corrected ratio  |  Gray bars: Ap",
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
                    color="#17becf", lw=2.0, ls="--", zorder=6, label="Gain-corrected")
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
            ax.text(0.98, 0.97,
                    f"k={fp['k']:.4f}\nAp_ref={fp['ap_ref']:.1f}\nR2={fp['r2']:.3f}",
                    transform=ax.transAxes,
                    fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_MAIN, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_MAIN}")


def plot_diagnostics(all_data: list[dict]) -> None:
    """ゲイン係数 k の棒グラフ + R² テーブル"""
    n_events = len(EVENTS)
    n_bands  = len(LAT_BANDS)
    band_labels = [b[0] for b in LAT_BANDS]

    fig, axes = plt.subplots(2, n_events, figsize=(6 * n_events, 8))
    fig.suptitle("Method 5 — Gain Factor k and R² by Event and Latitude Band",
                 fontsize=13, fontweight="bold")

    x = np.arange(n_bands)
    width = 0.55

    for col, (ev_data, ev_cfg) in enumerate(zip(all_data, EVENTS)):
        # 上段: k
        ax_k = axes[0, col]
        k_vals  = [ev_data["fit_params"][bl]["k"]   for bl in band_labels]
        r2_vals = [ev_data["fit_params"][bl]["r2"]   for bl in band_labels]
        ar_vals = [ev_data["fit_params"][bl]["ap_ref"] for bl in band_labels]

        colors_k = ["#1a9641" if v > 0 else "#d7191c" for v in k_vals]
        bars = ax_k.bar(x, k_vals, width, color=colors_k, alpha=0.75)
        ax_k.axhline(0, color="black", lw=0.8)
        ax_k.set_title(ev_cfg["label"], fontsize=11, fontweight="bold")
        ax_k.set_xticks(x)
        ax_k.set_xticklabels(band_labels, rotation=15, ha="right", fontsize=8)
        ax_k.set_ylabel("Gain factor k", fontsize=9)
        ax_k.grid(True, ls=":", alpha=0.4, axis="y")
        for bar, v in zip(bars, k_vals):
            ax_k.text(bar.get_x() + bar.get_width()/2,
                      v + (0.001 if v >= 0 else -0.004),
                      f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        # 下段: R²
        ax_r = axes[1, col]
        bars2 = ax_r.bar(x, r2_vals, width, color="steelblue", alpha=0.75)
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(band_labels, rotation=15, ha="right", fontsize=8)
        ax_r.set_ylabel("R² (ref period fit)", fontsize=9)
        ax_r.set_ylim(-0.5, 1.0)
        ax_r.axhline(0, color="black", lw=0.8)
        ax_r.grid(True, ls=":", alpha=0.4, axis="y")
        for bar, v, ap_r in zip(bars2, r2_vals, ar_vals):
            ax_r.text(bar.get_x() + bar.get_width()/2,
                      max(v, 0) + 0.02,
                      f"R2={v:.3f}\nAp_ref={ap_r:.1f}", ha="center", va="bottom", fontsize=7)

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
            delta_raw={},
            delta_corr={},
            daily_ap={},
            fit_params={},
        )

        for band_label, lat_lo, lat_hi in LAT_BANDS:
            band_mask   = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
            df_band     = df[band_mask]
            ref_mask_b  = (df_ref["lat"].abs() >= lat_lo) & (df_ref["lat"].abs() < lat_hi)
            df_ref_band = df_ref[ref_mask_b]

            fit = estimate_gain(df_ref_band)
            print(f"  {band_label}: k={fit['k']:.5f}, ap_ref={fit['ap_ref']:.1f}, "
                  f"a={fit['a']:.6f}, b={fit['b']:.5f}, R2={fit['r2']:.3f}")

            df_band_corr = apply_gain_correction(df_band, fit)

            daily_raw  = df_band.groupby("key")["density_ratio_msis"].median()
            daily_corr = df_band_corr.groupby("key")["rho_ratio_tuned"].median()
            daily_ap   = df_band.groupby("key")["AP_AVG"].mean()

            delta_raw  = compute_delta(daily_raw,  ev)
            delta_corr = compute_delta(daily_corr, ev)

            ev_data["delta_raw"][band_label]  = delta_raw
            ev_data["delta_corr"][band_label] = delta_corr
            ev_data["daily_ap"][band_label]   = daily_ap
            ev_data["fit_params"][band_label] = fit

        all_data.append(ev_data)

    print("\nPlotting main figure...")
    plot_main(all_data)
    print("Plotting diagnostics figure...")
    plot_diagnostics(all_data)
    print("\nMethod 5 complete.")


if __name__ == "__main__":
    main()
