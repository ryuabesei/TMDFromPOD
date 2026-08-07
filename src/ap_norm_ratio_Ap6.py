"""
ap_norm_ratio_Ap6.py

Purpose:
    既存の msis_activity_correlation_20xx.png で示された
    ratio vs Ap の回帰直線を再現し、Ap=6 基準で正規化した
    density ratio を 2018 / 2019 / 2021 の 3 イベントについてプロット。

正規化の考え方:
    回帰直線: ratio_pred = slope * Ap + intercept
    Ap=6 での予測値: ratio_at6 = slope * 6 + intercept
    正規化後の ratio:
        ratio_norm = ratio - slope * (Ap - 6)
        = ratio - (ratio_pred - ratio_at6)
    → どの日の ratio も「もし Ap が 6 だったら」という水準に補正

データ:
    2018 / 2019: SWARM-A (debug 図と同じ衛星)
    2021       : SWARM-C (debug 図と同じ衛星)

出力:
    Figure/Ap_removal/ap_norm_Ap6_3years.png
    Figure/Ap_removal/ap_norm_Ap6_scatter_3years.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# ─── Paths ───────────────────────────────────────────────────────────────────
P2018 = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet")
P2019 = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet")
P2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")

OUT_DIR = Path("Figure/Ap_removal")

AP_REF = 6.0   # 正規化基準 Ap 値
AP_KP3 = 15.0  # Kp=3

# ─── イベント設定 ─────────────────────────────────────────────────────────────
EVENTS = [
    dict(
        label="2018 NH SSW",
        sat="SWARM-A",
        parquet=P2018,
        mode="doy",
        doy_start=30, doy_end=65,
        ref_doy=[(30, 40), (61, 65)],
        ssw_peak_doy=43,
        year=2018,
        color_raw="#1f77b4",
        color_norm="#ff7f0e",
    ),
    dict(
        label="2019 SH SSW",
        sat="SWARM-A",
        parquet=P2019,
        mode="date",
        date_start=pd.Timestamp("2019-08-20", tz="UTC"),
        date_end=pd.Timestamp("2019-09-23", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2019-08-20", tz="UTC"), pd.Timestamp("2019-08-26", tz="UTC")),
            (pd.Timestamp("2019-09-20", tz="UTC"), pd.Timestamp("2019-09-23", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2019-09-19", tz="UTC"),
        year=2019,
        color_raw="#1f77b4",
        color_norm="#ff7f0e",
    ),
    dict(
        label="2021 NH SSW",
        sat="SWARM-C",
        parquet=P2021,
        mode="date",
        date_start=pd.Timestamp("2020-12-25", tz="UTC"),
        date_end=pd.Timestamp("2021-02-05", tz="UTC"),
        ref_dates=[
            (pd.Timestamp("2020-12-25", tz="UTC"), pd.Timestamp("2020-12-29", tz="UTC")),
            (pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-05", tz="UTC")),
        ],
        ssw_peak=pd.Timestamp("2021-01-04", tz="UTC"),
        year=2021,
        color_raw="#1f77b4",
        color_norm="#ff7f0e",
    ),
]


# ─── データロード ─────────────────────────────────────────────────────────────
def load_event_df(ev: dict) -> pd.DataFrame:
    df = pd.read_parquet(ev["parquet"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    # latitude 列名の統一
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


# ─── 全緯度での daily 集計 ─────────────────────────────────────────────────────
def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    """全緯度の daily median ratio + daily mean Ap"""
    daily = df.groupby("key").agg(
        ratio=("density_ratio_msis", "median"),
        ap=("AP_AVG", "mean"),
    ).reset_index()
    return daily


# ─── 回帰フィット (全期間) ────────────────────────────────────────────────────
def fit_regression(daily: pd.DataFrame) -> tuple[float, float, float]:
    """
    全期間の daily median で ratio ~ AP_AVG を線形フィット (debug 図と同一)
    Returns: (slope, intercept, r)
    """
    x = daily["ap"].values
    y = daily["ratio"].values
    p = np.polyfit(x, y, 1)
    slope, intercept = p[0], p[1]
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, intercept, r


# ─── Ap 線形回帰残差 Delta y_Ap = y_i - y_pred(Ap_i) ──────────────────────────
def calc_ap_residual(daily: pd.DataFrame, slope: float, intercept: float) -> pd.Series:
    """
    Delta y_Ap = y_i - y_pred(Ap_i) = y_i - (slope * Ap_i + intercept)
    """
    return daily["ratio"] - (slope * daily["ap"] + intercept)


# ─── ref 期間中央値を引いて delta を計算 ──────────────────────────────────────
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


# ─── メインプロット: 時系列 3 列 (3×2 パネル) ─────────────────────────────────
def plot_timeseries(all_data: list[dict]) -> None:
    """
    各イベント 1 列、各列に 2 段:
      上段: ratio_raw (青) vs Delta y_Ap = y_i - y_pred(Ap_i) (橙) の時系列
      下段: Ap index バー
    合計 2 行 × 3 列
    """
    fig, axes = plt.subplots(
        2, 3, figsize=(21, 7),
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.08, "wspace": 0.20},
    )
    fig.suptitle(
        r"Ap-Detrended Density Ratio Residual ($\Delta y_{Ap} = y_i - y_{pred}(Ap_i)$)  —  SWARM-A 2018/2019, SWARM-C 2021" "\n"
        r"Blue solid: original ratio ($y_i$)  |  Orange dashed: $\Delta y_{Ap} = y_i - (a \cdot Ap_i + b)$  |  Gray: Ap index",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for col, ev_data in enumerate(all_data):
        ev  = ev_data["ev"]
        ax_r = axes[0, col]
        ax_a = axes[1, col]

        daily    = ev_data["daily"]
        norm     = ev_data["norm"]
        slope    = ev_data["slope"]
        intercept = ev_data["intercept"]
        r        = ev_data["r"]

        x     = daily["key"]
        ratio = daily["ratio"]

        # ── Ap バー (下段) ──────────────────────────────────────────────
        if ev["mode"] == "doy":
            ax_a.bar(daily["key"], daily["ap"], width=0.8,
                     color="slategray", alpha=0.55, zorder=1)
        else:
            ax_a.bar(daily["key"], daily["ap"],
                     width=pd.Timedelta(days=1),
                     color="slategray", alpha=0.55, zorder=1)
        ax_a.axhline(AP_KP3, color="darkgray", lw=1.2, ls=":", zorder=2,
                     label=f"Kp=3 (Ap={AP_KP3:.0f})")
        ax_a.set_ylabel("Ap index", fontsize=9)
        ap_max = daily["ap"].max() if not daily["ap"].empty else AP_KP3
        ax_a.set_ylim(0, max(ap_max * 1.6, AP_KP3 * 1.5))
        ax_a.grid(True, ls=":", alpha=0.3)
        ax_a.legend(fontsize=7, loc="upper right")

        # x 軸設定
        if ev["mode"] == "date":
            ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax_a.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            plt.setp(ax_a.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax_a.set_xlabel(f"{'DOY' if ev['mode']=='doy' else 'Date'} ({ev['year']})", fontsize=9)

        # ── ratio 時系列 (上段) ─────────────────────────────────────────
        ax_r.plot(x, ratio.values,
                  color=ev["color_raw"], lw=2.0, marker="o", ms=4,
                  zorder=5, label="Original ratio ($y_i$)")
        ax_r.plot(x, norm.values,
                  color=ev["color_norm"], lw=2.0, ls="--", marker="^", ms=4,
                  zorder=6, label=r"$\Delta y_{Ap} = y_i - y_{pred}(Ap_i)$")
        ax_r.axhline(0, color="gray", lw=0.8, ls=":", zorder=3)

        # SSW ピーク
        pk = ev.get("ssw_peak_doy") or ev.get("ssw_peak")
        if pk is not None:
            ax_r.axvline(pk, color="red", lw=1.8, ls="--", alpha=0.8, zorder=7, label="SSW Peak")
            ax_a.axvline(pk, color="red", lw=1.8, ls="--", alpha=0.8, zorder=7)
            ax_r.text(pk, 0.94, " SSW Peak", transform=ax_r.get_xaxis_transform(),
                      fontsize=8.5, color="red", fontweight="bold", alpha=0.9, va="top")

        # ref 期間シェード
        if ev["mode"] == "doy":
            for lo, hi in ev["ref_doy"]:
                ax_r.axvspan(lo, hi, color="lightblue", alpha=0.25, lw=0)
                ax_a.axvspan(lo, hi, color="lightblue", alpha=0.25, lw=0)
        else:
            for s, e in ev["ref_dates"]:
                ax_r.axvspan(s, e, color="lightblue", alpha=0.25, lw=0)
                ax_a.axvspan(s, e, color="lightblue", alpha=0.25, lw=0)

        # 回帰係数テキスト
        ax_r.text(
            0.02, 0.97,
            f"slope a={slope:.5f}\nintercept b={intercept:.4f}\n"
            f"corr(y, Ap)={r:.3f}",
            transform=ax_r.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

        ax_r.set_title(f"{ev['label']}  ({ev['sat']})", fontsize=11, fontweight="bold")
        ax_r.set_ylabel("Density Ratio & Residual", fontsize=10)
        ax_r.grid(True, ls=":", alpha=0.45)
        ax_r.tick_params(labelbottom=False)
        ax_r.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "ap_norm_Ap6_3years.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─── 散布図 + 回帰診断プロット ────────────────────────────────────────────────
def plot_scatter_diagnostics(all_data: list[dict]) -> None:
    """
    各イベントについて:
      左列: ratio vs Ap (補正前, debug 図と同じ)
      右列: ratio_norm vs Ap (補正後)
    2 列 × 3 行
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    fig.suptitle(
        f"Scatter Diagnostics: ratio vs Ap  —  Before / After Ap={AP_REF:.0f} Normalization",
        fontsize=13, fontweight="bold",
    )

    for row, ev_data in enumerate(all_data):
        ev     = ev_data["ev"]
        daily  = ev_data["daily"]
        norm   = ev_data["norm"]
        slope  = ev_data["slope"]
        intercept = ev_data["intercept"]
        r      = ev_data["r"]

        x_ap   = daily["ap"].values
        y_raw  = daily["ratio"].values
        y_norm = norm.values

        # 回帰直線用 x
        ap_range = np.linspace(0, max(x_ap.max() * 1.2, AP_KP3 * 2), 100)

        # 補正後の corr
        r_norm = float(np.corrcoef(x_ap, y_norm)[0, 1]) if len(x_ap) >= 3 else np.nan

        # ── 左列: Before ─────────────────────────────────────────────────
        ax_b = axes[row, 0]
        ax_b.scatter(x_ap, y_raw, color="#d62728", s=60, edgecolors="k", zorder=3, alpha=0.8)
        ax_b.plot(ap_range, slope * ap_range + intercept, color="black", lw=1.5, ls="--",
                  label=f"y={slope:.5f}x+{intercept:.4f}")
        ax_b.axvline(AP_REF, color="darkorange", lw=1.5, ls="--", alpha=0.8,
                     label=f"Ap={AP_REF:.0f} ref")
        ax_b.axhline(slope * AP_REF + intercept, color="darkorange", lw=0.8, ls=":", alpha=0.6)
        ax_b.set_xlabel("Daily Mean Ap Index", fontsize=9)
        ax_b.set_ylabel("ratio (daily median)", fontsize=9)
        ax_b.set_title(f"{ev['label']} — Before  (corr={r:.3f})", fontsize=10, fontweight="bold")
        ax_b.grid(True, ls=":", alpha=0.4)
        ax_b.legend(fontsize=8)

        # ── 右列: After ──────────────────────────────────────────────────
        ax_a = axes[row, 1]
        ax_a.scatter(x_ap, y_norm, color="#ff7f0e", s=60, edgecolors="k", zorder=3, alpha=0.8)
        # 補正後の回帰直線
        if len(x_ap) >= 3:
            p_norm = np.polyfit(x_ap, y_norm, 1)
            ax_a.plot(ap_range, np.polyval(p_norm, ap_range),
                      color="black", lw=1.5, ls="--",
                      label=f"y={p_norm[0]:.5f}x+{p_norm[1]:.4f}")
        ax_a.axvline(AP_REF, color="darkorange", lw=1.5, ls="--", alpha=0.8,
                     label=f"Ap={AP_REF:.0f} ref")
        # 理想的な水平線 (slope=0)
        ax_a.axhline(float(np.mean(y_norm)), color="gray", lw=0.8, ls=":",
                     label=f"mean={np.mean(y_norm):.4f}")
        ax_a.set_xlabel("Daily Mean Ap Index", fontsize=9)
        ax_a.set_ylabel(f"ratio_norm (Ap={AP_REF:.0f} basis)", fontsize=9)
        ax_a.set_title(f"{ev['label']} — After  (corr={r_norm:.3f})", fontsize=10, fontweight="bold")
        ax_a.grid(True, ls=":", alpha=0.4)
        ax_a.legend(fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "ap_norm_Ap6_scatter_3years.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─── メイン ──────────────────────────────────────────────────────────────────
def main() -> None:
    all_data = []

    for ev in EVENTS:
        print(f"\n{'='*60}")
        print(f"Processing: {ev['label']}  ({ev['sat']})")
        df    = load_event_df(ev)
        daily = make_daily(df)

        # 全期間で回帰フィット (debug 図と同一ロジック)
        slope, intercept, r = fit_regression(daily)
        print(f"  slope={slope:.6f}, intercept={intercept:.5f}, corr={r:.3f}")

        # Delta y_Ap = y_i - y_pred(Ap_i)
        norm_series = calc_ap_residual(daily, slope, intercept)

        # daily を key をインデックスにして Series に変換
        daily_ratio = pd.Series(daily["ratio"].values, index=daily["key"])
        daily_norm  = pd.Series(norm_series.values,    index=daily["key"])

        # Ap=6 での予測 ratio
        ratio_at_ap6 = slope * AP_REF + intercept
        print(f"  ratio predicted at Ap={AP_REF:.0f}: {ratio_at_ap6:.5f}")
        print(f"  mean(raw): {daily['ratio'].mean():.5f}  "
              f"mean(norm): {norm_series.mean():.5f}")

        # 補正後の相関
        r_after = float(np.corrcoef(daily["ap"].values, norm_series.values)[0, 1])
        print(f"  corr(ratio, Ap) before: {r:.3f}  after: {r_after:.3f}")

        all_data.append({
            "ev": ev,
            "daily": daily,
            "norm": norm_series,
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "r_after": r_after,
        })

    print("\nPlotting time series figure...")
    plot_timeseries(all_data)

    print("Plotting scatter diagnostics figure...")
    plot_scatter_diagnostics(all_data)

    print("\n=== Summary ===")
    for d in all_data:
        print(f"  {d['ev']['label']:20s}  slope={d['slope']:.5f}  "
              f"corr before={d['r']:.3f}  after={d['r_after']:.3f}")


if __name__ == "__main__":
    main()
