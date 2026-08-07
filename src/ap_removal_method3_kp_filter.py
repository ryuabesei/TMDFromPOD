"""
ap_removal_method3_kp_filter.py

手法③: Kp/Ap フィルタリング (Threshold Filtering, Kp < 3)

アルゴリズム:
    - 日平均 AP_AVG >= 15 (Kp=3 相当) の日をマスク除外
    - 除外された日を赤シェードで可視化
    - 残ったデータ（Kp<3 静穏日のみ）の daily median を描画
    - 全データ版と静穏日フィルタ後を重ねてプロット

出力:
    Figure/Ap_removal/method3_kp_filter.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ─── Paths ───────────────────────────────────────────────────────────────────
PARQUET_2018 = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet")
PARQUET_2019 = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet")
PARQUET_2021 = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet")

OUT_DIR = Path("Figure/Ap_removal")
OUT_MAIN = OUT_DIR / "method3_kp_filter.png"

AP_KP3 = 15.0  # Kp=3 相当

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
    df = df.dropna(subset=["datetime", "lat", "density_ratio_msis", "AP_AVG"])

    if ev["mode"] == "doy":
        df["key"] = df["datetime"].dt.dayofyear
        df = df[(df["key"] >= ev["doy_start"]) & (df["key"] <= ev["doy_end"])]
    else:
        df["key"] = df["datetime"].dt.normalize()
        df = df[(df["datetime"] >= ev["date_start"]) &
                (df["datetime"] <= ev["date_end"] + pd.Timedelta(hours=23, minutes=59))]
    return df


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


def get_disturbed_keys(df: pd.DataFrame) -> set:
    """AP_AVG >= 15 の日の key 集合を返す"""
    daily_ap = df.groupby("key")["AP_AVG"].mean()
    return set(daily_ap[daily_ap >= AP_KP3].index)


# ─── メインプロット ───────────────────────────────────────────────────────────
def plot_main(all_data: list[dict]) -> None:
    n_bands = len(LAT_BANDS)
    n_events = len(EVENTS)
    fig = plt.figure(figsize=(20, 4.5 * n_bands))
    fig.suptitle(
        "Method 3 — Kp<3 Threshold Filtering (SWARM-C)\n"
        "Blue solid: all data  |  Purple dashed: Kp<3 only  |  Red shade: disturbed days (Ap>=15)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(n_bands, n_events, figure=fig, hspace=0.35, wspace=0.20)

    for row, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
        for col, (ev_data, ev_cfg) in enumerate(zip(all_data, EVENTS)):
            ax = fig.add_subplot(gs[row, col])
            ax2 = ax.twinx()

            delta_all  = ev_data["delta_all"][band_label]
            delta_filt = ev_data["delta_filt"][band_label]
            daily_ap   = ev_data["daily_ap"][band_label]
            disturbed  = ev_data["disturbed_keys"]

            # Ap バー
            if ev_cfg["mode"] == "doy":
                ax2.bar(daily_ap.index, daily_ap.values, width=0.8,
                        color="slategray", alpha=0.30, zorder=1)
                # 攪乱日のシェード
                for k in disturbed:
                    ax.axvspan(k - 0.5, k + 0.5, color="lightcoral", alpha=0.35, lw=0, zorder=2)
            else:
                ax2.bar(daily_ap.index, daily_ap.values, width=pd.Timedelta(days=1),
                        color="slategray", alpha=0.30, zorder=1)
                for k in disturbed:
                    ax.axvspan(k - pd.Timedelta(hours=12), k + pd.Timedelta(hours=12),
                               color="lightcoral", alpha=0.35, lw=0, zorder=2)

            ax2.axhline(AP_KP3, color="darkgray", lw=1.0, ls=":", zorder=2)
            ax2.set_ylabel("Ap", fontsize=8, color="slategray")
            ax2.tick_params(axis="y", labelcolor="slategray", labelsize=7)
            ap_max = daily_ap.max() if not daily_ap.empty else AP_KP3
            ax2.set_ylim(0, max(ap_max * 2.5, AP_KP3 * 3))

            # 全データ (青)
            ax.plot(delta_all.index, delta_all.values,
                    color="#1f77b4", lw=2.0, zorder=5, label="All data")
            # フィルタ後 (紫 dashed)
            ax.plot(delta_filt.index, delta_filt.values,
                    color="#9467bd", lw=2.0, ls="--", zorder=6, label="Kp<3 only")
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

            n_all  = ev_data["n_days_all"][band_label]
            n_filt = ev_data["n_days_filt"][band_label]
            n_removed = ev_data["n_days_removed"]
            ax.text(0.98, 0.97,
                    f"Days total: {n_all}\nKp<3 days: {n_filt}\nRemoved: {n_removed}",
                    transform=ax.transAxes,
                    fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_MAIN, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_MAIN}")


# ─── メイン ──────────────────────────────────────────────────────────────────
def main() -> None:
    all_data = []

    for ev in EVENTS:
        print(f"\n{'='*60}")
        print(f"Processing: {ev['label']}")
        df = load_event_df(ev)

        # 攪乱日のキー集合
        disturbed_keys = get_disturbed_keys(df)
        n_removed = len(disturbed_keys)
        print(f"  Disturbed days (Ap>=15): {n_removed} days")
        print(f"  Keys: {sorted(disturbed_keys)}")

        # フィルタ後 DataFrame
        df_filt = df[~df["key"].isin(disturbed_keys)]

        ev_data = dict(
            label=ev["label"],
            delta_all={},
            delta_filt={},
            daily_ap={},
            n_days_all={},
            n_days_filt={},
            n_days_removed=n_removed,
            disturbed_keys=disturbed_keys,
        )

        for band_label, lat_lo, lat_hi in LAT_BANDS:
            band_mask  = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
            filt_mask  = (df_filt["lat"].abs() >= lat_lo) & (df_filt["lat"].abs() < lat_hi)

            df_band      = df[band_mask]
            df_band_filt = df_filt[filt_mask]

            daily_all  = df_band.groupby("key")["density_ratio_msis"].median()
            daily_filt = df_band_filt.groupby("key")["density_ratio_msis"].median()
            daily_ap   = df_band.groupby("key")["AP_AVG"].mean()

            # delta (全データの ref で共通化)
            delta_all  = compute_delta(daily_all,  ev)
            delta_filt = compute_delta(daily_filt, ev)

            n_all  = len(daily_all)
            n_filt = len(daily_filt)
            print(f"  {band_label}: {n_all} days -> {n_filt} days after filter")

            ev_data["delta_all"][band_label]   = delta_all
            ev_data["delta_filt"][band_label]  = delta_filt
            ev_data["daily_ap"][band_label]    = daily_ap
            ev_data["n_days_all"][band_label]  = n_all
            ev_data["n_days_filt"][band_label] = n_filt

        all_data.append(ev_data)

    print("\nPlotting main figure...")
    plot_main(all_data)
    print("\nMethod 3 complete.")


if __name__ == "__main__":
    main()
