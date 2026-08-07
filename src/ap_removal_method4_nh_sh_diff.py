"""
ap_removal_method4_nh_sh_diff.py

手法④: 南北半球差分 (NH - SH Differential Signal)

アルゴリズム:
    1. 各緯度帯で NH (lat>0) と SH (lat<0) の daily median を計算
    2. それぞれ ref 期間中央値を引いて delta_NH, delta_SH を算出
    3. 差分シグナル = delta_NH - delta_SH を計算
       → Ap 由来の全球一様成分が相殺され、SSW の南北非対称成分が残る

    4. NH/SH それぞれのシグナルと差分を 3段レイアウトで描画:
       - 上段: delta_NH (青), delta_SH (赤)
       - 中段: NH - SH 差分 (黒 + fill)
       - 下段: Ap index バー

出力:
    Figure/Ap_removal/method4_nh_sh_diff.png
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
OUT_MAIN = OUT_DIR / "method4_nh_sh_diff.png"

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


# ─── メインプロット ───────────────────────────────────────────────────────────
def plot_main(all_data: list[dict]) -> None:
    n_bands = len(LAT_BANDS)
    n_events = len(EVENTS)

    # 各イベントに 3サブパネル (NH/SH, 差分, Ap) x 3緯度帯
    # Layout: n_bands 行 × n_events 列、各セルに 3段
    fig = plt.figure(figsize=(22, 5.5 * n_bands))
    fig.suptitle(
        "Method 4 — NH-SH Differential Signal (SWARM-C)\n"
        "Top: delta_NH (blue) & delta_SH (red)  |  Middle: NH - SH diff (black, filled)  |  Bottom: Ap",
        fontsize=14, fontweight="bold", y=0.995,
    )

    outer_gs = gridspec.GridSpec(n_bands, n_events, figure=fig, hspace=0.38, wspace=0.18)

    for row, (band_label, lat_lo, lat_hi) in enumerate(LAT_BANDS):
        for col, (ev_data, ev_cfg) in enumerate(zip(all_data, EVENTS)):
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=outer_gs[row, col],
                height_ratios=[2.5, 2.0, 1.0], hspace=0.05
            )

            ax_nh  = fig.add_subplot(inner_gs[0])
            ax_dif = fig.add_subplot(inner_gs[1], sharex=ax_nh)
            ax_ap  = fig.add_subplot(inner_gs[2], sharex=ax_nh)

            delta_nh  = ev_data["delta_nh"][band_label]
            delta_sh  = ev_data["delta_sh"][band_label]
            diff      = ev_data["diff"][band_label]
            daily_ap  = ev_data["daily_ap"][band_label]

            # ── 上段: NH (青) / SH (赤) ──────────────────────────────────
            ax_nh.plot(delta_nh.index, delta_nh.values,
                       color="#1f77b4", lw=2.0, label="NH")
            ax_nh.plot(delta_sh.index, delta_sh.values,
                       color="#d62728", lw=2.0, label="SH")
            ax_nh.axhline(0, color="gray", lw=0.6, ls=":")
            ax_nh.set_ylabel(f"{band_label}\ndelta ratio", fontsize=8)
            ax_nh.grid(True, ls=":", alpha=0.4)
            ax_nh.tick_params(labelbottom=False)

            # SSW ピーク
            pk = ev_cfg.get("ssw_peak_doy") or ev_cfg.get("ssw_peak")
            for ax_ in [ax_nh, ax_dif, ax_ap]:
                if pk is not None:
                    ax_.axvline(pk, color="red", lw=1.5, ls="--", alpha=0.8, zorder=7)
                # Ref 期間シェード
                if ev_cfg["mode"] == "doy":
                    for lo, hi in ev_cfg["ref_doy"]:
                        ax_.axvspan(lo, hi, color="lightblue", alpha=0.20, lw=0)
                else:
                    for s, e in ev_cfg["ref_dates"]:
                        ax_.axvspan(s, e, color="lightblue", alpha=0.20, lw=0)

            # ── 中段: NH - SH 差分 (黒) ──────────────────────────────────
            ax_dif.plot(diff.index, diff.values,
                        color="black", lw=2.2, label="NH - SH")
            ax_dif.fill_between(diff.index, 0, diff.values,
                                 where=(diff.values > 0), color="steelblue", alpha=0.30)
            ax_dif.fill_between(diff.index, 0, diff.values,
                                 where=(diff.values <= 0), color="salmon", alpha=0.30)
            ax_dif.axhline(0, color="gray", lw=0.8, ls=":")
            ax_dif.set_ylabel("NH - SH", fontsize=8)
            ax_dif.grid(True, ls=":", alpha=0.4)
            ax_dif.tick_params(labelbottom=False)

            # ── 下段: Ap バー ────────────────────────────────────────────
            if ev_cfg["mode"] == "doy":
                ax_ap.bar(daily_ap.index, daily_ap.values, width=0.8,
                          color="slategray", alpha=0.55, zorder=1)
            else:
                ax_ap.bar(daily_ap.index, daily_ap.values, width=pd.Timedelta(days=1),
                          color="slategray", alpha=0.55, zorder=1)
            ax_ap.axhline(AP_KP3, color="darkgray", lw=1.0, ls=":", zorder=2)
            ax_ap.set_ylabel("Ap", fontsize=8)
            ap_max = daily_ap.max() if not daily_ap.empty else AP_KP3
            ax_ap.set_ylim(0, max(ap_max * 1.5, AP_KP3 * 2))
            ax_ap.grid(True, ls=":", alpha=0.3)

            # x軸
            if ev_cfg["mode"] == "date":
                ax_ap.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                ax_ap.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                plt.setp(ax_ap.xaxis.get_majorticklabels(), rotation=30, ha="right")

            # 上段タイトル
            if row == 0:
                ax_nh.set_title(ev_cfg["label"], fontsize=11, fontweight="bold")

            if row == 0 and col == 0:
                ax_nh.legend(loc="upper right", fontsize=8, framealpha=0.85)
                ax_dif.legend(loc="upper right", fontsize=8, framealpha=0.85)

            # 相関係数 (diff と Ap の相関)
            common = diff.dropna()
            ap_common = daily_ap.reindex(common.index)
            valid = common.notna() & ap_common.notna()
            if valid.sum() >= 3:
                corr_diff_ap = float(np.corrcoef(common[valid].values, ap_common[valid].values)[0, 1])
            else:
                corr_diff_ap = np.nan
            ax_dif.text(0.98, 0.95,
                        f"corr(NH-SH, Ap)={corr_diff_ap:.3f}",
                        transform=ax_dif.transAxes,
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

        ev_data = dict(
            label=ev["label"],
            delta_nh={},
            delta_sh={},
            diff={},
            daily_ap={},
        )

        for band_label, lat_lo, lat_hi in LAT_BANDS:
            abs_mask = (df["lat"].abs() >= lat_lo) & (df["lat"].abs() < lat_hi)
            df_band  = df[abs_mask]

            # NH / SH 分割
            nh = df_band[df_band["lat"] > 0].groupby("key")["density_ratio_msis"].median()
            sh = df_band[df_band["lat"] < 0].groupby("key")["density_ratio_msis"].median()
            ap = df_band.groupby("key")["AP_AVG"].mean()

            delta_nh = compute_delta(nh, ev)
            delta_sh = compute_delta(sh, ev)

            # 差分 = delta_NH - delta_SH (共通インデックスで計算)
            common_idx = delta_nh.index.intersection(delta_sh.index)
            diff = delta_nh.reindex(common_idx) - delta_sh.reindex(common_idx)

            # corr(raw_nh, Ap), corr(diff, Ap)
            ap_common = ap.reindex(common_idx)
            nh_common = delta_nh.reindex(common_idx)
            valid = nh_common.notna() & ap_common.notna()
            corr_nh_ap   = float(np.corrcoef(nh_common[valid].values, ap_common[valid].values)[0, 1]) if valid.sum() >= 3 else np.nan
            valid2 = diff.notna() & ap_common.notna()
            corr_diff_ap = float(np.corrcoef(diff[valid2].values, ap_common[valid2].values)[0, 1]) if valid2.sum() >= 3 else np.nan

            print(f"  {band_label}: corr(delta_NH, Ap)={corr_nh_ap:.3f}  "
                  f"corr(NH-SH diff, Ap)={corr_diff_ap:.3f}")

            ev_data["delta_nh"][band_label] = delta_nh
            ev_data["delta_sh"][band_label] = delta_sh
            ev_data["diff"][band_label]     = diff
            ev_data["daily_ap"][band_label] = ap

        all_data.append(ev_data)

    print("\nPlotting main figure...")
    plot_main(all_data)
    print("\nMethod 4 complete.")


if __name__ == "__main__":
    main()
