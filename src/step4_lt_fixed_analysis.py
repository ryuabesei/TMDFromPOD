"""
step4_lt_fixed_analysis.py
===========================
STEP 4: LT固定解析

目的:
  Swarm衛星のLT driftがSSW期間と共変することが確認された（STEP1）ため、
  同じLT条件に限定して density_ratio_msis の時系列を再計算し、
  LT補正後もSSW期間中の密度異常が残存するかを診断する。

科学的意義:
  仮説3（LTドリフトによる見かけの変動）を部分的に除外する。
  LT固定後に負偏差が残れば仮説1・2の検討へ進む。
  LT固定後に負偏差が消えれば LT drift が主因と判断する。

出力:
  Figure/STEP4/{event}_{sat}/
    00_LT_coverage_*.png/pdf
    01_LT_bins_available.csv
    02_lt{LT}_timeseries.png/pdf   ← LT固定時系列
    03_multi_LT_overlay.png/pdf    ← 複数LT帯重ね描き
    04_LT_date_2D.png/pdf          ← LT×date 2D map
    05_satellite_comparison.png/pdf ← 衛星別比較
    06_latitude_comparison.png/pdf  ← 緯度帯別比較
  Figure/STEP4/STEP4_summary.csv   ← 全イベント×衛星×LT帯のサマリ
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────

COLS = {
    "time":     "datetime",
    "lat":      "lat",
    "lon":      "lon",
    "alt_km":   "alt_km",
    "LT":       "lst_h",
    "ratio":    "density_ratio_msis",
    "rho_obs":  "density",
    "Ap":       "AP_AVG",
    "F107":     "F107",
}

EVENTS = {
    "2018_NH": {
        "label":      "2018 NH SSW",
        "ssw_onset":  pd.Timestamp("2018-02-12", tz="UTC"),
        "ssw_end":    pd.Timestamp("2018-02-28", tz="UTC"),
        "plot_start": pd.Timestamp("2018-01-20", tz="UTC"),
        "plot_end":   pd.Timestamp("2018-03-20", tz="UTC"),
        "sats": {
            "A": "normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet",
            "B": "normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet",
            "C": "normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet",
        },
    },
    "2019_SH": {
        "label":      "2019 SH SSW",
        "ssw_onset":  pd.Timestamp("2019-09-04", tz="UTC"),
        "ssw_end":    pd.Timestamp("2019-09-19", tz="UTC"),
        "plot_start": pd.Timestamp("2019-08-20", tz="UTC"),
        "plot_end":   pd.Timestamp("2019-09-23", tz="UTC"),
        "sats": {
            "A": "normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet",
            "B": "normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet",
            "C": "normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet",
        },
    },
    "2021_NH": {
        "label":      "2021 NH SSW",
        "ssw_onset":  pd.Timestamp("2021-01-04", tz="UTC"),
        "ssw_end":    pd.Timestamp("2021-01-16", tz="UTC"),
        "plot_start": pd.Timestamp("2020-12-25", tz="UTC"),
        "plot_end":   pd.Timestamp("2021-02-05", tz="UTC"),
        "sats": {
            "C": "normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet",
        },
    },
}

LAT_BANDS = [
    ("all",  -60.0,  60.0),
    ("NH_hi", 30.0,  60.0),
    ("EQ",  -30.0,  30.0),
    ("SH_hi",-60.0, -30.0),
]

LT_HALF_WIDTHS = [0.5, 1.0, 1.5]   # ±0.5h, ±1.0h, ±1.5h
MIN_N_DAY = 5                       # 1日あたり最低観測数
MIN_PHASE_DAYS = 3                  # 各フェーズで最低3日必要

RATIO_VALID_MIN, RATIO_VALID_MAX = 0.1, 3.0
OUT_BASE = Path("Figure/STEP4")
OUT_BASE.mkdir(parents=True, exist_ok=True)

# 集計結果保存用
ALL_SUMMARY = []


# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────

def savefig(fig, path: Path) -> None:
    fig.savefig(str(path) + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf",           bbox_inches="tight")
    logger.info(f"  Saved: {str(path)}.png")
    plt.close(fig)


def assign_phase(t: pd.Series, onset, end) -> pd.Categorical:
    """
    SSWフェーズラベルを付与。tz-naive / tz-aware の混在を自動解決する。
    """
    # onset/end を tz-naive に統一
    def to_naive(ts):
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            return ts.tz_convert(None)
        return ts
    onset_n = to_naive(onset)
    end_n   = to_naive(end)

    # t 列を tz-naive に統一
    if hasattr(t, "dt"):
        if t.dt.tz is not None:
            t_cmp = t.dt.tz_convert(None)
        else:
            t_cmp = t
    else:
        t_cmp = t

    labels = pd.Categorical(
        ["SSW中" if (onset_n <= ts <= end_n) else
         ("SSW前" if ts < onset_n else "SSW後")
         for ts in t_cmp],
        categories=["SSW前", "SSW中", "SSW後"], ordered=True,
    )
    return labels


def load_data(path: str, ev: dict) -> pd.DataFrame:
    df = pd.read_parquet(path)
    tc = COLS["time"]
    df[tc] = pd.to_datetime(df[tc], utc=True, errors="coerce")
    df = df.dropna(subset=[tc, COLS["lat"], COLS["LT"], COLS["ratio"]])
    df = df[(df[tc] >= ev["plot_start"]) & (df[tc] <= ev["plot_end"])].copy()
    n_raw = len(df)
    df = df[(df[COLS["ratio"]] >= RATIO_VALID_MIN) & (df[COLS["ratio"]] <= RATIO_VALID_MAX)].copy()
    df["phase"] = assign_phase(df[tc], ev["ssw_onset"], ev["ssw_end"])
    df["date"] = df[tc].dt.normalize()
    logger.info(f"    rows: {n_raw:,} -> {len(df):,} after filter")
    return df


# ──────────────────────────────────────────────
# PART 1: LTカバレッジ分析
# ──────────────────────────────────────────────

def analyze_lt_coverage(df: pd.DataFrame, ev: dict, ev_key: str, sat: str,
                         out_dir: Path) -> pd.DataFrame:
    """
    日ごとのLT中央値・IQR・ヒストグラムを作成し、
    SSW前・中・後に共通して存在するLT帯を特定する。
    """
    phases = ["SSW前", "SSW中", "SSW後"]
    PHASE_COLORS = {"SSW前": "#2196F3", "SSW中": "#F44336", "SSW後": "#4CAF50"}

    # ── 図A: 日ごとの median LT + IQR ────────────────────────────────────
    daily_lt = df.groupby("date")[COLS["LT"]].agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="count",
    ).reset_index()
    daily_lt["phase"] = assign_phase(
        pd.to_datetime(daily_lt["date"], utc=True),
        ev["ssw_onset"], ev["ssw_end"]
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    for phase, grp in daily_lt.groupby("phase"):
        ax.fill_between(pd.to_datetime(grp["date"]),
                        grp["q25"], grp["q75"],
                        alpha=0.25, color=PHASE_COLORS[phase])
        ax.plot(pd.to_datetime(grp["date"]), grp["median"],
                "o-", ms=3.5, lw=1.5, color=PHASE_COLORS[phase], label=phase)
    ax.axvline(ev["ssw_onset"], color="red",    ls="--", lw=1.5)
    ax.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5)
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 3))
    ax.set_ylabel("Local Solar Time [h]")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(f"{ev['label']} Swarm-{sat} — Daily Median LT ± IQR", fontweight="bold")
    savefig(fig, out_dir / "00a_LT_daily_median")

    # ── 図B: フェーズ別LTヒストグラム ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, phase in zip(axes, phases):
        sub = df[df["phase"] == phase][COLS["LT"]]
        if len(sub) == 0:
            ax.set_title(f"{phase}\nN=0")
            continue
        ax.hist(sub, bins=np.arange(0, 24.5, 0.5), color=PHASE_COLORS[phase],
                alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("Local Solar Time [h]")
        ax.set_ylabel("Count")
        ax.set_title(f"{phase}  N={len(sub):,}")
        ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle(f"{ev['label']} Swarm-{sat} — LT Histogram by Phase", fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "00b_LT_histogram_by_phase")

    # ── 共通LT帯の特定 ─────────────────────────────────────────────────────
    # 各LT半幅ごとに、全フェーズにデータが存在するLT中心値を列挙
    lt_centers = np.arange(0.5, 24.0, 0.5)   # 0.5刻み

    available_bins = []
    for half_w in LT_HALF_WIDTHS:
        for lt_c in lt_centers:
            lo, hi = lt_c - half_w, lt_c + half_w
            # 0-24hで折り返し処理
            phase_counts = {}
            for phase in phases:
                sub = df[df["phase"] == phase]
                if lo < 0:
                    mask = (sub[COLS["LT"]] <= hi) | (sub[COLS["LT"]] >= 24 + lo)
                elif hi > 24:
                    mask = (sub[COLS["LT"]] >= lo) | (sub[COLS["LT"]] <= hi - 24)
                else:
                    mask = (sub[COLS["LT"]] >= lo) & (sub[COLS["LT"]] < hi)
                # 日ごとにカウント
                sub_f = sub[mask]
                n_days = sub_f.groupby("date")[COLS["ratio"]].count()
                n_days_ok = (n_days >= MIN_N_DAY).sum()
                phase_counts[phase] = {"total_obs": len(sub_f), "n_days_ok": n_days_ok}

            # 全フェーズで最低MIN_PHASE_DAYS日あれば有効
            all_ok = all(v["n_days_ok"] >= MIN_PHASE_DAYS for v in phase_counts.values())
            min_obs = min(v["total_obs"] for v in phase_counts.values())
            if all_ok:
                available_bins.append({
                    "half_width": half_w,
                    "lt_center": lt_c,
                    "lt_lo": lo,
                    "lt_hi": hi,
                    **{f"n_obs_{p}": phase_counts[p]["total_obs"] for p in phases},
                    **{f"n_days_{p}": phase_counts[p]["n_days_ok"] for p in phases},
                    "min_obs_across_phases": min_obs,
                })

    bins_df = pd.DataFrame(available_bins)
    if len(bins_df) > 0:
        bins_df.to_csv(out_dir / "01_LT_bins_available.csv", index=False, encoding="utf-8-sig")
        logger.info(f"  Available LT bins: {len(bins_df)}")
        logger.info(f"\n{bins_df.to_string(index=False)}")
    else:
        logger.warning(f"  No valid LT bins found for {ev_key} Swarm-{sat}!")

    return bins_df


# ──────────────────────────────────────────────
# PART 2: LT固定時系列の計算と可視化
# ──────────────────────────────────────────────

def lt_fixed_timeseries(df: pd.DataFrame, lt_center: float, half_w: float,
                         ev: dict, lat_band: tuple) -> pd.DataFrame | None:
    """
    指定LT帯・緯度帯に限定した日別統計を計算して返す。
    """
    band_name, lat_lo, lat_hi = lat_band
    lo, hi = lt_center - half_w, lt_center + half_w

    # LTフィルタ（折り返し考慮）
    if lo < 0:
        lt_mask = (df[COLS["LT"]] <= hi) | (df[COLS["LT"]] >= 24 + lo)
    elif hi > 24:
        lt_mask = (df[COLS["LT"]] >= lo) | (df[COLS["LT"]] <= hi - 24)
    else:
        lt_mask = (df[COLS["LT"]] >= lo) & (df[COLS["LT"]] < hi)

    lat_mask = (df[COLS["lat"]] >= lat_lo) & (df[COLS["lat"]] < lat_hi)
    sub = df[lt_mask & lat_mask].copy()
    if len(sub) < 10:
        return None

    daily = sub.groupby("date").agg(
        ratio_med=(COLS["ratio"], "median"),
        ratio_q25=(COLS["ratio"], lambda x: x.quantile(0.25)),
        ratio_q75=(COLS["ratio"], lambda x: x.quantile(0.75)),
        ratio_std=(COLS["ratio"], "std"),
        LT_med=(COLS["LT"], "median"),
        alt_med=(COLS["alt_km"], "median"),
        Ap_med=(COLS["Ap"], "median"),
        F107_med=(COLS["F107"], "median"),
        n_obs=(COLS["ratio"], "count"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    # date列はtz-naive（groupby後）なので、onset/endをtz-naiveに変換して比較
    onset_naive = ev["ssw_onset"].tz_localize(None) if ev["ssw_onset"].tzinfo is None else ev["ssw_onset"].tz_convert(None)
    end_naive   = ev["ssw_end"].tz_localize(None)   if ev["ssw_end"].tzinfo is None   else ev["ssw_end"].tz_convert(None)
    daily["phase"] = assign_phase(
        daily["date"],
        onset_naive, end_naive
    )
    daily["lat_band"] = band_name
    daily["lt_center"] = lt_center
    daily["half_width"] = half_w
    return daily


def plot_lt_fixed_single(daily: pd.DataFrame, lt_center: float, half_w: float,
                          ev: dict, ev_key: str, sat: str,
                          lat_band: tuple, out_dir: Path) -> None:
    """
    図A: 単一LT帯のdate vs density_ratio_msis
    """
    band_name, lat_lo, lat_hi = lat_band
    PHASE_COLORS = {"SSW前": "#2196F3", "SSW中": "#F44336", "SSW後": "#4CAF50"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2.5, 1]})

    dates = daily["date"]
    ax1.fill_between(dates, daily["ratio_q25"], daily["ratio_q75"], alpha=0.25, color="#1976D2")
    ax1.plot(dates, daily["ratio_med"], "o-", color="#1976D2", lw=1.8, ms=4.5,
             label=r"$\rho_{obs}/\rho_{MSIS}$ median±IQR")

    # フェーズ別に色分けでマーカーを重ねる
    for phase, col in PHASE_COLORS.items():
        g = daily[daily["phase"] == phase]
        if len(g):
            ax1.scatter(g["date"], g["ratio_med"], c=col, s=30, zorder=5,
                        edgecolors="k", linewidths=0.3, label=phase)

    ax1.axvspan(ev["ssw_onset"], ev["ssw_end"], color="red", alpha=0.07)
    ax1.axvline(ev["ssw_onset"], color="red",    ls="--", lw=1.5)
    ax1.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5)
    ax1.axhline(1.0, color="gray", ls=":", lw=1)

    # SSW前の中央値を基準線として表示
    pre_med = daily[daily["phase"] == "SSW前"]["ratio_med"].median()
    if np.isfinite(pre_med):
        ax1.axhline(pre_med, color="#1976D2", ls="-.", lw=1.0, alpha=0.7,
                    label=f"SSW前中央値={pre_med:.4f}")

    ax1.set_ylabel(r"$\rho_{obs}/\rho_{MSIS}$", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5, ncol=2)
    ax1.grid(True, ls=":", alpha=0.4)
    total_n = daily["n_obs"].sum()
    ax1.set_title(
        f"{ev['label']} Swarm-{sat} | LT={lt_center:.1f}h ±{half_w:.1f}h | "
        f"lat=[{lat_lo:.0f},{lat_hi:.0f}°]\n"
        f"N_total={total_n:,}  ({band_name})",
        fontweight="bold", fontsize=10
    )

    # Ap パネル
    ax2.bar(dates, daily["Ap_med"], color="#E57373", width=0.8, alpha=0.7)
    ax2.axvspan(ev["ssw_onset"], ev["ssw_end"], color="red", alpha=0.07)
    ax2.axvline(ev["ssw_onset"], color="red",    ls="--", lw=1.5)
    ax2.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5)
    ax2.set_ylabel("Ap [nT]", color="#E57373", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#E57373")
    ax2.set_xlabel("Date (UTC)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax2.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    lt_tag = f"lt{lt_center:.1f}_hw{half_w:.1f}_{band_name}"
    savefig(fig, out_dir / f"02_{lt_tag}_timeseries")


def plot_multi_lt_overlay(df: pd.DataFrame, valid_bins: pd.DataFrame,
                           ev: dict, ev_key: str, sat: str,
                           out_dir: Path, half_w: float = 1.0) -> None:
    """
    図B: 複数LT帯を重ねた時系列（全緯度）
    """
    sub_bins = valid_bins[valid_bins["half_width"] == half_w].copy()
    if len(sub_bins) == 0:
        return

    # データ数の多い上位10 LT帯を使用
    sub_bins = sub_bins.nlargest(10, "min_obs_across_phases")
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(sub_bins)))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axvspan(ev["ssw_onset"], ev["ssw_end"], color="red", alpha=0.07, label="SSW期間")
    ax.axvline(ev["ssw_onset"], color="red",    ls="--", lw=1.5)
    ax.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5)
    ax.axhline(1.0, color="gray", ls=":", lw=1)

    for idx, (_, row) in enumerate(sub_bins.iterrows()):
        lt_c = row["lt_center"]
        daily = lt_fixed_timeseries(df, lt_c, half_w, ev, ("all", -60.0, 60.0))
        if daily is None:
            continue
        ax.plot(daily["date"], daily["ratio_med"],
                "o-", ms=3, lw=1.5, color=cmap[idx],
                label=f"LT={lt_c:.1f}h (N≥{int(row['min_obs_across_phases'])})")

    ax.set_ylabel(r"$\rho_{obs}/\rho_{MSIS}$ (daily median)", fontweight="bold")
    ax.set_xlabel("Date (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(f"{ev['label']} Swarm-{sat} — Multi-LT Overlay  ±{half_w}h bins",
                 fontweight="bold")
    savefig(fig, out_dir / f"03_multi_LT_overlay_hw{half_w:.1f}")


def plot_lt_date_2d(df: pd.DataFrame, ev: dict, ev_key: str, sat: str,
                    out_dir: Path) -> None:
    """
    図C: LT × date の 2D map（密度比）— フル解像度
    """
    lt_edges   = np.arange(0, 24.5, 0.5)
    date_edges = pd.date_range(ev["plot_start"].normalize(),
                               ev["plot_end"].normalize() + pd.Timedelta(days=1), freq="D")
    Z = np.full((len(lt_edges)-1, len(date_edges)-1), np.nan)

    df2 = df.copy()
    df2["lt_i"]   = np.digitize(df2[COLS["LT"]].values, lt_edges) - 1
    dt_secs       = df2[COLS["time"]].astype("int64").values // 10**9
    edge_secs     = date_edges.astype("int64").values // 10**9
    df2["date_i"] = np.digitize(dt_secs, edge_secs) - 1
    valid = ((df2["lt_i"]   >= 0) & (df2["lt_i"]   < len(lt_edges)-1) &
             (df2["date_i"] >= 0) & (df2["date_i"] < len(date_edges)-1))
    grp = df2[valid].groupby(["lt_i", "date_i"])[COLS["ratio"]].median()
    for (li, di), val in grp.items():
        Z[li, di] = val

    X = mdates.date2num(date_edges)
    Y = lt_edges
    fig, ax = plt.subplots(figsize=(14, 5))
    mesh = ax.pcolormesh(X, Y, Z, cmap="RdBu_r", vmin=0.6, vmax=1.4, shading="flat")
    plt.colorbar(mesh, ax=ax, pad=0.01, label=r"median $\rho_{obs}/\rho_{MSIS}$")
    ax.axvline(mdates.date2num(ev["ssw_onset"]), color="red",    ls="--", lw=2,
               label=f"SSW onset {ev['ssw_onset'].date()}")
    ax.axvline(mdates.date2num(ev["ssw_end"]),   color="orange", ls="--", lw=2,
               label=f"SSW end {ev['ssw_end'].date()}")
    ax.set_ylabel("Local Solar Time [h]")
    ax.set_yticks(range(0, 25, 3))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.set_xlabel("Date (UTC)")
    ax.legend(fontsize=9)
    ax.set_title(f"{ev['label']} Swarm-{sat} — Date×LT 2D (0.5h bins)  N={len(df):,}",
                 fontweight="bold")
    savefig(fig, out_dir / "04_LT_date_2D")


def compute_phase_stats(df: pd.DataFrame, lt_center: float, half_w: float,
                         ev: dict, ev_key: str, sat: str) -> dict | None:
    """
    SSW前・中・後のratio中央値差を計算してサマリ辞書を返す
    """
    daily = lt_fixed_timeseries(df, lt_center, half_w, ev, ("all", -60.0, 60.0))
    if daily is None or len(daily) == 0:
        return None

    result = {
        "event": ev_key, "sat": sat,
        "lt_center": lt_center, "half_width": half_w,
    }
    for phase in ["SSW前", "SSW中", "SSW後"]:
        g = daily[daily["phase"] == phase]
        result[f"ratio_med_{phase}"]  = g["ratio_med"].median()
        result[f"n_days_{phase}"]     = len(g)
        result[f"n_obs_{phase}"]      = g["n_obs"].sum()
        result[f"Ap_med_{phase}"]     = g["Ap_med"].median()

    # SSW期間の偏差
    pre = result["ratio_med_SSW前"]
    ssw = result["ratio_med_SSW中"]
    if np.isfinite(pre) and np.isfinite(ssw) and pre > 0:
        result["delta_ratio"]      = ssw - pre
        result["relative_change"]  = (ssw - pre) / pre * 100
    else:
        result["delta_ratio"]      = np.nan
        result["relative_change"]  = np.nan
    return result


# ──────────────────────────────────────────────
# PART 3: 衛星間比較
# ──────────────────────────────────────────────

def plot_satellite_comparison(all_sat_data: dict, lt_center: float, half_w: float,
                               ev: dict, ev_key: str, out_dir: Path) -> None:
    """
    図D: 同じLT帯での衛星A/B/C比較
    """
    SAT_COLORS = {"A": "#E53935", "B": "#43A047", "C": "#1E88E5"}
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axvspan(ev["ssw_onset"], ev["ssw_end"], color="red", alpha=0.07)
    ax.axvline(ev["ssw_onset"], color="red",    ls="--", lw=1.5)
    ax.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5)
    ax.axhline(1.0, color="gray", ls=":", lw=1)

    plotted = False
    for sat, daily in all_sat_data.items():
        if daily is None or len(daily) == 0:
            continue
        col = SAT_COLORS.get(sat, "black")
        ax.fill_between(daily["date"], daily["ratio_q25"], daily["ratio_q75"],
                        alpha=0.12, color=col)
        ax.plot(daily["date"], daily["ratio_med"], "o-", ms=4, lw=1.8,
                color=col, label=f"Swarm-{sat} (N={daily['n_obs'].sum():,})")
        plotted = True

    if not plotted:
        plt.close(fig)
        return
    ax.set_ylabel(r"$\rho_{obs}/\rho_{MSIS}$ (daily median)", fontweight="bold")
    ax.set_xlabel("Date (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(f"{ev['label']} — Satellite Comparison  LT={lt_center:.1f}h ±{half_w:.1f}h",
                 fontweight="bold")
    savefig(fig, out_dir / f"05_sat_comparison_lt{lt_center:.1f}_hw{half_w:.1f}")


def plot_latband_comparison(df: pd.DataFrame, lt_center: float, half_w: float,
                             ev: dict, ev_key: str, sat: str, out_dir: Path) -> None:
    """
    図E: 緯度帯別LT固定時系列
    """
    BAND_COLORS = {"all": "#555", "NH_hi": "#E53935", "EQ": "#FB8C00", "SH_hi": "#1E88E5"}
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axvspan(ev["ssw_onset"], ev["ssw_end"], color="red", alpha=0.07)
    ax.axvline(ev["ssw_onset"], color="red",    ls="--", lw=1.5)
    ax.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5)
    ax.axhline(1.0, color="gray", ls=":", lw=1)

    plotted = False
    for band in LAT_BANDS:
        daily = lt_fixed_timeseries(df, lt_center, half_w, ev, band)
        if daily is None or len(daily) < 5:
            continue
        col = BAND_COLORS.get(band[0], "gray")
        ax.plot(daily["date"], daily["ratio_med"], "o-", ms=4, lw=1.8,
                color=col, label=f"{band[0]} ({band[1]:.0f}°~{band[2]:.0f}°) N={daily['n_obs'].sum():,}")
        plotted = True

    if not plotted:
        plt.close(fig)
        return
    ax.set_ylabel(r"$\rho_{obs}/\rho_{MSIS}$ (daily median)", fontweight="bold")
    ax.set_xlabel("Date (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(f"{ev['label']} Swarm-{sat} — Lat-band Comparison  LT={lt_center:.1f}h ±{half_w:.1f}h",
                 fontweight="bold")
    savefig(fig, out_dir / f"06_latband_lt{lt_center:.1f}_hw{half_w:.1f}")


# ──────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────

def process_event(ev_key: str, ev: dict) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(f"EVENT: {ev_key}  ({ev['label']})")
    logger.info(f"{'='*60}")

    out_dir_ev = OUT_BASE / ev_key
    out_dir_ev.mkdir(parents=True, exist_ok=True)

    all_dfs = {}
    all_bins = {}

    # ─── 各衛星のLTカバレッジ分析 ─────────────────────────────────────────
    for sat, path in ev["sats"].items():
        logger.info(f"\n  --- Swarm-{sat} ---")
        out_dir = OUT_BASE / f"{ev_key}_{sat}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = load_data(path, ev)
        all_dfs[sat] = df

        bins_df = analyze_lt_coverage(df, ev, ev_key, sat, out_dir)
        all_bins[sat] = bins_df

        # 2D LT×date マップ（全データ）
        plot_lt_date_2d(df, ev, ev_key, sat, out_dir)

    # ─── 有効LT帯の選択（衛星間共通） ────────────────────────────────────
    # 最もデータが多いLT帯をhalf_wごとに上位3つ選択
    for sat, df in all_dfs.items():
        bins_df = all_bins[sat]
        if bins_df is None or len(bins_df) == 0:
            logger.warning(f"  Swarm-{sat}: No valid LT bins")
            continue

        out_dir = OUT_BASE / f"{ev_key}_{sat}"

        # ① 複数LT帯重ね描き（half_w=1.0を主解析）
        for hw in LT_HALF_WIDTHS:
            plot_multi_lt_overlay(df, bins_df, ev, ev_key, sat, out_dir, half_w=hw)

        # ② 各LT帯の単一時系列（データ数上位のもの）
        primary_hw = 1.0
        sub_bins = bins_df[bins_df["half_width"] == primary_hw]
        if len(sub_bins) == 0:
            primary_hw = 1.5
            sub_bins = bins_df[bins_df["half_width"] == primary_hw]

        # データ数上位5 LT帯の詳細図
        top_bins = sub_bins.nlargest(5, "min_obs_across_phases")
        for _, row in top_bins.iterrows():
            lt_c = row["lt_center"]
            # 全緯度帯
            daily = lt_fixed_timeseries(df, lt_c, primary_hw, ev, ("all", -60.0, 60.0))
            if daily is not None:
                plot_lt_fixed_single(daily, lt_c, primary_hw, ev, ev_key, sat,
                                      ("all", -60.0, 60.0), out_dir)
            # 緯度帯別
            plot_latband_comparison(df, lt_c, primary_hw, ev, ev_key, sat, out_dir)

        # ③ フェーズ別統計サマリ収集
        for _, row in bins_df.iterrows():
            stats = compute_phase_stats(df, row["lt_center"], row["half_width"],
                                         ev, ev_key, sat)
            if stats:
                ALL_SUMMARY.append(stats)

    # ─── 衛星間比較 ─────────────────────────────────────────────────────
    # 共通LT帯（全衛星で有効なもの）を探す
    common_lt = None
    if len(all_bins) >= 2:
        # 全衛星の有効 LT 中心値の共通部分
        sets = [set(zip(b["half_width"], b["lt_center"]))
                for b in all_bins.values() if b is not None and len(b) > 0]
        if sets:
            common_set = sets[0]
            for s in sets[1:]:
                common_set = common_set & s
            common_lt = sorted(common_set, key=lambda x: x[1])
            logger.info(f"  Common LT bins across satellites: {len(common_lt)}")

    if common_lt:
        primary_hw = 1.0
        common_primary = [(hw, lt) for hw, lt in common_lt if hw == primary_hw]
        if not common_primary:
            primary_hw = 1.5
            common_primary = [(hw, lt) for hw, lt in common_lt if hw == primary_hw]

        # データ数が多い共通LT帯を上位5つ選ぶ
        scored = []
        for hw, lt_c in common_primary:
            total = 0
            for sat, df in all_dfs.items():
                d = lt_fixed_timeseries(df, lt_c, hw, ev, ("all", -60.0, 60.0))
                if d is not None:
                    total += d["n_obs"].sum()
            scored.append((total, hw, lt_c))
        scored.sort(reverse=True)

        for total, hw, lt_c in scored[:5]:
            sat_dailies = {}
            for sat, df in all_dfs.items():
                sat_dailies[sat] = lt_fixed_timeseries(df, lt_c, hw, ev, ("all", -60.0, 60.0))
            plot_satellite_comparison(sat_dailies, lt_c, hw, ev, ev_key, out_dir_ev)


def main() -> None:
    for ev_key, ev in EVENTS.items():
        process_event(ev_key, ev)

    # 全サマリ保存
    if ALL_SUMMARY:
        summary_df = pd.DataFrame(ALL_SUMMARY)
        summary_df = summary_df.sort_values(["event", "sat", "half_width", "lt_center"])
        summary_df.to_csv(OUT_BASE / "STEP4_summary.csv", index=False, encoding="utf-8-sig")
        logger.info(f"\n  Saved: {OUT_BASE / 'STEP4_summary.csv'}")

        # 主要結果のコンソール表示
        print("\n" + "="*80)
        print("STEP 4 主要結果: SSW中のratio偏差 (half_width=1.0, 全緯度)")
        print("="*80)
        disp = summary_df[
            (summary_df["half_width"] == 1.0) &
            summary_df["delta_ratio"].notna()
        ].sort_values(["event", "sat", "delta_ratio"])
        print(disp[["event","sat","lt_center","ratio_med_SSW前","ratio_med_SSW中",
                     "delta_ratio","relative_change","Ap_med_SSW中"]
                   ].to_string(index=False))

    logger.info("\n✅ STEP 4 完了。Figure/STEP4/ を確認してください。")


if __name__ == "__main__":
    main()
