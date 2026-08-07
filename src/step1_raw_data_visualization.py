"""
step1_raw_data_visualization.py
================================
STEP 1: 生データと観測条件の可視化

目的:
  各SSWイベント・各衛星について観測条件（LT, 高度, 緯度, 経度）と
  密度比 density_ratio_msis の変化を可視化し、
  SSW期間の密度変動がLT・高度・Apと同時変化していないかを診断する。

出力:
  Figure/STEP1/{event}_{sat}/   以下にPNG・PDF
  Figure/STEP1/{event}_{sat}/stats_by_phase.csv   統計サマリ

科学的意義:
  SSW信号と競合する以下の仮説を診断する：
  - 仮説3: LTドリフト・高度変化・Apによる見かけの変動
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
# 設定（列名・期間を一箇所にまとめる）
# ──────────────────────────────────────────────

COLS = {
    "time":    "datetime",
    "lat":     "lat",
    "lon":     "lon",
    "alt_km":  "alt_km",
    "LT":      "lst_h",
    "rho_obs": "density",
    "rho_msis":"rho_model_real",
    "ratio":   "density_ratio_msis",
    "Ap":      "AP_AVG",
    "F107":    "F107",
    "F107A":   "F107A",
}

# SSWイベント定義
EVENTS = {
    "2018_NH": {
        "label":    "2018 NH SSW",
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
        "label":    "2019 SH SSW",
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
        "label":    "2021 NH SSW",
        "ssw_onset":  pd.Timestamp("2021-01-04", tz="UTC"),
        "ssw_end":    pd.Timestamp("2021-01-16", tz="UTC"),
        "plot_start": pd.Timestamp("2020-12-25", tz="UTC"),
        "plot_end":   pd.Timestamp("2021-02-05", tz="UTC"),
        "sats": {
            "C": "normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet",
        },
    },
}

OUT_BASE = Path("Figure/STEP1")
OUT_BASE.mkdir(parents=True, exist_ok=True)
RATIO_VALID_MIN, RATIO_VALID_MAX = 0.1, 3.0

# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────

def savefig(fig, path):
    fig.savefig(str(path) + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf",           bbox_inches="tight")
    logger.info(f"  Saved: {str(path)}.png")
    plt.close(fig)


def phase_label(t, onset, end, plot_start, plot_end):
    labels = pd.Categorical(
        ["SSW中" if (onset <= ts <= end) else
         ("SSW前" if ts < onset else "SSW後")
         for ts in t],
        categories=["SSW前", "SSW中", "SSW後"],
        ordered=True,
    )
    return labels


def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2)
                 / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def load_data(path, ev):
    df = pd.read_parquet(path)
    tc = COLS["time"]
    df[tc] = pd.to_datetime(df[tc], utc=True, errors="coerce")
    df = df.dropna(subset=[tc])
    df = df[(df[tc] >= ev["plot_start"]) & (df[tc] <= ev["plot_end"])].copy()
    n_raw = len(df)

    ratio_col = COLS["ratio"]
    if ratio_col in df.columns:
        n_before = len(df)
        df = df[(df[ratio_col] >= RATIO_VALID_MIN) & (df[ratio_col] <= RATIO_VALID_MAX)].copy()
        n_after = len(df)
        if n_before != n_after:
            logger.warning(f"  異常値除去: {n_before} -> {n_after} 行 (ratio範囲外: {n_before - n_after}行)")

    df["phase"] = phase_label(df[tc], ev["ssw_onset"], ev["ssw_end"],
                              ev["plot_start"], ev["plot_end"])
    df["date"] = df[tc].dt.normalize()
    logger.info(f"  データ: {path}")
    logger.info(f"  期間内: {n_raw:,} -> フィルタ後: {len(df):,}")
    return df


# ──────────────────────────────────────────────
# プロット関数群
# ──────────────────────────────────────────────

def plot_date_lt(df, ev, sat, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"SSW前": "#2196F3", "SSW中": "#F44336", "SSW後": "#4CAF50"}
    for phase, grp in df.groupby("phase"):
        ax.scatter(grp[COLS["time"]], grp[COLS["LT"]],
                   c=colors[phase], s=0.5, alpha=0.3, label=phase, rasterized=True)
    ax.axvline(ev["ssw_onset"], color="red", ls="--", lw=1.5, label=f"SSW onset ({ev['ssw_onset'].date()})")
    ax.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5, label=f"SSW end ({ev['ssw_end'].date()})")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Local Solar Time [h]")
    ax.set_ylim(0, 24)
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.legend(loc="upper right", markerscale=8, fontsize=9)
    ax.set_title(f"{ev['label']} Swarm-{sat} — Date vs Local Solar Time  N={len(df):,}", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    savefig(fig, out_dir / "01_date_vs_LT")


def plot_date_alt(df, ev, sat, out_dir):
    daily_med = df.groupby("date")[COLS["alt_km"]].median()
    daily_q25 = df.groupby("date")[COLS["alt_km"]].quantile(0.25)
    daily_q75 = df.groupby("date")[COLS["alt_km"]].quantile(0.75)
    dates = pd.to_datetime(daily_med.index)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, daily_q25.values, daily_q75.values, alpha=0.3, color="#1976D2", label="IQR")
    ax.plot(dates, daily_med.values, color="#1976D2", lw=1.5, label="Median altitude")
    ax.axvspan(ev["ssw_onset"], ev["ssw_end"], color="red", alpha=0.1, label="SSW期間")
    ax.axvline(ev["ssw_onset"], color="red", ls="--", lw=1.5)
    ax.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5)
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Altitude [km]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.legend(fontsize=9)
    ax.set_title(f"{ev['label']} Swarm-{sat} — Daily Altitude Distribution  N={len(df):,}", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    savefig(fig, out_dir / "02_date_vs_altitude")


def plot_date_ratio(df, ev, sat, out_dir):
    daily = df.groupby("date").agg(
        ratio_med=(COLS["ratio"], "median"),
        ratio_q25=(COLS["ratio"], lambda x: x.quantile(0.25)),
        ratio_q75=(COLS["ratio"], lambda x: x.quantile(0.75)),
        Ap_mean=(COLS["Ap"], "mean"),
        F107_mean=(COLS["F107"], "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    dates = pd.to_datetime(daily["date"])
    ax1.fill_between(dates, daily["ratio_q25"], daily["ratio_q75"], alpha=0.3, color="#1976D2")
    ax1.plot(dates, daily["ratio_med"], "o-", color="#1976D2", lw=1.8, ms=4,
             label=r"$\rho_{obs}/\rho_{MSIS}$ (median+IQR)")
    ax1.axvspan(ev["ssw_onset"], ev["ssw_end"], color="red", alpha=0.08, label="SSW期間")
    ax1.axvline(ev["ssw_onset"], color="red", ls="--", lw=1.5)
    ax1.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5)
    ax1.axhline(1.0, color="gray", ls=":", lw=1)
    ax1.set_ylabel(r"$\rho_{obs}/\rho_{MSIS}$", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, ls=":", alpha=0.4)
    n_days = len(daily)
    ax1.set_title(f"{ev['label']} Swarm-{sat} — Density Ratio & Geomagnetic Activity\n"
                  f"N={len(df):,} obs, {n_days} days", fontweight="bold")

    ax2_f = ax2.twinx()
    ax2.bar(dates, daily["Ap_mean"], color="#E57373", width=0.8, alpha=0.7, label="Ap (日平均)")
    ax2_f.plot(dates, daily["F107_mean"], "^-", color="#1565C0", lw=1.5, ms=3.5, label="F10.7")
    ax2.axvspan(ev["ssw_onset"], ev["ssw_end"], color="red", alpha=0.08)
    ax2.axvline(ev["ssw_onset"], color="red", ls="--", lw=1.5)
    ax2.axvline(ev["ssw_end"],   color="orange", ls="--", lw=1.5)
    ax2.set_ylabel("Ap Index [nT]", color="#E57373", fontweight="bold")
    ax2_f.set_ylabel("F10.7 [sfu]", color="#1565C0", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#E57373")
    ax2_f.tick_params(axis="y", labelcolor="#1565C0")
    ax2.set_xlabel("Date (UTC)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2_f.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8.5)
    ax2.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    savefig(fig, out_dir / "03_date_vs_ratio_Ap_F107")


def plot_lt_ratio(df, ev, sat, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"SSW前": "#2196F3", "SSW中": "#F44336", "SSW後": "#4CAF50"}
    for phase, grp in df.groupby("phase"):
        ax.scatter(grp[COLS["LT"]], grp[COLS["ratio"]],
                   c=colors[phase], s=0.8, alpha=0.3, label=phase, rasterized=True)
    # LTビン中央値
    lt_bins = np.arange(0, 24.5, 1.0)
    lt_mids = 0.5 * (lt_bins[:-1] + lt_bins[1:])
    for phase, grp in df.groupby("phase"):
        ratio_med = []
        for lo, hi in zip(lt_bins[:-1], lt_bins[1:]):
            sub = grp[(grp[COLS["LT"]] >= lo) & (grp[COLS["LT"]] < hi)][COLS["ratio"]]
            ratio_med.append(sub.median() if len(sub) >= 5 else np.nan)
        ax.plot(lt_mids, ratio_med, color=colors[phase], lw=2.0, marker="o", ms=5)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Local Solar Time [h]")
    ax.set_ylabel(r"$\rho_{obs}/\rho_{MSIS}$")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.legend(markerscale=4, fontsize=9, loc="upper right")
    ax.set_title(f"{ev['label']} Swarm-{sat} — LT vs Density Ratio by Phase  N={len(df):,}", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)
    savefig(fig, out_dir / "04_LT_vs_ratio")


def plot_ap_ratio(df, ev, sat, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"SSW前": "#2196F3", "SSW中": "#F44336", "SSW後": "#4CAF50"}
    for ax, xcol, xlabel in zip(axes, [COLS["Ap"], COLS["F107"]], ["Ap (日平均) [nT]", "F10.7 [sfu]"]):
        for phase, grp in df.groupby("phase"):
            ax.scatter(grp[xcol], grp[COLS["ratio"]],
                       c=colors[phase], s=0.8, alpha=0.3, label=phase, rasterized=True)
        non_ssw = df[df["phase"] != "SSW中"]
        if len(non_ssw) > 10:
            x = non_ssw[xcol].values
            y = non_ssw[COLS["ratio"]].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 2:
                slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
                xr = np.array([x[mask].min(), x[mask].max()])
                ax.plot(xr, slope * xr + intercept, "k--", lw=1.5,
                        label=f"非SSW回帰 r={r:.3f}")
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\rho_{obs}/\rho_{MSIS}$")
        ax.legend(markerscale=4, fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].set_title(f"{ev['label']} Swarm-{sat} — Ap vs Ratio", fontweight="bold")
    axes[1].set_title(f"{ev['label']} Swarm-{sat} — F10.7 vs Ratio", fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "05_Ap_F107_vs_ratio")


def plot_alt_ratio(df, ev, sat, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"SSW前": "#2196F3", "SSW中": "#F44336", "SSW後": "#4CAF50"}
    for phase, grp in df.groupby("phase"):
        ax.scatter(grp[COLS["alt_km"]], grp[COLS["ratio"]],
                   c=colors[phase], s=0.8, alpha=0.3, label=phase, rasterized=True)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel(r"$\rho_{obs}/\rho_{MSIS}$")
    ax.legend(markerscale=4, fontsize=9)
    ax.set_title(f"{ev['label']} Swarm-{sat} — Altitude vs Ratio  N={len(df):,}", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)
    savefig(fig, out_dir / "06_altitude_vs_ratio")


def plot_date_lt_2d(df, ev, sat, out_dir):
    lt_edges = np.arange(0, 25, 1.0)
    date_edges = pd.date_range(ev["plot_start"].normalize(),
                               ev["plot_end"].normalize() + pd.Timedelta(days=1), freq="D")
    Z = np.full((len(lt_edges)-1, len(date_edges)-1), np.nan)
    df2 = df.copy()
    df2["lt_i"] = np.digitize(df2[COLS["LT"]].values, lt_edges) - 1
    dt_secs   = df2[COLS["time"]].astype("int64").values // 10**9
    edge_secs = date_edges.astype("int64").values // 10**9
    df2["date_i"] = np.digitize(dt_secs, edge_secs) - 1
    valid = ((df2["lt_i"] >= 0) & (df2["lt_i"] < len(lt_edges)-1) &
             (df2["date_i"] >= 0) & (df2["date_i"] < len(date_edges)-1))
    grp = df2[valid].groupby(["lt_i", "date_i"])[COLS["ratio"]].median()
    for (li, di), val in grp.items():
        Z[li, di] = val

    X = mdates.date2num(date_edges)
    Y = lt_edges
    fig, ax = plt.subplots(figsize=(13, 5))
    mesh = ax.pcolormesh(X, Y, Z, cmap="RdBu_r", vmin=0.6, vmax=1.4, shading="flat")
    plt.colorbar(mesh, ax=ax, pad=0.01, label=r"median $\rho_{obs}/\rho_{MSIS}$")
    ax.axvline(mdates.date2num(ev["ssw_onset"]), color="red", ls="--", lw=2,
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
    ax.set_title(f"{ev['label']} Swarm-{sat} — Date vs LT 2D (median ratio)  N={len(df):,}", fontweight="bold")
    savefig(fig, out_dir / "07_date_LT_2D_ratio")


def phase_statistics(df, ev, sat, out_dir):
    records = []
    for phase in ["SSW前", "SSW中", "SSW後"]:
        sub = df[df["phase"] == phase]
        if len(sub) == 0:
            continue
        rec = {
            "event": f"{ev['label']}_Swarm-{sat}",
            "phase": phase,
            "n_obs": len(sub),
            "LT_median": sub[COLS["LT"]].median(),
            "LT_q25":    sub[COLS["LT"]].quantile(0.25),
            "LT_q75":    sub[COLS["LT"]].quantile(0.75),
            "LT_min":    sub[COLS["LT"]].min(),
            "LT_max":    sub[COLS["LT"]].max(),
            "alt_median": sub[COLS["alt_km"]].median(),
            "alt_q25":    sub[COLS["alt_km"]].quantile(0.25),
            "alt_q75":    sub[COLS["alt_km"]].quantile(0.75),
            "ratio_median": sub[COLS["ratio"]].median(),
            "ratio_q25":    sub[COLS["ratio"]].quantile(0.25),
            "ratio_q75":    sub[COLS["ratio"]].quantile(0.75),
            "Ap_median":    sub[COLS["Ap"]].median(),
            "F107_median":  sub[COLS["F107"]].median(),
        }
        records.append(rec)
    stats_df = pd.DataFrame(records)

    grp_pre = df[df["phase"] == "SSW前"]
    grp_ssw = df[df["phase"] == "SSW中"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    colors_map = {"SSW前": "#2196F3", "SSW中": "#F44336", "SSW後": "#4CAF50"}
    for ax, col_key, ylabel in zip(axes,
                                    [COLS["LT"], COLS["ratio"], COLS["alt_km"]],
                                    ["Local Solar Time [h]",
                                     r"$\rho_{obs}/\rho_{MSIS}$",
                                     "Altitude [km]"]):
        data_list = [df[df["phase"] == p][col_key].dropna().values
                     for p in ["SSW前", "SSW中", "SSW後"]]
        labels = ["SSW前", "SSW中", "SSW後"]
        parts = ax.violinplot(data_list, positions=[1, 2, 3],
                              showmedians=True, showextrema=False)
        for pc, phase in zip(parts["bodies"], labels):
            pc.set_facecolor(colors_map[phase])
            pc.set_alpha(0.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, ls=":", alpha=0.4)
        ylo, yhi = ax.get_ylim()
        for xi, d in enumerate(data_list, 1):
            ax.text(xi, ylo + 0.02 * (yhi - ylo), f"N={len(d):,}", ha="center", fontsize=7.5)
        # KS + Cohen's d
        if len(grp_pre) > 0 and len(grp_ssw) > 0:
            a = grp_pre[col_key].dropna().values
            b = grp_ssw[col_key].dropna().values
            if len(a) > 0 and len(b) > 0:
                ks_stat, ks_p = stats.ks_2samp(a, b)
                d_val = cohen_d(a, b)
                ax.text(0.03, 0.97, f"KS p={ks_p:.3f}\nCohen's d={d_val:.3f}",
                        transform=ax.transAxes, va="top", fontsize=8,
                        bbox=dict(fc="white", ec="gray", alpha=0.8))
                logger.info(f"  [{col_key}] KS={ks_stat:.4f} p={ks_p:.4f} Cohen d={d_val:.4f}")
    axes[0].set_title(f"{ev['label']} Swarm-{sat}", fontweight="bold")
    axes[1].set_title(f"密度比分布", fontweight="bold")
    axes[2].set_title(f"高度分布", fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "08_phase_distribution_violin")

    stats_df.to_csv(out_dir / "stats_by_phase.csv", index=False, encoding="utf-8-sig")
    logger.info(f"  Saved: {out_dir / 'stats_by_phase.csv'}")
    return stats_df


# ──────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────

def process_event_satellite(ev_key, sat, path, ev):
    out_dir = OUT_BASE / f"{ev_key}_{sat}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing: {ev_key} Swarm-{sat}")
    logger.info(f"{'='*50}")
    df = load_data(path, ev)
    if len(df) == 0:
        logger.warning(f"  データなし: {ev_key} Swarm-{sat}")
        return
    plot_date_lt(df, ev, sat, out_dir)
    plot_date_alt(df, ev, sat, out_dir)
    plot_date_ratio(df, ev, sat, out_dir)
    plot_lt_ratio(df, ev, sat, out_dir)
    plot_ap_ratio(df, ev, sat, out_dir)
    plot_alt_ratio(df, ev, sat, out_dir)
    plot_date_lt_2d(df, ev, sat, out_dir)
    stats_df = phase_statistics(df, ev, sat, out_dir)
    print(f"\n  === {ev_key} Swarm-{sat} フェーズ別統計 ===")
    print(stats_df.to_string(index=False))


def main():
    for ev_key, ev in EVENTS.items():
        for sat, path in ev["sats"].items():
            process_event_satellite(ev_key, sat, path, ev)
    logger.info("\nSTEP 1 完了。Figure/STEP1/ を確認してください。")


if __name__ == "__main__":
    main()
