"""
plot_2d_ratio_and_residual_2018_DOY30-65.py

Purpose:
    Plot 2D (DOY vs Latitude) maps of:
      Top   (Row 0): rho_ratio (density_ratio_msis = rho_obs / rho_MSIS_real) [cmap="turbo"]
      Bottom (Row 1): delta_ratio (residual ratio vs non-SSW reference) [cmap="RdBu_r"]
    for SWARM-A, B, C during 2018 SSW period (DOY 30-65).

    Non-SSW reference: DOY 30-40 & DOY 61-65
    SSW peak period  : DOY 41-60

    LT sectors per satellite (with representative LT lines):
      SWARM-A/C: Morning (04–11 LT) / Evening (16–23 LT)
      SWARM-B:   Nightside (22–05 LT, wrapped) / Dayside (11–17 LT)

Output:
    Figure/2018/2D_ratio_and_residual_2018_SWARM-A_DOY30-65.png
    Figure/2018/2D_ratio_and_residual_2018_SWARM-B_DOY30-65.png
    Figure/2018/2D_ratio_and_residual_2018_SWARM-C_DOY30-65.png
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # 非対話型バックエンドで表示ブロックを回避

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# Settings
# ============================================================
SATELLITES = [
    dict(
        label         = "SWARM-A",
        parquet       = Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        out_png       = Path("Figure/2018/2D_ratio_and_residual_2018_SWARM-A_DOY30-65.png"),
        sec1          = (4, 11),
        sec2          = (16, 23),
        sec1_wrap     = False,
        sec1_title    = "Morning (04–11 LT)",
        sec2_title    = "Evening (16–23 LT)"
    ),
    dict(
        label         = "SWARM-B",
        parquet       = Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        out_png       = Path("Figure/2018/2D_ratio_and_residual_2018_SWARM-B_DOY30-65.png"),
        sec1          = (22, 5),
        sec2          = (11, 17),
        sec1_wrap     = True,
        sec1_title    = "Nightside (22–05 LT)",
        sec2_title    = "Dayside (11–17 LT)"
    ),
    dict(
        label         = "SWARM-C",
        parquet       = Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        out_png       = Path("Figure/2018/2D_ratio_and_residual_2018_SWARM-C_DOY30-65.png"),
        sec1          = (4, 11),
        sec2          = (16, 23),
        sec1_wrap     = False,
        sec1_title    = "Morning (04–11 LT)",
        sec2_title    = "Evening (16–23 LT)"
    ),
]

# DOY 30-65 に絞ったデータ期間
T_START = "2018-01-30 00:00:00"
T_END   = "2018-03-06 23:59:59"

LAT_MIN, LAT_MAX = -60, 60
DOY_MIN, DOY_MAX = 30, 65
DOY_BIN  = 0.5
LAT_BIN  = 3.0
N_LEVELS = 21

# 非SSW参照期間
DOY_REF1 = (30, 40)
DOY_REF2 = (61, 65)
DOY_SSW_START, DOY_SSW_END = 41, 60

VALUE_COL = "density_ratio_msis"


# ============================================================
# Utilities
# ============================================================
def add_doy(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    out = df.copy()
    out["DOY"] = (
        dt.dt.dayofyear
        + dt.dt.hour / 24.0
        + dt.dt.minute / 1440.0
        + dt.dt.second / 86400.0
    )
    return out


def grid_median(
    df: pd.DataFrame,
    doy_bins: np.ndarray,
    lat_bins: np.ndarray,
    value_col: str,
) -> np.ndarray:
    Z = np.full((len(lat_bins) - 1, len(doy_bins) - 1), np.nan)
    if len(df) == 0:
        return Z

    doy = df["DOY"].to_numpy()
    lat = df["lat"].to_numpy()
    val = df[value_col].to_numpy()

    ok = np.isfinite(doy) & np.isfinite(lat) & np.isfinite(val)
    doy, lat, val = doy[ok], lat[ok], val[ok]

    doy_i = np.digitize(doy, doy_bins) - 1
    lat_i = np.digitize(lat, lat_bins) - 1

    ok = (
        (doy_i >= 0) & (doy_i < len(doy_bins) - 1)
        & (lat_i >= 0) & (lat_i < len(lat_bins) - 1)
    )
    doy_i, lat_i, val = doy_i[ok], lat_i[ok], val[ok]

    bucket = defaultdict(list)
    for i, j, v in zip(lat_i, doy_i, val):
        bucket[(i, j)].append(float(v))

    for (i, j), arr in bucket.items():
        Z[i, j] = float(np.median(arr))

    return Z


def compute_residual_grid(Z_full: np.ndarray, doy_bins: np.ndarray) -> np.ndarray:
    doy_centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    ref_mask = (
        ((doy_centers >= DOY_REF1[0]) & (doy_centers <= DOY_REF1[1])) |
        ((doy_centers >= DOY_REF2[0]) & (doy_centers <= DOY_REF2[1]))
    )
    ref_cols = Z_full[:, ref_mask]
    ref_profile = np.nanmedian(ref_cols, axis=1)  # (n_lat,)
    Z_residual = Z_full - ref_profile[:, np.newaxis]
    return Z_residual


def daily_representative_lt_line(
    df: pd.DataFrame,
    lt_min: float,
    lt_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    g = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)].copy()
    if len(g) == 0:
        return np.array([]), np.array([])
    g = g.set_index("datetime")
    daily = g.resample("D")["lst_h"].median().dropna()
    if len(daily) == 0:
        return np.array([]), np.array([])
    x = daily.index.dayofyear.to_numpy() + 0.5
    y = daily.to_numpy()
    return x, y


def daily_representative_lt_line_wrap(
    df: pd.DataFrame,
    lt_start: float,
    lt_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    g = df[(df["lst_h"] >= lt_start) | (df["lst_h"] < lt_end)].copy()
    if len(g) == 0:
        return np.array([]), np.array([])
    g["lst_h"] = g["lst_h"].where(g["lst_h"] >= lt_start, g["lst_h"] + 24)
    g = g.set_index("datetime")
    daily = g.resample("D")["lst_h"].median().dropna()
    if len(daily) == 0:
        return np.array([]), np.array([])
    x = daily.index.dayofyear.to_numpy() + 0.5
    y = daily.to_numpy()
    return x, y


def load_and_prepare(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "datetime" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "datetime":
            df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    lat_col = next((c for c in ["lat", "latitude", "geod_lat"] if c in df.columns), None)
    if lat_col is None:
        raise KeyError("Latitude column not found")
    if lat_col != "lat":
        df = df.rename(columns={lat_col: "lat"})

    df = df.dropna(subset=["datetime", "lat", "lst_h", VALUE_COL]).copy()
    t0 = pd.Timestamp(T_START, tz="UTC")
    t1 = pd.Timestamp(T_END, tz="UTC")
    df = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].copy()
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)].copy()
    df = add_doy(df)
    df = df[(df["DOY"] >= DOY_MIN) & (df["DOY"] <= DOY_MAX)].copy()
    return df


# ============================================================
# Plot Job
# ============================================================
def plot_satellite(sat: dict) -> None:
    label      = sat["label"]
    parquet    = sat["parquet"]
    out_png    = sat["out_png"]
    sec1       = sat["sec1"]
    sec2       = sat["sec2"]
    sec1_wrap  = sat["sec1_wrap"]
    sec1_title = sat["sec1_title"]
    sec2_title = sat["sec2_title"]

    print(f"\n=== {label} ===")
    df = load_and_prepare(parquet)
    print(f"  Total records: {len(df):,}")

    doy_bins = np.arange(DOY_MIN, DOY_MAX + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)
    X, Y = np.meshgrid(0.5 * (doy_bins[:-1] + doy_bins[1:]),
                       0.5 * (lat_bins[:-1] + lat_bins[1:]))

    # LTセクターごとのフィルタリング
    if sec1_wrap:
        df_sec1 = df[(df["lst_h"] >= sec1[0]) | (df["lst_h"] < sec1[1])].copy()
    else:
        df_sec1 = df[(df["lst_h"] >= sec1[0]) & (df["lst_h"] < sec1[1])].copy()
    df_sec2 = df[(df["lst_h"] >= sec2[0]) & (df["lst_h"] < sec2[1])].copy()

    # グリッドメディアン
    Z_ratio_1 = grid_median(df_sec1, doy_bins, lat_bins, VALUE_COL)
    Z_ratio_2 = grid_median(df_sec2, doy_bins, lat_bins, VALUE_COL)

    # 残差グリッド
    Z_resid_1 = compute_residual_grid(Z_ratio_1, doy_bins)
    Z_resid_2 = compute_residual_grid(Z_ratio_2, doy_bins)

    # カラー上限下限
    all_ratio = np.concatenate([
        Z_ratio_1[np.isfinite(Z_ratio_1)],
        Z_ratio_2[np.isfinite(Z_ratio_2)]
    ])
    vmax_r = float(np.nanpercentile(all_ratio, 99))
    vmin_r = float(np.nanpercentile(all_ratio, 1))
    levels_ratio = np.linspace(vmin_r, vmax_r, N_LEVELS + 1)

    all_res = np.concatenate([
        Z_resid_1[np.isfinite(Z_resid_1)],
        Z_resid_2[np.isfinite(Z_resid_2)]
    ])
    vmax_d = float(np.nanpercentile(np.abs(all_res), 98))
    vmin_d = -vmax_d
    levels_resid = np.linspace(vmin_d, vmax_d, N_LEVELS + 1)

    # ── プロット作成 (2行×2列) ────────────────────────────────
    fig = plt.figure(figsize=(15, 10))

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           width_ratios=[1, 1, 0.04],
                           height_ratios=[1, 1],
                           wspace=0.08, hspace=0.18)

    # 共通の装飾関数
    def decorate_ax(ax: plt.Axes, title: str, ylabel: bool = False) -> None:
        # 非SSW参照期間
        ax.axvspan(*DOY_REF1, color="lightblue", alpha=0.20, lw=0, label="Non-SSW ref (DOY 30-40)")
        ax.axvspan(*DOY_REF2, color="lightblue", alpha=0.20, lw=0, label="Non-SSW ref (DOY 61-65)")
        # SSWピーク期間
        ax.axvspan(DOY_SSW_START, DOY_SSW_END,
                   color="lightyellow", alpha=0.30, lw=0, label="SSW period (DOY 41-60)")
        # SSW境界線
        ax.axvline(DOY_SSW_START, color="red", lw=1.2, ls="--", alpha=0.7)
        ax.axvline(DOY_SSW_END,   color="red", lw=1.2, ls="--", alpha=0.7)
        ax.set_xlim(DOY_MIN, DOY_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xticks(range(DOY_MIN, DOY_MAX + 1, 5))
        ax.grid(alpha=0.2, color="white", linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        if ylabel:
            ax.set_ylabel("Geographic Latitude (°)", fontsize=11, fontweight="bold")
        else:
            ax.tick_params(axis="y", labelleft=False)

    # LT線のオーバーレイ関数
    def overlay_lt_line(ax: plt.Axes, df_sec: pd.DataFrame,
                        sec: tuple[float, float], wrap: bool) -> None:
        ax_r = ax.twinx()
        ax_r.set_ylabel("Local Time (h)", fontsize=10, color="k")
        if wrap:
            ax_r.set_ylim(sec[0], sec[0] + 7)
            ax_r.set_yticks([22, 23, 24, 25, 26, 27, 28, 29])
            ax_r.set_yticklabels(["22", "23", "00", "01", "02", "03", "04", "05"])
            x_lt, y_lt = daily_representative_lt_line_wrap(df_sec, sec[0], sec[1])
        else:
            ax_r.set_ylim(sec[0], sec[1])
            ax_r.set_yticks(np.linspace(sec[0], sec[1], 4, dtype=int))
            x_lt, y_lt = daily_representative_lt_line(df_sec, sec[0], sec[1])

        if len(x_lt) > 0:
            ax_r.plot(x_lt, y_lt, color="black", lw=1.5, ls="-")

    # ── Row 0: rho_ratio ── (cmap="turbo")
    ax00 = fig.add_subplot(gs[0, 0])
    cf00 = ax00.contourf(X, Y, Z_ratio_1, levels=levels_ratio, cmap="turbo", extend="both")
    decorate_ax(ax00, f"rho_ratio — {sec1_title}", ylabel=True)
    overlay_lt_line(ax00, df_sec1, sec1, sec1_wrap)
    ax00.legend(loc="upper left", fontsize=7, framealpha=0.7)

    ax01 = fig.add_subplot(gs[0, 1])
    cf01 = ax01.contourf(X, Y, Z_ratio_2, levels=levels_ratio, cmap="turbo", extend="both")
    decorate_ax(ax01, f"rho_ratio — {sec2_title}")
    overlay_lt_line(ax01, df_sec2, sec2, False)

    cb_ax0 = fig.add_subplot(gs[0, 2])
    cbar0 = fig.colorbar(cf01, cax=cb_ax0)
    cbar0.set_label("rho_ratio\n(obs / MSIS)", fontsize=10, fontweight="bold")

    # ── Row 1: delta_ratio (Residual) ──
    ax10 = fig.add_subplot(gs[1, 0])
    cf10 = ax10.contourf(X, Y, Z_resid_1, levels=levels_resid, cmap="RdBu_r", extend="both")
    decorate_ax(ax10, f"delta_ratio — {sec1_title}", ylabel=True)
    overlay_lt_line(ax10, df_sec1, sec1, sec1_wrap)
    ax10.set_xlabel(f"Day of Year 2018 (DOY {DOY_MIN}-{DOY_MAX})", fontsize=11, fontweight="bold")

    ax11 = fig.add_subplot(gs[1, 1])
    cf11 = ax11.contourf(X, Y, Z_resid_2, levels=levels_resid, cmap="RdBu_r", extend="both")
    decorate_ax(ax11, f"delta_ratio — {sec2_title}")
    overlay_lt_line(ax11, df_sec2, sec2, False)
    ax11.set_xlabel(f"Day of Year 2018 (DOY {DOY_MIN}-{DOY_MAX})", fontsize=11, fontweight="bold")

    cb_ax1 = fig.add_subplot(gs[1, 2])
    cbar1 = fig.colorbar(cf11, cax=cb_ax1)
    cbar1.set_label("delta_ratio (Residual)\n(ratio − non-SSW ref)", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"{label}  density_ratio_msis (DOY {DOY_MIN}–{DOY_MAX}, 2018)\n"
        "Top: rho_ratio (rho_obs/rho_MSIS_real)  |  Bottom: delta_ratio (Residual)\n"
        f"Non-SSW ref: DOY {DOY_REF1[0]}–{DOY_REF1[1]} & DOY {DOY_REF2[0]}–{DOY_REF2[1]}   "
        f"SSW period: DOY {DOY_SSW_START}–{DOY_SSW_END}",
        fontsize=13, fontweight="bold", y=0.98
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    for sat in SATELLITES:
        plot_satellite(sat)
    print("\n✅ All 2D ratio and residual plots (DOY 30-65) completed with turbo cmap.")


if __name__ == "__main__":
    main()
