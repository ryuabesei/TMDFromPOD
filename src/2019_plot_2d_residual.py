# ==============================================================================
# 2019_plot_2d_residual.py
# 目的: 2019年SSW期間のSwarm A, B, Cについて、基準プロファイル（DOY 252〜254平均）
#       からの残差（2D分布: 緯度 × DoY）をプロットする。
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path

# =========================
# 共通設定
# =========================
T_START = "2019-09-09 00:00:00"
T_END   = "2019-09-23 23:59:59"
DOY_MIN = 252.0
DOY_MAX = 266.0

LAT_MIN, LAT_MAX = -60, 60
DOY_BIN  = 0.2
LAT_BIN  = 3.0
N_LEVELS = 20

# 基準期間 (DOY 252〜254: 最初の3日間)
DOY_REF_MIN = 252.0
DOY_REF_MAX = 254.0

NORM_COL = "density_norm"

# 衛星ごとの個別設定
SATELLITE_CONFIGS = {
    "A": dict(
        norm_parquet  = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_SSW.parquet"),
        out_png       = Path("Figure/2019/swarm_dnsapod_2019_2d_residual_SSW.png"),
        ref_alt_km    = 450.0,
        sector_morning = (3, 6),
        sector_evening = (15, 18),
        morning_wraps  = False,
        title         = "Swarm-A (ref: 450 km)"
    ),
    "B_510": dict(
        norm_parquet  = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_SSW.parquet"),
        out_png       = Path("Figure/2019/swarm_dnsbpod_2019_2d_residual_SSW.png"),
        ref_alt_km    = 510.0,
        sector_morning = (22, 5),  # 深夜またぎ
        sector_evening = (11, 14),
        morning_wraps  = True,
        title         = "Swarm-B (ref: 510 km)"
    ),
    "B_450": dict(
        norm_parquet  = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_SSW(450km).parquet"),
        out_png       = Path("Figure/2019/swarm_dnsbpod_2019_2d_residual_SSW_450km.png"),
        ref_alt_km    = 450.0,
        sector_morning = (22, 5),  # 深夜またぎ
        sector_evening = (11, 14),
        morning_wraps  = True,
        title         = "Swarm-B (ref: 450 km)"
    ),
    "C": dict(
        norm_parquet  = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_SSW.parquet"),
        out_png       = Path("Figure/2019/swarm_dnscpod_2019_2d_residual_SSW.png"),
        ref_alt_km    = 450.0,
        sector_morning = (3, 6),
        sector_evening = (15, 18),
        morning_wraps  = False,
        title         = "Swarm-C (ref: 450 km)"
    ),
}


# =========================
# ユーティリティ
# =========================
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


def ensure_required_columns(df: pd.DataFrame, required: list[str], parquet_path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{parquet_path} に必要列がありません: {missing}")


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


def compute_residual_grid(
    Z_full: np.ndarray,
    doy_bins: np.ndarray,
) -> np.ndarray:
    """
    基準期間（DOY 252〜254）の列を平均して緯度ごとの基準プロファイルを作成し、
    Z_full から差し引いて残差（Residual）を計算します。
    """
    doy_centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    ref_mask = (doy_centers >= DOY_REF_MIN) & (doy_centers <= DOY_REF_MAX)

    # 基準期間内の列を抽出して平均（NaNは無視）
    ref_cols = Z_full[:, ref_mask]
    ref_profile = np.nanmean(ref_cols, axis=1)  # 形状: (n_lat,)

    # 基準期間内に完全にNaNしかない緯度ビンがある場合、全体平均で補間
    if np.isnan(ref_profile).any():
        global_mean = np.nanmean(ref_profile)
        ref_profile = np.where(np.isnan(ref_profile), global_mean, ref_profile)

    Z_residual = Z_full - ref_profile[:, np.newaxis]
    return Z_residual


def daily_representative_lt_line(
    df: pd.DataFrame,
    lt_min: float,
    lt_max: float,
    stat: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    g = df[(df["lst_h"] >= lt_min) & (df["lst_h"] < lt_max)].copy()
    if len(g) == 0:
        return np.array([]), np.array([])

    g = g.set_index("datetime")
    if stat == "median":
        daily = g.resample("D")["lst_h"].median().dropna()
    elif stat == "mean":
        daily = g.resample("D")["lst_h"].mean().dropna()
    else:
        raise ValueError("stat は 'median' または 'mean' を指定してください。")

    if len(daily) == 0:
        return np.array([]), np.array([])

    x = daily.index.dayofyear.to_numpy() + 0.5
    y = daily.to_numpy()
    return x, y


def daily_representative_lt_line_wrap(
    df: pd.DataFrame,
    lt_start: float,
    lt_end: float,
    stat: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    g = df[(df["lst_h"] >= lt_start) | (df["lst_h"] < lt_end)].copy()
    if len(g) == 0:
        return np.array([]), np.array([])

    g["lst_h"] = g["lst_h"].where(g["lst_h"] >= lt_start, g["lst_h"] + 24)
    g = g.set_index("datetime")

    if stat == "median":
        daily = g.resample("D")["lst_h"].median().dropna()
    elif stat == "mean":
        daily = g.resample("D")["lst_h"].mean().dropna()
    else:
        raise ValueError("stat は 'median' または 'mean' を指定してください。")

    if len(daily) == 0:
        return np.array([]), np.array([])

    x = daily.index.dayofyear.to_numpy() + 0.5
    y = daily.to_numpy()
    return x, y


def load_and_prepare(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "datetime" not in df.columns:
        df = df.reset_index()
        if "datetime" in df.columns:
            pass
        elif "index" in df.columns:
            df = df.rename(columns={"index": "datetime"})
        else:
            df = df.rename(columns={df.columns[0]: "datetime"})

    ensure_required_columns(df, ["datetime", "lat", "lst_h", NORM_COL], parquet_path)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", NORM_COL]).copy()

    keep_cols = ["datetime", "lat", "lon", "lst_h", NORM_COL]
    df = df[keep_cols].copy()
    df = df.rename(columns={NORM_COL: "density_value"})

    t0 = pd.Timestamp(T_START, tz="UTC")
    t1 = pd.Timestamp(T_END, tz="UTC")
    df = df[(df["datetime"] >= t0) & (df["datetime"] <= t1)].copy()
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)].copy()

    df = add_doy(df)
    return df


def split_sectors(df: pd.DataFrame, morning_range: tuple, evening_range: tuple, wraps: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if wraps:
        df_m = df[(df["lst_h"] >= morning_range[0]) | (df["lst_h"] < morning_range[1])].copy()
    else:
        df_m = df[(df["lst_h"] >= morning_range[0]) & (df["lst_h"] < morning_range[1])].copy()
    df_e = df[(df["lst_h"] >= evening_range[0]) & (df["lst_h"] < evening_range[1])].copy()
    return df_m, df_e


def make_mesh_from_bins(
    doy_bins: np.ndarray, lat_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    doy_centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    return np.meshgrid(doy_centers, lat_centers)


def plot_residual_2d(cfg: dict) -> None:
    print(f"\nGenerating residual plot for {cfg['title']} ...")
    df = load_and_prepare(cfg["norm_parquet"])

    doy_bins = np.arange(DOY_MIN, DOY_MAX + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

    df_m, df_e = split_sectors(df, cfg["sector_morning"], cfg["sector_evening"], cfg["morning_wraps"])

    # 1. 2D グリッドの作成
    Z_norm_m = grid_median(df_m, doy_bins, lat_bins, "density_value")
    Z_norm_e = grid_median(df_e, doy_bins, lat_bins, "density_value")

    # 2. 残差の計算
    Z_res_m = compute_residual_grid(Z_norm_m, doy_bins)
    Z_res_e = compute_residual_grid(Z_norm_e, doy_bins)

    # 3. 対称カラースケールの決定
    all_res = np.concatenate([
        Z_res_m[np.isfinite(Z_res_m)].ravel(),
        Z_res_e[np.isfinite(Z_res_e)].ravel(),
    ])
    if len(all_res) == 0:
        print(f"⚠️ {cfg['title']} の残差データが空です。スキップします。")
        return

    vmax = float(np.nanpercentile(np.abs(all_res), 98))
    vmin = -vmax
    levels = np.linspace(vmin, vmax, N_LEVELS + 1)

    X, Y = make_mesh_from_bins(doy_bins, lat_bins)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        f"{cfg['title']} Residual Normalized Density (2019, DOY 252–266)\n"
        f"Reference: DOY {int(DOY_REF_MIN)}–{int(DOY_REF_MAX)} mean per latitude (baseline)",
        fontsize=13, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[1, 1, 0.05],
        wspace=0.08,
    )

    m_title_suffix = f"({cfg['sector_morning'][0]}–{cfg['sector_morning'][1]} LT)"
    if cfg["morning_wraps"]:
        m_title_suffix = f"({cfg['sector_morning'][0]}–0{cfg['sector_morning'][1]} LT)"
    e_title_suffix = f"({cfg['sector_evening'][0]}–{cfg['sector_evening'][1]} LT)"

    panels = [
        (0, Z_res_m, df_m, cfg["sector_morning"], f"Residual {m_title_suffix}"),
        (1, Z_res_e, df_e, cfg["sector_evening"], f"Residual {e_title_suffix}"),
    ]

    cf_last = None
    df_lookup = {0: df_m, 1: df_e}

    for col, Z_res, df_sec, sector, p_title in panels:
        ax = fig.add_subplot(gs[0, col])

        cf = ax.contourf(
            X, Y, Z_res,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
        )
        cf_last = cf

        # 基準期間（Reference Period）の網掛け表示
        ax.axvspan(DOY_REF_MIN, DOY_REF_MAX,
                   color="lightblue", alpha=0.20, lw=0, label="Baseline reference")

        ax.set_title(p_title, fontsize=11)
        ax.set_xlabel("Day of Year 2019 (DOY 252–266)", fontsize=10)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlim(DOY_MIN, DOY_MAX)
        ax.set_xticks(range(int(DOY_MIN), int(DOY_MAX) + 1, 2))
        ax.grid(alpha=0.2, color="white", linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Geographic Latitude (°)", fontsize=10)
            ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
        else:
            ax.tick_params(axis="y", labelleft=False)

        # 代表 LST 線
        ax_r = ax.twinx()
        ax_r.set_ylabel("LT (h)", fontsize=9)

        if sector == cfg["sector_morning"] and cfg["morning_wraps"]:
            ax_r.set_ylim(22, 29)
            ax_r.set_yticks([22, 23, 24, 25, 26, 27, 28, 29])
            ax_r.set_yticklabels(["22", "23", "0", "1", "2", "3", "4", "5"])
            x_lt, y_lt = daily_representative_lt_line_wrap(
                df_sec,
                lt_start=cfg["sector_morning"][0],
                lt_end=cfg["sector_morning"][1],
                stat="median",
            )
        elif sector == cfg["sector_morning"]:
            ax_r.set_ylim(*cfg["sector_morning"])
            ax_r.set_yticks(range(int(cfg["sector_morning"][0]), int(cfg["sector_morning"][1]) + 1, 1))
            x_lt, y_lt = daily_representative_lt_line(df_sec, sector[0], sector[1], stat="median")
        else:
            ax_r.set_ylim(*cfg["sector_evening"])
            ax_r.set_yticks(range(int(cfg["sector_evening"][0]), int(cfg["sector_evening"][1]) + 1, 1))
            x_lt, y_lt = daily_representative_lt_line(df_sec, sector[0], sector[1], stat="median")

        if len(x_lt) > 0:
            ax_r.plot(x_lt, y_lt, color="k", lw=1.2)

    # カラーバー
    cb_ax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(cf_last, cax=cb_ax)
    cbar.set_label(
        "Residual density [kg m$^{-3}$]\n(obs − ref)",
        fontsize=10,
    )

    cfg["out_png"].parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg["out_png"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved residual plot to: {cfg['out_png']}")


if __name__ == "__main__":
    for name, config in SATELLITE_CONFIGS.items():
        plot_residual_2d(config)
    print("\n✅ All 2019 2D residual plots generated successfully.")
