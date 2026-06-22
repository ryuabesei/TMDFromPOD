# ==============================================================================
# 2019_plot_density_before_after_normalization.py
# 目的: 2019年SSW期間のSwarm A, B, Cについて正規化前後の密度分布（緯度 × DoY）をプロットする。
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
DOY_BIN  = 0.2  # 15日間のため、0.2日間刻みで細かく表示
LAT_BIN  = 3.0
N_LEVELS = 20

RAW_COL  = "density"
NORM_COL = "density_norm"

# 衛星ごとの個別設定
SATELLITE_CONFIGS = {
    "A": dict(
        raw_parquet   = Path("integrateddata/2019/swarm_dnsapod_2019_SSW.parquet"),
        norm_parquet  = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_SSW.parquet"),
        out_png       = Path("Figure/2019/swarm_dnsapod_2019_before_after_normalization_SSW.png"),
        ref_alt_km    = 450.0,
        sector_morning = (3, 6),
        sector_evening = (15, 18),
        morning_wraps  = False,
        title         = "Swarm-A (ref: 450 km)"
    ),
    "B_510": dict(
        raw_parquet   = Path("integrateddata/2019/swarm_dnsbpod_2019_SSW.parquet"),
        norm_parquet  = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_SSW.parquet"),
        out_png       = Path("Figure/2019/swarm_dnsbpod_2019_before_after_normalization_SSW.png"),
        ref_alt_km    = 510.0,
        sector_morning = (22, 5),  # 深夜またぎ
        sector_evening = (11, 14),
        morning_wraps  = True,
        title         = "Swarm-B (ref: 510 km)"
    ),
    "B_450": dict(
        raw_parquet   = Path("integrateddata/2019/swarm_dnsbpod_2019_SSW.parquet"),
        norm_parquet  = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_SSW(450km).parquet"),
        out_png       = Path("Figure/2019/swarm_dnsbpod_2019_before_after_normalization_SSW_450km.png"),
        ref_alt_km    = 450.0,
        sector_morning = (22, 5),  # 深夜またぎ
        sector_evening = (11, 14),
        morning_wraps  = True,
        title         = "Swarm-B (ref: 450 km)"
    ),
    "C": dict(
        raw_parquet   = Path("integrateddata/2019/swarm_dnscpod_2019_SSW.parquet"),
        norm_parquet  = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_SSW.parquet"),
        out_png       = Path("Figure/2019/swarm_dnscpod_2019_before_after_normalization_SSW.png"),
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


def load_and_prepare(parquet_path: Path, density_col: str) -> pd.DataFrame:
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

    ensure_required_columns(df, ["datetime", "lat", "lst_h", density_col], parquet_path)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lst_h", density_col]).copy()

    keep_cols = ["datetime", "lat", "lon", "lst_h", density_col]
    df = df[keep_cols].copy()
    df = df.rename(columns={density_col: "density_value"})

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


def row_vmin_vmax(*grids: np.ndarray) -> tuple[float, float]:
    arrs = [g[np.isfinite(g)].ravel() for g in grids if np.isfinite(g).any()]
    if not arrs:
        raise ValueError("有効なグリッド値がありません。")
    vals = np.concatenate(arrs)
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    if vmin == vmax:
        eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps
    return vmin, vmax


def make_levels(vmin: float, vmax: float, n_levels: int) -> np.ndarray:
    return np.linspace(vmin, vmax, n_levels + 1)


def make_mesh_from_bins(
    doy_bins: np.ndarray, lat_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    doy_centers = 0.5 * (doy_bins[:-1] + doy_bins[1:])
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    return np.meshgrid(doy_centers, lat_centers)


def plot_before_after(cfg: dict) -> None:
    print(f"\nGenerating plot for {cfg['title']} ...")
    df_raw  = load_and_prepare(cfg["raw_parquet"], RAW_COL)
    df_norm = load_and_prepare(cfg["norm_parquet"], NORM_COL)

    doy_bins = np.arange(DOY_MIN, DOY_MAX + DOY_BIN, DOY_BIN)
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_BIN, LAT_BIN)

    df_raw_m, df_raw_e   = split_sectors(df_raw, cfg["sector_morning"], cfg["sector_evening"], cfg["morning_wraps"])
    df_norm_m, df_norm_e = split_sectors(df_norm, cfg["sector_morning"], cfg["sector_evening"], cfg["morning_wraps"])

    Z = {
        "raw_m":  grid_median(df_raw_m, doy_bins, lat_bins, "density_value"),
        "raw_e":  grid_median(df_raw_e, doy_bins, lat_bins, "density_value"),
        "norm_m": grid_median(df_norm_m, doy_bins, lat_bins, "density_value"),
        "norm_e": grid_median(df_norm_e, doy_bins, lat_bins, "density_value"),
    }

    vmin_raw, vmax_raw   = row_vmin_vmax(Z["raw_m"], Z["raw_e"])
    vmin_norm, vmax_norm = row_vmin_vmax(Z["norm_m"], Z["norm_e"])

    levels_raw  = make_levels(vmin_raw, vmax_raw, N_LEVELS)
    levels_norm = make_levels(vmin_norm, vmax_norm, N_LEVELS)

    X, Y = make_mesh_from_bins(doy_bins, lat_bins)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"{cfg['title']} Thermospheric Mass Density (2019, DOY 252–266)\n"
        "Before Normalization (top) vs After Normalization (bottom)",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1, 1, 0.05],
        hspace=0.30,
        wspace=0.08,
    )

    m_title_suffix = f"({cfg['sector_morning'][0]}–{cfg['sector_morning'][1]} LT)"
    if cfg["morning_wraps"]:
        m_title_suffix = f"({cfg['sector_morning'][0]}–0{cfg['sector_morning'][1]} LT)"
    e_title_suffix = f"({cfg['sector_evening'][0]}–{cfg['sector_evening'][1]} LT)"

    panels = [
        (0, 0, "raw_m",  levels_raw,  f"Before norm. {m_title_suffix}",  cfg["sector_morning"]),
        (0, 1, "raw_e",  levels_raw,  f"Before norm. {e_title_suffix}",  cfg["sector_evening"]),
        (1, 0, "norm_m", levels_norm, f"After norm. {m_title_suffix}",   cfg["sector_morning"]),
        (1, 1, "norm_e", levels_norm, f"After norm. {e_title_suffix}",   cfg["sector_evening"]),
    ]

    df_lookup = {
        "raw_m": df_raw_m, "raw_e": df_raw_e,
        "norm_m": df_norm_m, "norm_e": df_norm_e,
    }
    cf_lookup = {}

    for row, col, key, levels, p_title, sector in panels:
        ax = fig.add_subplot(gs[row, col])

        cf = ax.contourf(
            X, Y, Z[key],
            levels=levels,
            cmap="turbo",
            extend="both",
        )
        cf_lookup[(row, col)] = cf

        ax.set_title(p_title, fontsize=11)
        ax.set_xlabel("Day of Year 2019 (DOY 252–266)", fontsize=10)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlim(DOY_MIN, DOY_MAX)
        ax.set_xticks(range(int(DOY_MIN), int(DOY_MAX) + 1, 2))
        ax.grid(alpha=0.2, color="white", linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Geographic Latitude (°)", fontsize=10)
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
                df_lookup[key],
                lt_start=cfg["sector_morning"][0],
                lt_end=cfg["sector_morning"][1],
                stat="median",
            )
        elif sector == cfg["sector_morning"]:
            ax_r.set_ylim(*cfg["sector_morning"])
            ax_r.set_yticks(range(int(cfg["sector_morning"][0]), int(cfg["sector_morning"][1]) + 1, 1))
            x_lt, y_lt = daily_representative_lt_line(
                df_lookup[key],
                lt_min=cfg["sector_morning"][0],
                lt_max=cfg["sector_morning"][1],
                stat="median",
            )
        else:
            ax_r.set_ylim(*cfg["sector_evening"])
            ax_r.set_yticks(range(int(cfg["sector_evening"][0]), int(cfg["sector_evening"][1]) + 1, 1))
            x_lt, y_lt = daily_representative_lt_line(
                df_lookup[key],
                lt_min=cfg["sector_evening"][0],
                lt_max=cfg["sector_evening"][1],
                stat="median",
            )

        if len(x_lt) > 0:
            ax_r.plot(x_lt, y_lt, color="k", lw=1.2)

    cb_ax_raw  = fig.add_subplot(gs[0, 2])
    cb_ax_norm = fig.add_subplot(gs[1, 2])

    fig.colorbar(cf_lookup[(0, 1)], cax=cb_ax_raw).set_label(
        "Density [kg m$^{-3}$]", fontsize=10
    )
    fig.colorbar(cf_lookup[(1, 1)], cax=cb_ax_norm).set_label(
        f"Normalized density [kg m$^{{-3}}$]\n(ref: {cfg['ref_alt_km']} km, F10.7=70, Ap=4)",
        fontsize=10
    )

    cfg["out_png"].parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg["out_png"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved plot to: {cfg['out_png']}")


if __name__ == "__main__":
    for name, config in SATELLITE_CONFIGS.items():
        plot_before_after(config)
    print("\n✅ All 2019 before-after plots generated successfully.")
