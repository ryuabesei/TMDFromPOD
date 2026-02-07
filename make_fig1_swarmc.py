"""
Swarm-C (normalized density) で Liu et al. Figure 1 相当（等高線版）を作る

出力:
- figures/swarmc_fig1_like_contour.png

入力:
- normalizeddata/swarm_dnscpod_2018_normalized.parquet
- data/Kpindex/SW-All.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 入出力
# =========================
IN_PARQUET = Path("normalizeddata/swarm_dnscpod_2018_normalized.parquet")
KP_CSV     = Path("data/Kpindex/SW-All.csv")
OUT_PNG    = Path("figures/swarmc_fig1_like_contour.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 設定（ここだけ必要に応じて調整）
# =========================
# LTセクター（空白が多いなら幅を広げる）
# 例: 16–18 を 15–19 にしている（先行研究のセクター概念に寄せる）
AFT_LT = (12.0, 22.0)  # afternoon sector
PRE_LT = ( 2.0,  8.0)  # pre-dawn sector

# 緯度範囲
LAT_MIN, LAT_MAX = -70.0, 70.0

# 2Dビン幅
LAT_BIN_DEG = 2.0       # 緯度ビン（deg）
DOY_BIN_DAY = 0.25      # DoYビン（day）= 6時間

# 赤道域平均
EQ_LAT = 30.0

# 等高線のレベル数（多いほど細かい）
N_LEVELS = 12

# 平滑化（等高線を滑らかにする：0で無効、3 or 5 が無難）
SMOOTH_K = 3

# 欠け領域（NaN）をどこまで補間するか（False推奨：データの無い所を塗らない）
FILL_NAN = False


# =========================
# ユーティリティ
# =========================
def add_doy(df: pd.DataFrame) -> pd.DataFrame:
    """
    役割: datetime(UTC) から DoY（小数日）を作る
    """
    dt = pd.to_datetime(df["datetime"], utc=True)
    doy = dt.dt.dayofyear + (dt.dt.hour + dt.dt.minute/60 + dt.dt.second/3600)/24.0
    out = df.copy()
    out["doy"] = doy
    return out


def filter_lt(df: pd.DataFrame, lt_range: tuple[float, float]) -> pd.DataFrame:
    """
    役割: 指定LT範囲だけ抽出
    """
    a, b = lt_range
    return df[(df["lst_h"] >= a) & (df["lst_h"] < b)].copy()


def smooth_2d_nan(Z: np.ndarray, k: int = 3) -> np.ndarray:
    """
    役割: NaNを保持しつつ移動平均で平滑化（等高線を滑らかに見せる）
    - k は奇数（3 or 5 推奨）
    """
    if k <= 1:
        return Z

    Z = Z.copy()
    mask = np.isfinite(Z).astype(float)
    Z0 = np.nan_to_num(Z, nan=0.0)

    pad = k // 2
    Zp = np.pad(Z0, pad, mode="edge")
    Mp = np.pad(mask, pad, mode="edge")

    out = np.zeros_like(Z0, dtype=float)
    wout = np.zeros_like(Z0, dtype=float)

    for i in range(k):
        for j in range(k):
            out += Zp[i:i+Z0.shape[0], j:j+Z0.shape[1]]
            wout += Mp[i:i+Z0.shape[0], j:j+Z0.shape[1]]

    out = np.where(wout > 0, out / wout, np.nan)
    return out


def make_binned_grid(df: pd.DataFrame, value_col: str = "density_norm"):
    """
    役割:
    - (doy, lat) をビン平均して 2Dグリッド（lat × doy）を作る
    - 併せて、そのビンの平均LTも作る（右軸の参考線用）
    """
    # ビン端（edges）作成
    doy_min = np.floor(df["doy"].min() / DOY_BIN_DAY) * DOY_BIN_DAY
    doy_max = np.ceil (df["doy"].max() / DOY_BIN_DAY) * DOY_BIN_DAY
    doy_edges = np.arange(doy_min, doy_max + DOY_BIN_DAY, DOY_BIN_DAY)

    lat_edges = np.arange(LAT_MIN, LAT_MAX + LAT_BIN_DEG, LAT_BIN_DEG)

    # どのビンに入るか（0..n-1）
    doy_bin = np.digitize(df["doy"].to_numpy(), doy_edges) - 1
    lat_bin = np.digitize(df["lat"].to_numpy(), lat_edges) - 1

    ok = (
        (doy_bin >= 0) & (doy_bin < len(doy_edges)-1) &
        (lat_bin >= 0) & (lat_bin < len(lat_edges)-1)
    )
    df2 = df.loc[ok].copy()
    doy_bin = doy_bin[ok]
    lat_bin = lat_bin[ok]

    tmp = pd.DataFrame({
        "doy_bin": doy_bin,
        "lat_bin": lat_bin,
        "val": df2[value_col].to_numpy(float),
        "lst": df2["lst_h"].to_numpy(float),
    })

    g = tmp.groupby(["lat_bin", "doy_bin"], as_index=False).mean()

    # Z: density grid
    Z = np.full((len(lat_edges)-1, len(doy_edges)-1), np.nan, dtype=float)
    Z[g["lat_bin"].to_numpy(int), g["doy_bin"].to_numpy(int)] = g["val"].to_numpy(float)

    # LTgrid: mean LT per bin（参考）
    LTgrid = np.full_like(Z, np.nan)
    LTgrid[g["lat_bin"].to_numpy(int), g["doy_bin"].to_numpy(int)] = g["lst"].to_numpy(float)

    return doy_edges, lat_edges, Z, LTgrid


def equatorial_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    役割:
    - 赤道域（|lat|<=30）で平均した density_norm の時系列（DoYビン平均）
    """
    df_eq = df[df["lat"].abs() <= EQ_LAT].copy()

    # DoYビン
    doy_edges = np.arange(
        np.floor(df_eq["doy"].min()/DOY_BIN_DAY)*DOY_BIN_DAY,
        np.ceil (df_eq["doy"].max()/DOY_BIN_DAY)*DOY_BIN_DAY + DOY_BIN_DAY,
        DOY_BIN_DAY
    )
    doy_bin = np.digitize(df_eq["doy"].to_numpy(), doy_edges) - 1
    ok = (doy_bin >= 0) & (doy_bin < len(doy_edges)-1)
    df_eq = df_eq.loc[ok].copy()
    df_eq["doy_bin"] = doy_bin[ok]

    ts = df_eq.groupby("doy_bin", as_index=False)["density_norm"].mean()
    ts["doy_center"] = doy_edges[ts["doy_bin"].to_numpy(int)] + DOY_BIN_DAY/2
    return ts[["doy_center", "density_norm"]]


def load_kp_as_daily_doy() -> pd.DataFrame:
    """
    役割:
    - SW-All.csv から日平均Kpを作って DoY に変換する
    - KP1..KP8 は 0–90 のような形式（例: 27=2.7）なので /10 する
    """
    dfk = pd.read_csv(KP_CSV, parse_dates=["DATE"])
    kp_cols = [f"KP{i}" for i in range(1, 9)]
    for c in kp_cols:
        if c not in dfk.columns:
            raise KeyError(f"{c} が SW-All.csv に見つかりません。")

    kp_3h = dfk[kp_cols].astype(float) / 10.0
    dfk["kp_daily_mean"] = kp_3h.mean(axis=1)

    dt = pd.to_datetime(dfk["DATE"], utc=True)
    dfk["doy"] = dt.dt.dayofyear.astype(float)

    return dfk[["doy", "kp_daily_mean"]]


def make_common_levels(Z_list: list[np.ndarray], n_levels: int = 12) -> np.ndarray:
    """
    役割:
    - 左右比較しやすいように、複数Zから共通の等高線レベルを作る
    - 外れ値に引っ張られないよう 5–95% を使用
    """
    vals = np.concatenate([Z[np.isfinite(Z)] for Z in Z_list if np.isfinite(Z).any()])
    if len(vals) == 0:
        raise ValueError("等高線レベルを作るための有効データがありません。")
    vmin = np.percentile(vals, 5)
    vmax = np.percentile(vals, 95)
    if vmin == vmax:
        vmax = vmin * 1.001
    return np.linspace(vmin, vmax, n_levels)


def mean_lt_track(df_sector: pd.DataFrame) -> pd.DataFrame:
    """
    役割:
    - そのLTセクター内での「平均LT」を DoYビンごとに求める（右軸の参考線用）
    注意:
    - これは“軌道トラックのLT”を厳密再現するものではなく、
      セクター内に入ったデータの平均LTの時間変化を示す。
    """
    doy_edges = np.arange(
        np.floor(df_sector["doy"].min()/DOY_BIN_DAY)*DOY_BIN_DAY,
        np.ceil (df_sector["doy"].max()/DOY_BIN_DAY)*DOY_BIN_DAY + DOY_BIN_DAY,
        DOY_BIN_DAY
    )
    doy_bin = np.digitize(df_sector["doy"].to_numpy(), doy_edges) - 1
    ok = (doy_bin >= 0) & (doy_bin < len(doy_edges)-1)
    tmp = df_sector.loc[ok, ["lst_h"]].copy()
    tmp["doy_bin"] = doy_bin[ok]
    lt = tmp.groupby("doy_bin", as_index=False)["lst_h"].mean()
    lt["doy_center"] = doy_edges[lt["doy_bin"].to_numpy(int)] + DOY_BIN_DAY/2
    return lt[["doy_center", "lst_h"]]


# =========================
# メイン
# =========================
def main():
    # ---------- 入力 ----------
    if not IN_PARQUET.exists():
        raise FileNotFoundError(f"入力parquetが見つかりません: {IN_PARQUET}")
    if not KP_CSV.exists():
        raise FileNotFoundError(f"SW-All.csvが見つかりません: {KP_CSV}")

    df = pd.read_parquet(IN_PARQUET)

    need = ["datetime", "lat", "lst_h", "density_norm"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"必要な列がありません: {miss} / cols={list(df.columns)}")

    # 緯度範囲を揃える
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)].copy()

    # DoY付与
    df = add_doy(df)

    # ---------- LTセクター抽出 ----------
    df_aft = filter_lt(df, AFT_LT)
    df_pre = filter_lt(df, PRE_LT)

    if len(df_aft) == 0 or len(df_pre) == 0:
        raise ValueError("LTセクター抽出後のデータが空です。LT幅を広げるか期間を見直してください。")

    # ---------- 2Dグリッド ----------
    doy_e_a, lat_e_a, Z_a, LT_a = make_binned_grid(df_aft)
    doy_e_p, lat_e_p, Z_p, LT_p = make_binned_grid(df_pre)

    # 平滑化（等高線を滑らかに）
    if SMOOTH_K and SMOOTH_K > 1:
        Z_a_s = smooth_2d_nan(Z_a, k=SMOOTH_K)
        Z_p_s = smooth_2d_nan(Z_p, k=SMOOTH_K)
    else:
        Z_a_s = Z_a
        Z_p_s = Z_p

    # NaNを埋めるか（おすすめはFalse：データが無いところを塗らない）
    if FILL_NAN:
        # 役割: 欠けを0埋めなどは危険なので、ここでは“最近傍っぽい”埋めはしない
        # 必要なら別途、補間手法を明示して採用すること
        pass

    # 共通レベル
    levels = make_common_levels([Z_a_s, Z_p_s], n_levels=N_LEVELS)

    # 2D座標（中心）
    doy_c_a = (doy_e_a[:-1] + doy_e_a[1:]) / 2
    lat_c_a = (lat_e_a[:-1] + lat_e_a[1:]) / 2
    X_a, Y_a = np.meshgrid(doy_c_a, lat_c_a)

    doy_c_p = (doy_e_p[:-1] + doy_e_p[1:]) / 2
    lat_c_p = (lat_e_p[:-1] + lat_e_p[1:]) / 2
    X_p, Y_p = np.meshgrid(doy_c_p, lat_c_p)

    # ---------- 赤道域平均 ----------
    ts_a = equatorial_timeseries(df_aft)
    ts_p = equatorial_timeseries(df_pre)

    # ---------- Kp ----------
    df_kp = load_kp_as_daily_doy()

    # ---------- LT参考線 ----------
    lt_a = mean_lt_track(df_aft)
    lt_p = mean_lt_track(df_pre)

    # ---------- 図 ----------
    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 1, 3)

    # --- 上段左（afternoon）: 等高線 ---
    cf1 = ax1.contourf(X_a, Y_a, Z_a_s, levels=levels)
    ax1.contour(X_a, Y_a, Z_a_s, levels=levels, linewidths=0.6)
    ax1.set_title(f"Swarm-C normalized density ({AFT_LT[0]:.0f}–{AFT_LT[1]:.0f} LT)")
    ax1.set_xlabel("DoY (2018)")
    ax1.set_ylabel("GLAT (deg)")
    c1 = fig.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
    c1.set_label("density_norm [kg m$^{-3}$]")

    # 右軸：平均LT（参考）
    ax1r = ax1.twinx()
    ax1r.plot(lt_a["doy_center"], lt_a["lst_h"], color="k", linewidth=1.0)
    ax1r.set_ylabel("LT (h)")
    ax1r.set_ylim(AFT_LT[0], AFT_LT[1])

    # --- 上段右（pre-dawn）: 等高線 ---
    cf2 = ax2.contourf(X_p, Y_p, Z_p_s, levels=levels)
    ax2.contour(X_p, Y_p, Z_p_s, levels=levels, linewidths=0.6)
    ax2.set_title(f"Swarm-C normalized density ({PRE_LT[0]:.0f}–{PRE_LT[1]:.0f} LT)")
    ax2.set_xlabel("DoY (2018)")
    ax2.set_ylabel("GLAT (deg)")
    c2 = fig.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
    c2.set_label("density_norm [kg m$^{-3}$]")

    ax2r = ax2.twinx()
    ax2r.plot(lt_p["doy_center"], lt_p["lst_h"], color="k", linewidth=1.0)
    ax2r.set_ylabel("LT (h)")
    ax2r.set_ylim(PRE_LT[0], PRE_LT[1])

    # --- 中段：赤道域平均 ---
    ax3.plot(ts_a["doy_center"], ts_a["density_norm"], marker="o", markersize=2, linewidth=1)
    ax3.set_title(f"Equatorial mean (|lat|<=30): {AFT_LT[0]:.0f}–{AFT_LT[1]:.0f} LT")
    ax3.set_xlabel("DoY (2018)")
    ax3.set_ylabel("density_norm [kg m$^{-3}$]")

    ax4.plot(ts_p["doy_center"], ts_p["density_norm"], marker="o", markersize=2, linewidth=1)
    ax4.set_title(f"Equatorial mean (|lat|<=30): {PRE_LT[0]:.0f}–{PRE_LT[1]:.0f} LT")
    ax4.set_xlabel("DoY (2018)")
    ax4.set_ylabel("density_norm [kg m$^{-3}$]")

    # --- 下段：Kp ---
    ax5.bar(df_kp["doy"], df_kp["kp_daily_mean"], width=0.8)
    ax5.set_title("Kp (daily mean from SW-All.csv)")
    ax5.set_xlabel("DoY (2018)")
    ax5.set_ylabel("Kp")

    # x範囲を揃える（図の見栄え）
    doy_min = float(np.floor(df["doy"].min()))
    doy_max = float(np.ceil(df["doy"].max()))
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim(doy_min, doy_max)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f"✅ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
