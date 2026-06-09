"""
NormalizationMISIS_FIXED.py — pymsis正規化の修正版（共通ライブラリ）

修正内容（旧版からの変更点）:
  1. f107s: 当日値 → 前日値 (pymsis仕様: "Daily F10.7 of the previous day")
  2. ap[5]: 単一12h前値 → 12〜33h前の8区間平均
     ap[6]: 単一15h前値 → 36〜57h前の8区間平均
     （前々日のAPデータも読み込んで計算）
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# 正規化の基準条件
# =========================
F107_REF   = 70.0
AP_REF     = 4.0
ALT_REF_KM = 450.0

USE_LAT_LIMIT   = True
LAT_LIMIT_DEG   = 60.0
USE_VALIDITY_FLAG = True
DEBUG_PRINT_MSIS_SHAPE = False


# =========================
# ユーティリティ
# =========================
def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"必要な列が見つかりません。候補={candidates} / 実際の列={list(df.columns)}")
    return None


def get_datetime_column(df: pd.DataFrame) -> pd.Series:
    time_col = pick_column(
        df,
        candidates=["datetime", "time", "epoch", "utc", "timestamp", "date_time"],
        required=False
    )
    if time_col is not None:
        t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        if t.isna().any():
            raise ValueError(f"時刻列 {time_col} に datetime 変換できない値が含まれています。")
        return t
    if isinstance(df.index, pd.DatetimeIndex):
        t = pd.to_datetime(df.index, utc=True, errors="coerce")
        if t.isna().any():
            raise ValueError("Index を datetime に変換できない値が含まれています。")
        return pd.Series(t, index=df.index, name="datetime")
    raise KeyError("Swarmデータに時刻が見つかりません。")


def check_no_missing(df: pd.DataFrame, cols: list[str]) -> None:
    bad = df[cols].isna().any()
    if bad.any():
        missing_cols = bad[bad].index.tolist()
        raise ValueError(f"欠損があるため処理を中断します。欠損列={missing_cols}")


# =========================
# 修正版 Ap7行列
# =========================
def build_ap7_matrix_fixed(df: pd.DataFrame) -> np.ndarray:
    """
    pymsis の ap 配列仕様（7要素）:
      ap[0]: Daily Ap（日平均）
      ap[1]: 現在の3時間Ap
      ap[2]: 3時間前
      ap[3]: 6時間前
      ap[4]: 9時間前
      ap[5]: 12〜33時間前の8区間平均   ← 旧版は単一12h前のみ（バグ）
      ap[6]: 36〜57時間前の8区間平均   ← 旧版は単一15h前のみ（バグ）

    今日のAP1..AP8: 0-3h, 3-6h, 6-9h, 9-12h, 12-15h, 15-18h, 18-21h, 21-24h UTC
    前日 (_prev) : 同様
    前々日 (_prev2): 同様（ap[6]の計算に必要）
    """
    hours = df["datetime"].dt.hour.to_numpy()
    k = (hours // 3).astype(int)  # 0..7

    ap_today = df[[f"AP{i}"       for i in range(1, 9)]].to_numpy(float)
    ap_prev  = df[[f"AP{i}_prev"  for i in range(1, 9)]].to_numpy(float)
    ap_prev2 = df[[f"AP{i}_prev2" for i in range(1, 9)]].to_numpy(float)
    ap_avg   = df["AP_AVG"].to_numpy(float)

    N    = len(df)
    idxN = np.arange(N)

    def get_ap_shift(shift: int) -> np.ndarray:
        """shift*3時間前の単一Ap区間値を返す（shift=0..7: 今日, 8..15: 前日, 16..23: 前々日）"""
        idx = k - shift
        # 今日側
        mask_today = idx >= 0
        val_today = ap_today[idxN, np.clip(idx, 0, 7)]
        # 前日側
        idx_prev = idx + 8
        mask_prev = (idx < 0) & (idx_prev >= 0)
        val_prev = ap_prev[idxN, np.clip(idx_prev, 0, 7)]
        # 前々日側
        idx_prev2 = idx + 16
        val_prev2 = ap_prev2[idxN, np.clip(idx_prev2, 0, 7)]

        result = np.where(mask_today, val_today,
                 np.where(mask_prev,  val_prev, val_prev2))
        return result

    # ap[0]〜ap[4]: 個別区間（旧版と同じ）
    ap_now = get_ap_shift(0)
    ap_m3  = get_ap_shift(1)
    ap_m6  = get_ap_shift(2)
    ap_m9  = get_ap_shift(3)

    # ap[5]: 12〜33h前の8区間平均（shifts 4〜11）
    ap5_vals = np.column_stack([get_ap_shift(s) for s in range(4, 12)])
    ap5 = np.nanmean(ap5_vals, axis=1)

    # ap[6]: 36〜57h前の8区間平均（shifts 12〜19）
    ap6_vals = np.column_stack([get_ap_shift(s) for s in range(12, 20)])
    ap6 = np.nanmean(ap6_vals, axis=1)

    ap7 = np.column_stack([ap_avg, ap_now, ap_m3, ap_m6, ap_m9, ap5, ap6]).astype(float)
    return ap7


# =========================
# MSIS密度計算
# =========================
def msis_density(time_utc, lat, lon, alt_km, f107s, f107as, aps) -> np.ndarray:
    from pymsis import msis
    out = msis.run(time_utc, lon, lat, alt_km,
                   f107s=f107s, f107as=f107as, aps=aps)
    out = np.asarray(out)
    if DEBUG_PRINT_MSIS_SHAPE:
        print("MSIS out shape:", out.shape)
    rho = out[:, 0].astype(float) if out.ndim >= 2 else out.astype(float)
    if np.any(rho <= 0) or np.any(~np.isfinite(rho)):
        raise ValueError("MSISが不正な密度を返しました。入力値を確認してください。")
    return rho


# =========================
# 地磁気データ読み込み（修正版）
# =========================
def load_geo_data(kpindex_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    戻り値: (df_geo当日, df_geo前日_prev, df_geo前々日_prev2)
    """
    ap_cols   = [f"AP{i}" for i in range(1, 9)]
    keep_cols = ["DATE", "F10.7_ADJ", "F10.7_ADJ_CENTER81", "AP_AVG", *ap_cols]
    df_geo = pd.read_csv(kpindex_csv, parse_dates=["DATE"])[keep_cols].copy()
    df_geo.rename(columns={"F10.7_ADJ": "F107", "F10.7_ADJ_CENTER81": "F107A"}, inplace=True)
    df_geo["DATE"] = pd.to_datetime(df_geo["DATE"], utc=True).dt.floor("D").dt.tz_localize(None)
    df_geo = df_geo.sort_values("DATE").reset_index(drop=True)

    # ✅ 修正1: f107s は前日値（shift -1）
    df_geo["F107_prev_day"] = df_geo["F107"].shift(1).fillna(df_geo["F107"])

    # 前日 AP データ（DATE+1日してmerge用）
    df_prev = df_geo[["DATE", "AP_AVG", *ap_cols]].copy()
    df_prev["DATE"] = df_prev["DATE"] + pd.Timedelta(days=1)
    df_prev = df_prev.rename(columns={c: f"{c}_prev" for c in ["AP_AVG", *ap_cols]})

    # 前々日 AP データ（DATE+2日してmerge用）
    df_prev2 = df_geo[["DATE", *ap_cols]].copy()
    df_prev2["DATE"] = df_prev2["DATE"] + pd.Timedelta(days=2)
    df_prev2 = df_prev2.rename(columns={c: f"{c}_prev2" for c in ap_cols})

    return df_geo, df_prev, df_prev2


# =========================
# メイン正規化関数
# =========================
def normalize(swarm_parquet: Path,
              kpindex_csv: Path,
              out_parquet: Path,
              alt_ref_km: float = ALT_REF_KM) -> None:

    print(f"\n{'='*60}")
    print(f"正規化開始: {swarm_parquet.name}")
    print(f"  基準高度: {alt_ref_km} km, F10.7_ref={F107_REF}, Ap_ref={AP_REF}")
    print(f"{'='*60}")

    # ---------- Swarmデータ読み込み ----------
    df_swarm = pd.read_parquet(swarm_parquet).copy()
    df_swarm["datetime"] = get_datetime_column(df_swarm)

    rho_col = pick_column(df_swarm, ["density", "rho", "dens", "rho_obs"])
    lat_col = pick_column(df_swarm, ["lat", "latitude", "geod_lat"])
    lon_col = pick_column(df_swarm, ["lon", "longitude", "geod_lon"])

    alt_m_col  = pick_column(df_swarm, ["altitude_m", "alt_m", "height_m"], required=False)
    alt_km_col = pick_column(df_swarm, ["altitude_km", "alt_km", "height_km"], required=False)
    if alt_km_col is None and alt_m_col is None:
        raise KeyError("高度列が見つかりません。")

    if alt_km_col is None:
        df_swarm["alt_km"] = df_swarm[alt_m_col].astype(float) / 1000.0
    else:
        df_swarm["alt_km"] = df_swarm[alt_km_col].astype(float)

    if USE_VALIDITY_FLAG and "validity_flag" in df_swarm.columns:
        df_swarm = df_swarm[df_swarm["validity_flag"] == 0].copy()
    if USE_LAT_LIMIT:
        df_swarm = df_swarm[df_swarm[lat_col].abs() <= LAT_LIMIT_DEG].copy()

    print(f"  Swarm行数: {len(df_swarm):,}")

    # ---------- 地磁気データ読み込み ----------
    df_geo, df_prev, df_prev2 = load_geo_data(kpindex_csv)
    ap_cols = [f"AP{i}" for i in range(1, 9)]

    # ---------- マージ ----------
    df_swarm["DATE"] = df_swarm["datetime"].dt.floor("D").dt.tz_localize(None)
    df = df_swarm.merge(df_geo, on="DATE", how="left")
    df = df.merge(df_prev,  on="DATE", how="left")
    df = df.merge(df_prev2, on="DATE", how="left")

    # 前日・前々日APが欠ける場合の代用
    for i in range(1, 9):
        c = f"AP{i}"
        df[f"{c}_prev"].fillna(df[c], inplace=True)
        df[f"{c}_prev2"].fillna(df[f"{c}_prev"], inplace=True)

    check_no_missing(df,
        ["F107", "F107_prev_day", "F107A", "AP_AVG"] +
        ap_cols +
        [f"{c}_prev"  for c in ap_cols] +
        [f"{c}_prev2" for c in ap_cols]
    )

    # ---------- MSIS入力用配列 ----------
    rho_obs  = df[rho_col].astype(float).to_numpy()
    lat      = df[lat_col].astype(float).to_numpy()
    lon      = df[lon_col].astype(float).to_numpy()
    alt_km_v = df["alt_km"].astype(float).to_numpy()
    time_utc = pd.to_datetime(df["datetime"], utc=True).to_numpy()

    # ✅ 修正1: f107s = 前日のF10.7
    f107s_real  = df["F107_prev_day"].astype(float).to_numpy()
    f107as_real = df["F107A"].astype(float).to_numpy()

    # ✅ 修正2: ap[5,6] = 正しい区間平均
    aps_real = build_ap7_matrix_fixed(df)

    # ---------- 基準条件 ----------
    f107s_ref  = np.full_like(f107s_real,  F107_REF, dtype=float)
    f107as_ref = np.full_like(f107as_real, F107_REF, dtype=float)
    aps_ref    = np.full_like(aps_real,    AP_REF,   dtype=float)
    alt_ref_arr = np.full_like(alt_km_v,  alt_ref_km, dtype=float)

    # ---------- MSIS計算 ----------
    print("  MSISモデル計算中 (real cond)...")
    rho_model_real = msis_density(
        time_utc, lat, lon, alt_km_v,
        f107s=f107s_real, f107as=f107as_real, aps=aps_real
    )
    print("  MSISモデル計算中 (ref cond)...")
    rho_model_ref = msis_density(
        time_utc, lat, lon, alt_ref_arr,
        f107s=f107s_ref, f107as=f107as_ref, aps=aps_ref
    )

    # ---------- 正規化 ----------
    ratio = rho_model_ref / rho_model_real
    df["density_norm"] = rho_obs * ratio
    df["norm_ratio_model_ref_over_real"] = ratio
    df["norm_ref_alt_km"]  = alt_ref_km
    df["norm_ref_F107"]    = F107_REF
    df["norm_ref_AP"]      = AP_REF

    # ---------- 出力 ----------
    ensure_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    print(f"  ✅ 保存完了: {out_parquet}")
    print(f"  行数: {len(df):,}  |  ratio: {ratio.min():.3f}〜{ratio.max():.3f}")
    print(f"  density_norm: {df['density_norm'].describe().to_dict()}")
