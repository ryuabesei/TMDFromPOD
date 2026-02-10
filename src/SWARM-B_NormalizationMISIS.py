"""
NormalizationMISIS.py（pymsis 0.12.0 対応・決定版）

目的:
- Swarm DNSBPOD の観測密度 rho_obs を
  「高度変動」「太陽活動(F10.7)」「地磁気活動(Ap)」の影響が最小になるように
  基準条件へ正規化する。

正規化の式（先行研究と同じ“モデル比”）:
  rho_norm = rho_obs * ( rho_model(ref_cond) / rho_model(real_cond) )

ここで
  real_cond: 観測時刻・観測位置・観測高度 + その日の F10.7 / Ap
  ref_cond : 観測時刻・観測位置・基準高度 + F10.7=70 / Kp=1相当(Ap≈4)

入力:
- integrateddata/swarm_dnsbpod_2018.parquet
- data/Kpindex/SW-All.csv

出力:
- normalizeddata/swarm_dnsbpod_2018_normalized.parquet
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# 入出力パス（ユーザー指定）
# =========================
SWARM_PARQUET = Path("integrateddata/swarm_dnsbpod_2018.parquet")
KPINDEX_CSV   = Path("data/Kpindex/SW-All.csv")
OUT_PARQUET   = Path("normalizeddata/swarm_dnsbpod_2018_normalized.parquet")


# =========================
# 正規化の基準条件
# =========================
F107_REF = 70.0

# 「Kp=1」をApで近似（運用上よく使われる近似）
# ※厳密にやりたい場合はKp→Ap換算表を明示して論文に書けます
AP_REF = 4.0

# Swarm用の基準高度（km）
ALT_REF_KM = 520.0

# 先行研究にならって高緯度を落とす（必要なければ False に）
USE_LAT_LIMIT = True
LAT_LIMIT_DEG = 60.0

# validity_flagがある場合に0のみ使う（必要なければ False に）
USE_VALIDITY_FLAG = True

# デバッグ（MSIS出力shapeを確認したいとき True）
DEBUG_PRINT_MSIS_SHAPE = False


# =========================
# ユーティリティ
# =========================
def ensure_dir(p: Path) -> None:
    """役割: 出力ディレクトリが無ければ作る"""
    p.parent.mkdir(parents=True, exist_ok=True)


def pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """役割: 列名候補から実際に存在する列名を1つ選ぶ（データ差分に強くする）"""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"必要な列が見つかりません。候補={candidates} / 実際の列={list(df.columns)}")
    return None


def get_datetime_column(df: pd.DataFrame) -> pd.Series:
    """
    役割: Swarmデータから時刻を取り出して UTC datetime に統一する
    """
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

    raise KeyError(
        "Swarmデータに時刻が見つかりません。datetime/time/epoch列 or DatetimeIndex を確認してください。"
    )


def check_no_missing(df: pd.DataFrame, cols: list[str]) -> None:
    """役割: 欠損があると正規化できないため、早期に止めて原因を明確化する"""
    bad = df[cols].isna().any()
    if bad.any():
        missing_cols = bad[bad].index.tolist()
        raise ValueError(f"欠損があるため処理を中断します。欠損列={missing_cols}")


# =========================
# Ap履歴（ap[0..6]）を作る
# =========================
def build_ap7_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    役割:
    - SW-All.csv の AP1..AP8（3時間Ap）と AP_AVG（日平均Ap）から
      MSIS入力として典型的な ap[0..6] を各サンプルで作る。

    ここでの定義（よくあるNRLMSISのap配列）:
    ap[0] = 日平均Ap
    ap[1] = 現在の3時間Ap
    ap[2] = 3時間前
    ap[3] = 6時間前
    ap[4] = 9時間前
    ap[5] = 12時間前
    ap[6] = 15時間前

    注意:
    - 0〜15時間前の参照で前日に跨るので、前日分 AP*_prev をmergeしておく必要がある
    """
    # その時刻が属する3時間区間 index（0..7）
    hours = df["datetime"].dt.hour.to_numpy()
    k = (hours // 3).astype(int)  # 0..7

    # 今日のAP1..AP8（shape: N x 8）
    ap_today = df[[f"AP{i}" for i in range(1, 9)]].to_numpy(float)

    # 前日のAP1..AP8（shape: N x 8）
    ap_prev = df[[f"AP{i}_prev" for i in range(1, 9)]].to_numpy(float)

    # 日平均Ap（shape: N）
    ap_avg = df["AP_AVG"].to_numpy(float)

    N = len(df)
    idxN = np.arange(N)

    def get_ap_shift(shift: int) -> np.ndarray:
        """
        役割:
        - 現在から shift*3時間前のApを取り出す
        - 前日に跨る場合は *_prev 側を参照する
        """
        idx = k - shift  # 例えば shift=0なら現在区間、shift=1なら-3h
        use_today = idx >= 0

        # 今日側：idxは0..7（AP1..8に対応）
        val_today = ap_today[idxN, np.clip(idx, 0, 7)]

        # 前日側：idxが-1なら前日AP8、-2なら前日AP7 ... なので idx+8 で 7..3 などになる
        idx_prev = idx + 8
        val_prev = ap_prev[idxN, np.clip(idx_prev, 0, 7)]

        return np.where(use_today, val_today, val_prev)

    ap_now  = get_ap_shift(0)
    ap_m3   = get_ap_shift(1)
    ap_m6   = get_ap_shift(2)
    ap_m9   = get_ap_shift(3)
    ap_m12  = get_ap_shift(4)
    ap_m15  = get_ap_shift(5)

    ap7 = np.column_stack([ap_avg, ap_now, ap_m3, ap_m6, ap_m9, ap_m12, ap_m15]).astype(float)
    return ap7


# =========================
# MSIS密度計算（pymsis 0.12.0 仕様）
# =========================
def msis_density(time_utc: np.ndarray,
                 lat: np.ndarray,
                 lon: np.ndarray,
                 alt_km: np.ndarray,
                 f107s: np.ndarray,
                 f107as: np.ndarray,
                 aps: np.ndarray) -> np.ndarray:
    """
    役割:
    - pymsis (NRLMSIS) を使って「モデル密度」を計算する
    - 正規化では “ref/real の比” を作るために使う

    注意:
    - あなたの環境 (pymsis 0.12.0) では引数名が f107s / f107as / aps（複数形）
    """
    from pymsis import msis

    out = msis.run(
        time_utc,   # dates
        lon,        # lons
        lat,        # lats
        alt_km,     # alts [km]
        f107s=f107s,
        f107as=f107as,
        aps=aps
    )

    out = np.asarray(out)
    if DEBUG_PRINT_MSIS_SHAPE:
        print("MSIS out shape:", out.shape)

    # 役割: 出力配列から「総質量密度」を取り出す
    # 多くの実装で out[:,0] が total mass density になっているが、もし違えばshape確認して調整する
    rho = out[:, 0].astype(float) if out.ndim >= 2 else out.astype(float)

    # 役割: 比計算が壊れるのを防ぐため、異常値チェック
    if np.any(rho <= 0) or np.any(~np.isfinite(rho)):
        raise ValueError("MSISが不正な密度（<=0 または NaN/inf）を返しました。入力値を確認してください。")

    return rho


# =========================
# メイン
# =========================
def main() -> None:
    # ---------- Swarmデータ読み込み ----------
    if not SWARM_PARQUET.exists():
        raise FileNotFoundError(f"Swarm parquet が見つかりません: {SWARM_PARQUET}")

    df_swarm = pd.read_parquet(SWARM_PARQUET).copy()

    # 役割: 時刻をUTCに統一して列に入れる
    df_swarm["datetime"] = get_datetime_column(df_swarm)

    # 役割: 必須列（密度・緯度経度・高度）を特定
    rho_col = pick_column(df_swarm, ["density", "rho", "dens", "rho_obs"])
    lat_col = pick_column(df_swarm, ["lat", "latitude", "geod_lat"])
    lon_col = pick_column(df_swarm, ["lon", "longitude", "geod_lon"])

    alt_m_col  = pick_column(df_swarm, ["altitude_m", "alt_m", "height_m"], required=False)
    alt_km_col = pick_column(df_swarm, ["altitude_km", "alt_km", "height_km"], required=False)
    if alt_km_col is None and alt_m_col is None:
        raise KeyError("高度列が見つかりません（altitude_m / altitude_km 等）。")

    # 役割: 高度を km に統一（MSIS入力はkm）
    if alt_km_col is None:
        df_swarm["alt_km"] = df_swarm[alt_m_col].astype(float) / 1000.0
    else:
        df_swarm["alt_km"] = df_swarm[alt_km_col].astype(float)

    # 役割: validity_flag があれば品質の悪い点を除外
    if USE_VALIDITY_FLAG and "validity_flag" in df_swarm.columns:
        df_swarm = df_swarm[df_swarm["validity_flag"] == 0].copy()

    # 役割: 先行研究にならって高緯度を除外（電離圏・極域ダイナミクスの影響を減らす）
    if USE_LAT_LIMIT:
        df_swarm = df_swarm[df_swarm[lat_col].abs() <= LAT_LIMIT_DEG].copy()

    # ---------- SW-All.csv 読み込み ----------
    if not KPINDEX_CSV.exists():
        raise FileNotFoundError(f"SW-All.csv が見つかりません: {KPINDEX_CSV}")

    ap_cols = [f"AP{i}" for i in range(1, 9)]
    keep_cols = ["DATE", "F10.7_ADJ", "F10.7_ADJ_CENTER81", "AP_AVG", *ap_cols]

    df_geo = pd.read_csv(KPINDEX_CSV, parse_dates=["DATE"])[keep_cols].copy()

    # 役割: 列名を扱いやすくする（MSIS入力に対応させる）
    df_geo.rename(columns={
        "F10.7_ADJ": "F107",
        "F10.7_ADJ_CENTER81": "F107A"
    }, inplace=True)

    # 役割: 日付キーを tz-naive に統一（mergeが安定する）
    df_geo["DATE"] = pd.to_datetime(df_geo["DATE"], utc=True).dt.floor("D").dt.tz_localize(None)

    # ---------- 前日分APを作る（過去15時間のApを埋めるため） ----------
    # 役割: 観測が 0〜15UTC 付近だと過去Apが前日に跨るので、前日のAP1..AP8が必要
    df_geo_prev = df_geo[["DATE", "AP_AVG", *ap_cols]].copy()
    df_geo_prev["DATE"] = df_geo_prev["DATE"] + pd.Timedelta(days=1)
    df_geo_prev = df_geo_prev.rename(columns={c: f"{c}_prev" for c in ["AP_AVG", *ap_cols]})

    # ---------- Swarmと地磁気データを結合 ----------
    # 役割: Swarmの各サンプルに「その日のF10.7/Ap」を付与する
    df_swarm["DATE"] = df_swarm["datetime"].dt.floor("D").dt.tz_localize(None)

    df = df_swarm.merge(df_geo, on="DATE", how="left")
    df = df.merge(df_geo_prev, on="DATE", how="left")

    # --- 前日APが欠ける場合の埋め（例: 解析開始日が2/1で、1/31がCSVに無い） ---
    for i in range(1, 9):
        c = f"AP{i}"
        prev = f"{c}_prev"
        # 役割: 前日データが無い場合に、同日のAPで代用してAp履歴を作れるようにする
        df[prev] = df[prev].fillna(df[c])


    # 欠損チェック（結合がズレていたらここで止まる）
    check_no_missing(df, ["F107", "F107A", "AP_AVG"] + ap_cols + [f"{c}_prev" for c in ap_cols])

    # ---------- MSIS入力用配列を準備 ----------
    # 役割: pandas -> numpy へ（高速化・モデル入力用）
    rho_obs = df[rho_col].astype(float).to_numpy()
    lat = df[lat_col].astype(float).to_numpy()
    lon = df[lon_col].astype(float).to_numpy()
    alt_km = df["alt_km"].astype(float).to_numpy()
    time_utc = pd.to_datetime(df["datetime"], utc=True).to_numpy()

    # 役割: F10.7（その日）と F10.7A（81日平均）
    f107s_real  = df["F107"].astype(float).to_numpy()
    f107as_real = df["F107A"].astype(float).to_numpy()

    # 役割: Ap履歴配列（N x 7）を作る
    aps_real = build_ap7_matrix(df)

    # ---------- 基準条件の配列を作る ----------
    # 役割: “基準条件でのモデル密度” を計算するための入力を作る
    f107s_ref  = np.full_like(f107s_real,  F107_REF, dtype=float)
    f107as_ref = np.full_like(f107as_real, F107_REF, dtype=float)

    # Ap基準（Kp=1相当）として、ap[0..6]を全て4で埋める（近似）
    aps_ref = np.full_like(aps_real, AP_REF, dtype=float)

    alt_ref_km = np.full_like(alt_km, ALT_REF_KM, dtype=float)

    # ---------- MSISでモデル密度を計算 ----------
    # 役割:
    #  - rho_model_real: 実条件（観測高度 + 実F10.7 + 実Ap履歴）
    #  - rho_model_ref : 基準条件（基準高度 + F10.7=70 + Ap=4）
    rho_model_real = msis_density(
        time_utc=time_utc, lat=lat, lon=lon, alt_km=alt_km,
        f107s=f107s_real, f107as=f107as_real, aps=aps_real
    )

    rho_model_ref = msis_density(
        time_utc=time_utc, lat=lat, lon=lon, alt_km=alt_ref_km,
        f107s=f107s_ref, f107as=f107as_ref, aps=aps_ref
    )

    # ---------- 正規化（核心） ----------
    # 役割: 観測密度を「基準条件へ換算」し、非SSW要因（高度・太陽・地磁気）を抑える
    ratio = rho_model_ref / rho_model_real
    df["density_norm"] = rho_obs * ratio

    # 役割: 後から検証できるよう、比と基準条件も保存
    df["norm_ratio_model_ref_over_real"] = ratio
    df["norm_ref_alt_km"] = ALT_REF_KM
    df["norm_ref_F107"] = F107_REF
    df["norm_ref_AP"] = AP_REF

    # ---------- 出力 ----------
    ensure_dir(OUT_PARQUET)
    df.to_parquet(OUT_PARQUET, index=False)

    print("✅ 正規化完了")
    print(f"- input swarm : {SWARM_PARQUET}")
    print(f"- input geo   : {KPINDEX_CSV}")
    print(f"- output      : {OUT_PARQUET}")
    print(f"- N rows      : {len(df):,}")
    print(f"- used cols   : rho={rho_col}, lat={lat_col}, lon={lon_col}, alt_km=alt_km")
    print(f"- ref cond    : F10.7={F107_REF}, Ap={AP_REF}, alt={ALT_REF_KM} km")


if __name__ == "__main__":
    main()
