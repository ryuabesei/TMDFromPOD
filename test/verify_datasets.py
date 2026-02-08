from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# =========================
# パス（ユーザー指定）
# =========================
RAW_EXAMPLE_CDF = Path("data/SWARM_C/SW_OPER_DNSCPOD_2__20180201T000000_20180201T235930_0301.cdf")
INTEGRATED_PARQUET = Path("integrateddata/swarm_dnscpod_2018.parquet")
NORMALIZED_PARQUET = Path("normalizeddata/swarm_dnscpod_2018_normalized.parquet")
KP_CSV = Path("data/Kpindex/SW-All.csv")


# =========================
# ユーティリティ
# =========================
def assert_exists(p: Path) -> None:
    """意図: 入力ファイルが存在するかを最初に保証して、後段のエラー原因を明確化する"""
    if not p.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {p}")


def parse_dates_from_filename(cdf_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    意図: CDFファイル名に含まれる開始・終了時刻（UTC）を取り出し、
          CDFの実データ時刻と一致するか検証するための基準を作る
    例:
      SW_OPER_DNSCPOD_2__20180201T000000_20180201T235930_0301.cdf
    """
    m = re.search(r"__(\d{8}T\d{6})_(\d{8}T\d{6})_", cdf_path.name)
    if not m:
        raise ValueError(f"ファイル名から時刻をパースできません: {cdf_path.name}")
    t0 = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S", utc=True)
    t1 = pd.to_datetime(m.group(2), format="%Y%m%dT%H%M%S", utc=True)
    return t0, t1


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """意図: parquet側の列名揺れに耐えるため、候補から実在列を選ぶ"""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"列が見つかりません。候補={candidates} / 実列={list(df.columns)}")


def basic_stats(df: pd.DataFrame, name: str, time_col: str) -> None:
    """意図: 期間・欠損・重複など、データの健康診断をまず行う"""
    t = pd.to_datetime(df[time_col], utc=True)
    print(f"\n=== {name} BASIC CHECK ===")
    print("rows:", len(df))
    print("time min:", t.min(), "time max:", t.max())
    print("duplicated time:", t.duplicated().sum())
    print("NaN ratio (per col):")
    print(df.isna().mean().sort_values(ascending=False).head(12))


def check_physical_ranges(df: pd.DataFrame, name: str, time_col: str) -> None:
    """
    意図: 値が物理的に破綻していないかをチェック（単位ミスや列取り違えの検出に強い）
    """
    rho_col = pick_col(df, ["density", "rho", "dens"])
    lat_col = pick_col(df, ["lat", "latitude"])
    lon_col = pick_col(df, ["lon", "longitude"])
    alt_col = pick_col(df, ["altitude_m", "alt_m", "altitude_km", "alt_km"])

    rho = df[rho_col].astype(float)
    lat = df[lat_col].astype(float)
    lon = df[lon_col].astype(float)
    alt = df[alt_col].astype(float)

    print(f"\n=== {name} PHYSICAL RANGE CHECK ===")
    print("density min/max:", rho.min(), rho.max(), " (should be >0)")
    print("lat min/max:", lat.min(), lat.max(), " (should be within [-90, 90])")
    print("lon min/max:", lon.min(), lon.max(), " (often within [-180,180] or [0,360])")
    print("alt min/max:", alt.min(), alt.max(), " (m or km depending on column)")

    if (rho <= 0).any():
        print("⚠ density<=0 が含まれます（ゼロ埋めや欠損処理の可能性）")
    if (lat.abs() > 90).any():
        print("⚠ |lat|>90 が含まれます（列取り違えの可能性）")


# =========================
# Step 1: Raw CDF を読む（1日分）
# =========================
def read_swarm_cdf_minimal(cdf_path: Path) -> pd.DataFrame:
    """
    意図:
    - Raw CDFが“読めること”を確認し、
    - 少なくとも (datetime, density, lat, lon, altitude) を取り出す
    注意:
    - CDFの変数名は製品やbaselineで微妙に違う可能性があるため、
      まずは「変数一覧を見て候補から探す」方式にする
    """
    try:
        from cdflib import CDF
    except ImportError as e:
        raise ImportError(
            "cdflib が必要です。次を実行してください: pip install cdflib"
        ) from e

    cdf = CDF(str(cdf_path))
    vars_all = list(cdf.cdf_info().get("zVariables", [])) + list(cdf.cdf_info().get("rVariables", []))

    # 意図: まずはCDF内に何が入っているかをログ表示（最初の1回だけでもOK）
    print("\n=== CDF VARIABLES (first 80) ===")
    print(vars_all[:80])

    def find_var(candidates: list[str]) -> str:
        for c in candidates:
            if c in vars_all:
                return c
        raise KeyError(f"CDFに候補変数が見つかりません: {candidates}")

    # 意図: 代表的な候補名（製品によって異なるので、ここは必要に応じて増やす）
    t_var   = find_var(["Timestamp", "UTC", "time", "t", "Epoch", "epoch"])
    rho_var = find_var(["rho", "Density", "density", "NeutralDensity", "Rho"])
    lat_var = find_var(["lat", "Latitude", "latitude", "GeodeticLatitude"])
    lon_var = find_var(["lon", "Longitude", "longitude", "GeodeticLongitude"])
    alt_var = find_var(["alt", "Altitude", "altitude", "Height", "height", "radius"])

    # 意図: cdflib は時刻を “CDF_EPOCH” の数値として返すことがあるので、変換する
    t_raw = cdf.varget(t_var)

    # CDF epochの変換（形式により分岐）
    # - cdflib.cdfepoch.to_datetime は epoch配列を Python datetime にする
    from cdflib import cdfepoch
    try:
        t_dt = pd.to_datetime(cdfepoch.to_datetime(t_raw), utc=True)
    except Exception:
        # もしすでにdatetime互換ならそのまま
        t_dt = pd.to_datetime(t_raw, utc=True, errors="coerce")

    df = pd.DataFrame({
        "datetime": t_dt,
        "density": np.asarray(cdf.varget(rho_var)).astype(float),
        "lat": np.asarray(cdf.varget(lat_var)).astype(float),
        "lon": np.asarray(cdf.varget(lon_var)).astype(float),
        "alt_raw": np.asarray(cdf.varget(alt_var)).astype(float),
    })

    # 意図: 時刻変換失敗がないか確認
    if df["datetime"].isna().any():
        raise ValueError("CDFの時刻変換に失敗しました（datetimeがNaN）。t_var候補を見直してください。")

    return df


# =========================
# Step 2: Integrate / Normalize を読む
# =========================
def load_integrated() -> pd.DataFrame:
    """意図: 統合parquetを読み、時刻列を datetime(UTC) に揃える"""
    assert_exists(INTEGRATED_PARQUET)
    df = pd.read_parquet(INTEGRATED_PARQUET).copy()

    # 意図: 時刻列は df.index の場合もあるので吸収する
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    else:
        raise KeyError("integrated parquet に datetime 列も DatetimeIndex もありません。")

    return df


def load_normalized() -> pd.DataFrame:
    """意図: 正規化parquetを読み、時刻列を datetime(UTC) に揃える"""
    assert_exists(NORMALIZED_PARQUET)
    df = pd.read_parquet(NORMALIZED_PARQUET).copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


# =========================
# Step 3: Raw vs Integrate の一致確認（サンプル照合）
# =========================
def compare_raw_to_integrated(raw_df: pd.DataFrame, int_df: pd.DataFrame, tol: dict[str, float] | None = None) -> None:
    """
    意図:
    - 1日分CDFの値が Integrate に正しく入っているかを照合する
    - まずは “時刻でinner join” して、主要列が一致するかを見る
    """
    if tol is None:
        tol = {
            "density": 0.0,   # 密度はそのまま移したなら完全一致が期待（丸めがあるなら緩める）
            "lat": 0.0,
            "lon": 0.0,
            "alt_km": 0.0,
        }

    # 意図: integrate側の列名を決める
    rho_col = pick_col(int_df, ["density", "rho", "dens"])
    lat_col = pick_col(int_df, ["lat", "latitude"])
    lon_col = pick_col(int_df, ["lon", "longitude"])
    # 高度は m か km のどちらか。比較は km に統一
    if "alt_km" in int_df.columns:
        alt_int_km = int_df["alt_km"].astype(float)
    else:
        alt_m = pick_col(int_df, ["altitude_m", "alt_m"])
        alt_int_km = int_df[alt_m].astype(float) / 1000.0

    int_min = int_df[["datetime"]].copy()
    int_min["density"] = int_df[rho_col].astype(float).to_numpy()
    int_min["lat"] = int_df[lat_col].astype(float).to_numpy()
    int_min["lon"] = int_df[lon_col].astype(float).to_numpy()
    int_min["alt_km"] = alt_int_km.to_numpy()

    raw_min = raw_df.copy()
    # 意図: rawの高度変数は種類が分からないので、ここではそのまま alt_raw として保持。
    #       もし raw が km で、integrate が km なら一致するはず。
    #       不一致なら単位確認が必要（m / km / radius）。
    # まずは “比較用に alt_raw を km と仮定” してみる（ダメなら後で検出する）
    raw_min["alt_km"] = raw_min["alt_raw"].astype(float)

    merged = pd.merge(raw_min, int_min, on="datetime", suffixes=("_raw", "_int"), how="inner")
    print("\n=== RAW vs INTEGRATED MERGE ===")
    print("raw rows:", len(raw_min), "integrated rows:", len(int_min), "matched:", len(merged))

    if len(merged) == 0:
        raise ValueError("時刻が一致する行が0です。RawとIntegrateで時刻定義が違う可能性があります。")

    def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.nanmax(np.abs(a - b)))

    d_rho = max_abs_diff(merged["density_raw"].to_numpy(), merged["density_int"].to_numpy())
    d_lat = max_abs_diff(merged["lat_raw"].to_numpy(), merged["lat_int"].to_numpy())
    d_lon = max_abs_diff(merged["lon_raw"].to_numpy(), merged["lon_int"].to_numpy())
    d_alt = max_abs_diff(merged["alt_km_raw"].to_numpy(), merged["alt_km_int"].to_numpy())

    print("max |density_raw - density_int| =", d_rho)
    print("max |lat_raw - lat_int|       =", d_lat)
    print("max |lon_raw - lon_int|       =", d_lon)
    print("max |alt_km_raw - alt_km_int| =", d_alt)

    # 意図: altが大きくズレるなら単位推定（m↔km）を試す
    if d_alt > 5:  # 5km以上ズレたら怪しい
        d_alt_if_m = max_abs_diff(merged["alt_km_raw"].to_numpy() / 1000.0, merged["alt_km_int"].to_numpy())
        print("  (alt_rawをmとして/1000した場合の max diff) =", d_alt_if_m)
        if d_alt_if_m < d_alt:
            print("⚠ alt_raw は m の可能性が高いです（CDFの高度単位を確認してください）")

    # 意図: 期待精度（tol）を超えたら警告（丸めや補間があるなら tol を緩める）
    def warn_if(diff: float, name: str, threshold: float):
        if diff > threshold:
            print(f"⚠ {name} の差が閾値を超えています: diff={diff} > tol={threshold}")

    warn_if(d_rho, "density", tol["density"])
    warn_if(d_lat, "lat", tol["lat"])
    warn_if(d_lon, "lon", tol["lon"])
    warn_if(d_alt, "alt_km", tol["alt_km"])


# =========================
# Step 4: Integrate vs Normalize の一致確認
# =========================
def compare_integrated_to_normalized(int_df: pd.DataFrame, norm_df: pd.DataFrame) -> None:
    """
    意図:
    - Normalize が Integrate の “同一行” を元に作られているか確認
      (i) 行数が一致
      (ii) datetime が一致（同順 or merge可能）
      (iii) density_norm が正規化として妥当（有限・正）
      (iv) ratio列があるなら式通りかチェック
    """
    print("\n=== INTEGRATED vs NORMALIZED CHECK ===")
    print("integrated rows:", len(int_df), "normalized rows:", len(norm_df))

    # (i) 行数
    if len(int_df) != len(norm_df):
        print("⚠ 行数が一致しません（フィルタ処理の有無を確認）")

    # (ii) datetime一致（集合として一致するか）
    s_int = pd.Series(int_df["datetime"])
    s_norm = pd.Series(norm_df["datetime"])
    same_set = set(s_int) == set(s_norm)
    print("datetime set identical:", same_set)

    if not same_set:
        # 意図: どの日時が欠けているかを把握する
        missing_in_norm = sorted(set(s_int) - set(s_norm))
        missing_in_int  = sorted(set(s_norm) - set(s_int))
        print("missing in normalized (first 5):", missing_in_norm[:5])
        print("missing in integrated (first 5):", missing_in_int[:5])

    # (iii) density_norm の健康診断
    if "density_norm" not in norm_df.columns:
        raise KeyError("normalized parquet に density_norm がありません。")

    dn = norm_df["density_norm"].astype(float)
    print("density_norm min/max:", float(dn.min()), float(dn.max()))
    print("density_norm NaN:", int(dn.isna().sum()), "inf:", int(np.isinf(dn.to_numpy()).sum()), "<=0:", int((dn <= 0).sum()))
    if (dn <= 0).any() or (~np.isfinite(dn.to_numpy())).any():
        print("⚠ density_norm に不正値（<=0/NaN/inf）が含まれます。")

    # (iv) ratio列があるなら式チェック（存在する場合のみ）
    # 期待: density_norm = density * norm_ratio_model_ref_over_real
    if "norm_ratio_model_ref_over_real" in norm_df.columns:
        # 意図: 同一datetimeで照合する（順序が違ってもOK）
        rho_col = pick_col(int_df, ["density", "rho", "dens"])
        tmp = int_df[["datetime", rho_col]].merge(
            norm_df[["datetime", "density_norm", "norm_ratio_model_ref_over_real"]],
            on="datetime", how="inner"
        )
        calc = tmp[rho_col].astype(float) * tmp["norm_ratio_model_ref_over_real"].astype(float)
        diff = np.nanmax(np.abs(calc.to_numpy() - tmp["density_norm"].astype(float).to_numpy()))
        print("max |density*dRatio - density_norm| =", float(diff))
        if diff > 0:
            print("⚠ 完全一致しません（丸め・型変換・別列density使用の可能性）。diff許容を決めましょう。")
    else:
        print("note: ratio列が無いので式の厳密チェックはスキップ（設計としてOK）")


# =========================
# 実行（Step-by-step）
# =========================
def main():
    # ---- Step 0: ファイル存在確認 ----
    assert_exists(RAW_EXAMPLE_CDF)
    assert_exists(INTEGRATED_PARQUET)
    assert_exists(NORMALIZED_PARQUET)
    assert_exists(KP_CSV)

    # ---- Step 1: Raw CDF 読み込み（1日分） & 基本確認 ----
    raw_df = read_swarm_cdf_minimal(RAW_EXAMPLE_CDF)

    # 意図: ファイル名の期間と CDF時刻範囲が一致するかを見る
    t0_name, t1_name = parse_dates_from_filename(RAW_EXAMPLE_CDF)
    t0_data = raw_df["datetime"].min()
    t1_data = raw_df["datetime"].max()
    print("\n=== RAW CDF TIME RANGE CHECK ===")
    print("from filename:", t0_name, "to", t1_name)
    print("from data    :", t0_data, "to", t1_data)
    print("delta start (sec):", (t0_data - t0_name).total_seconds())
    print("delta end   (sec):", (t1_data - t1_name).total_seconds())

    basic_stats(raw_df, "RAW (1-day CDF)", "datetime")
    check_physical_ranges(raw_df, "RAW (1-day CDF)", "datetime")

    # ---- Step 2: Integrate / Normalize 読み込み & 基本確認 ----
    int_df = load_integrated()
    norm_df = load_normalized()

    basic_stats(int_df, "INTEGRATED (parquet)", "datetime")
    check_physical_ranges(int_df, "INTEGRATED (parquet)", "datetime")

    basic_stats(norm_df, "NORMALIZED (parquet)", "datetime")
    # normalized は density_norm のレンジも見たいので、physicalチェックは density 列がある場合のみ
    if "density" in norm_df.columns:
        check_physical_ranges(norm_df, "NORMALIZED (parquet)", "datetime")

    # ---- Step 3: Raw vs Integrate 照合（この1日がIntegrateに正しく入っているか） ----
    # 意図: integrateを “同日だけ” に絞って比較する（比較対象の時刻集合を揃える）
    day = pd.Timestamp("2018-02-01", tz="UTC")
    int_day = int_df[(int_df["datetime"] >= day) & (int_df["datetime"] < day + pd.Timedelta(days=1))].copy()
    print("\n=== INTEGRATED DAY SLICE ===")
    print("integrated rows on 2018-02-01:", len(int_day))

    compare_raw_to_integrated(raw_df, int_day)

    # ---- Step 4: Integrate vs Normalize 照合（正規化がIntegrate由来か） ----
    compare_integrated_to_normalized(int_df, norm_df)

    print("\n✅ Step-by-step verification finished.")


if __name__ == "__main__":
    main()
