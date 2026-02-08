# =========================
# Swarm DNSCPOD.CDF を読み込み、
# 期間分まとめて DataFrame 化し、parquet に保存するスクリプト
# =========================

from pathlib import Path
import numpy as np
import pandas as pd
from cdflib import CDF, cdfepoch

# =========================
# 定数：FILL値（欠損を表す巨大値）
# SwarmのCDFでは、欠損値として 0.99900e33 のような値が入っていることがある
# これを NaN に変換して解析しやすくする
# =========================
FILL = 0.99900e33


# =========================
# 1ファイル（1つのCDF）を読み込んで DataFrame にする関数
# - 時刻を index にして整形
# - FILL値をNaNに置換
# - validity_flag が 0（正常）のデータだけ残す
# =========================
def read_swarm_dnscpod(path: str) -> pd.DataFrame:
    # CDFファイルを開く
    cdf = CDF(path)

    # ---- 時刻の読み込みと変換 ----
    # "time" はCDF epoch形式（数値）
    t_raw = cdf.varget("time")

    # cdflib の cdfepoch.to_datetime でPython datetimeに変換し、
    # pandasのDatetimeIndexへ（utc=TrueでUTCとして扱う）
    # tz_convert(None) で "timezoneなしのUTC時刻" にする（扱いやすい）
    t = pd.to_datetime(cdfepoch.to_datetime(t_raw), utc=True).tz_convert(None)

    # ---- 各変数を読み込んでDataFrame化 ----
    # index = 時刻
    df = pd.DataFrame({
        "density": cdf.varget("density"),                 # 観測密度（単位はデータ仕様に依存）
        "density_orbitmean": cdf.varget("density_orbitmean"),  # 軌道平均密度
        "validity_flag": cdf.varget("validity_flag"),     # 0=正常, 1=異常 など
        "altitude_m": cdf.varget("altitude"),             # 高度 [m]（変数名は altitude だが列名は altitude_m）
        "lat": cdf.varget("latitude"),                    # 緯度
        "lon": cdf.varget("longitude"),                   # 経度
        "lst_h": cdf.varget("local_solar_time"),          # Local Solar Time [hour]
    }, index=t).sort_index()  # 時刻順にソート

    # ---- FILL値を NaN に置換 ----
    # 欠損を示す巨大値(FILL)が入っている列を対象に、NaNへ変換
    # np.isclose を使い、浮動小数の誤差を許容して比較
    for col in ["density", "density_orbitmean", "altitude_m", "lat", "lon", "lst_h"]:
        df.loc[np.isclose(df[col], FILL), col] = np.nan

    # ---- validity_flag による品質フィルタ ----
    # validity_flag: 0 が nominal（正常）という前提で、0以外を除外
    df = df[df["validity_flag"] == 0].copy()

    return df


# =========================
# 複数のCDFファイルを一括で読み込み、連結して1つのDataFrameにする関数
# - data_dir 以下から pattern に合うCDFを全部拾う
# - 各CDFを read_swarm_dnscpod() でDataFrame化して list に集める
# - concat で縦結合して時系列データにする
# - 同一時刻が重複していたら1つにする（keep="first"）
# =========================
def collect_cdfs(data_dir="data/SWARM_B", pattern="SW_OPER_DNSBPOD_2__*.cdf") -> pd.DataFrame:
    # ディレクトリ内のCDFファイルを列挙（ファイル名順）
    files = sorted(Path(data_dir).glob(pattern))

    # 見つからない場合はエラーにする（パスや拡張子のミス検出用）
    if not files:
        raise FileNotFoundError(f"No CDF files found in {data_dir} with pattern {pattern}")

    dfs = []
    for fp in files:
        # 1ファイル読み込み
        df = read_swarm_dnscpod(str(fp))

        # 追跡用：この行がどのファイル由来かを残す（不要なら削除OK）
        df["source_file"] = fp.name

        dfs.append(df)

    # ---- 全ファイル分を縦結合 ----
    all_df = pd.concat(dfs).sort_index()

    # ---- 同一時刻が重複していた場合の処理 ----
    # まれに同一の時刻 index が重複する可能性があるので、最初のレコードだけ残す
    # （もし平均化したいなら groupby(level=0).mean() などに置き換える）
    all_df = all_df[~all_df.index.duplicated(keep="first")]

    return all_df


# =========================
# 実行パート
# - data/SWARM_C からCDFをまとめて読み込み
# - 先頭を表示して、期間とサンプル数を確認
# - parquet に保存して後続解析（プロット等）を高速化
# =========================
all_df = collect_cdfs(data_dir="data/SWARM_B")

# 読み込み結果の先頭5行を表示
print(all_df.head(), "\n")

# 時刻範囲と総データ数を表示
print(all_df.index.min(), "->", all_df.index.max(), "N=", len(all_df))

# ---- 保存（おすすめ：parquet） ----
# parquet は高速・軽量で、後の解析に便利
all_df.to_parquet("integrateddata/swarm_dnsbpod_2018.parquet")
print("Saved: integrateddata/swarm_dnsbpod_2018.parquet")