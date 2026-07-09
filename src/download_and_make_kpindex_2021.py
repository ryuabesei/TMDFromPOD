"""
download_and_make_kpindex_2021.py

Purpose:
    Download geomagnetic and solar indices from GFZ Potsdam and generate
    the format-compatible index CSV for the 2021 SSW analysis period (2020-12-20 to 2021-02-28).
    Includes computing 81-day centered and backward moving averages of F10.7.

GFZ Source URL:
    https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt

Output:
    data/SSW2021/Kpindex/SW-20201220_20210228.csv
"""

from __future__ import annotations
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
OUT_CSV = Path("data/SSW2021/Kpindex/SW-20201220_20210228.csv")

def main():
    print(f"Downloading indices file from: {URL}")
    temp_file = Path("data/SSW2021/Kpindex/gfz_kp_temp.txt")
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    
    urllib.request.urlretrieve(URL, temp_file)
    print("Download completed. Parsing data...")

    # 空白区切りのデータ行をパース
    data_rows = []
    with open(temp_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # 必要な最低限の列数があるかチェック
            if len(parts) >= 26:
                data_rows.append(parts)

    print(f"Loaded {len(data_rows):,} rows of daily data. Creating DataFrame...")
    
    # DataFrameの作成
    # 列名の対応 (GFZの列インデックスに準拠)
    # 0: YYYY, 1: MM, 2: DD
    # 7..14: KP1..KP8
    # 15..22: AP1..AP8
    # 23: Ap (average)
    # 25: F10.7 (observed)
    
    processed = []
    for r in data_rows:
        try:
            year  = int(r[0])
            month = int(r[1])
            day   = int(r[2])
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            
            row_dict = {
                "DATE": date_str,
                "KP1": float(r[7]),
                "KP2": float(r[8]),
                "KP3": float(r[9]),
                "KP4": float(r[10]),
                "KP5": float(r[11]),
                "KP6": float(r[12]),
                "KP7": float(r[13]),
                "KP8": float(r[14]),
                "AP1": float(r[15]),
                "AP2": float(r[16]),
                "AP3": float(r[17]),
                "AP4": float(r[18]),
                "AP5": float(r[19]),
                "AP6": float(r[20]),
                "AP7": float(r[21]),
                "AP8": float(r[22]),
                "AP_AVG": float(r[23]),
                "F10.7_OBS": float(r[25])
            }
            processed.append(row_dict)
        except Exception as e:
            continue

    df = pd.DataFrame(processed)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()

    # 欠損値 (-1.0 など) を処理
    df = df.replace(-1.0, np.nan)
    # 欠損の穴埋め (前方補完)
    df["F10.7_OBS"] = df["F10.7_OBS"].ffill()

    # 同一値として F10.7_ADJ を定義
    df["F10.7_ADJ"] = df["F10.7_OBS"]

    # F10.7 の 81日移動平均の計算
    print("Computing 81-day moving averages of F10.7...")
    # centered (前後40日、当日の計81日)
    df["F10.7_ADJ_CENTER81"] = df["F10.7_ADJ"].rolling(window=81, center=True, min_periods=1).mean()
    df["F10.7_OBS_CENTER81"] = df["F10.7_OBS"].rolling(window=81, center=True, min_periods=1).mean()
    
    # backward (過去80日、当日の計81日)
    df["F10.7_ADJ_LAST81"]   = df["F10.7_ADJ"].rolling(window=81, center=False, min_periods=1).mean()
    df["F10.7_OBS_LAST81"]   = df["F10.7_OBS"].rolling(window=81, center=False, min_periods=1).mean()

    # 切り出し期間の設定: 2020-12-20 から 2021-02-28
    # 前々日APの計算用に 12-20 からに設定
    start_date = "2020-12-20"
    end_date   = "2021-02-28"
    df_slice = df.loc[start_date:end_date].copy()
    
    # インデックスをリセットしてDATE列に戻す
    df_slice = df_slice.reset_index()
    # 日付フォーマットを yyyy-mm-dd に変更
    df_slice["DATE"] = df_slice["DATE"].dt.strftime("%Y-%m-%d")

    # CSVの出力
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_slice.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved processed index CSV to: {OUT_CSV}")
    print(f"   Row count: {len(df_slice)}")
    
    # 一時ファイルの削除
    if temp_file.exists():
        temp_file.unlink()

if __name__ == "__main__":
    main()
