"""
2021_GRACE-FO_DataIntegrate.py

Purpose:
    Integrate GRACE-FO density text files for the 2021 SSW period (2020-12-25 to 2021-02-05).
    Reads the latest version (v02c, falling back to v02b then v02) of text files,
    parses columns, filters nominal data (flag == 0), and merges to Parquet.

Output:
    integrateddata/2021/grace_fo_dns_2021_integrated.parquet
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

DATE_START = pd.Timestamp("2020-12-25", tz=None)
DATE_END   = pd.Timestamp("2021-02-05 23:59:59", tz=None)

# 各月で最新のファイルを決定する関数
def get_best_files(data_dir: Path) -> list[Path]:
    # 対象とする月: 2020_12, 2021_01, 2021_02
    months = ["2020_12", "2021_01", "2021_02"]
    best_files = []
    
    for m in months:
        # v02c -> v02b -> v02 の順にファイルを探す
        for suffix in ["v02c", "v02b", "v02"]:
            fp = data_dir / f"GC_DNS_ACC_{m}_{suffix}.txt"
            if fp.exists():
                best_files.append(fp)
                print(f"  Selected: {fp.name}")
                break
        else:
            print(f"  Warning: No text file found for month {m}")
            
    return best_files

def read_grace_fo_txt(path: Path) -> pd.DataFrame:
    # 列名の定義
    cols = [
        "date_str", "time_str", "time_sys", "altitude_m", "lon", "lat", "lst_h", 
        "arglat", "density", "density_orbitmean", "validity_flag", "validity_flag_orbitmean"
    ]
    
    # pandas で空白区切りのファイルを読み込む (コメント行 # をスキップ)
    df = pd.read_csv(
        path,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=cols,
        dtype={"date_str": str, "time_str": str}
    )
    
    # date と time から datetime 列を生成
    df["datetime"] = pd.to_datetime(df["date_str"] + " " + df["time_str"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    
    # インデックス設定
    df = df.set_index("datetime").sort_index()
    
    # validity_flag == 0 のみ抽出 (nominal)
    df = df[df["validity_flag"] == 0.0].copy()
    
    # 不要な列を落として整理
    df = df.drop(columns=["date_str", "time_str", "time_sys", "validity_flag_orbitmean"])
    return df

def main():
    data_dir = Path("data/SSW2021/GRACE-FO")
    out_path = Path("integrateddata/2021/grace_fo_dns_2021_integrated.parquet")
    
    print("Determining best files to read...")
    best_files = get_best_files(data_dir)
    if not best_files:
        raise FileNotFoundError("No GRACE-FO text files found.")
        
    dfs = []
    for fp in best_files:
        print(f"  Parsing {fp.name}...")
        df = read_grace_fo_txt(fp)
        df["source_file"] = fp.name
        dfs.append(df)
        
    all_df = pd.concat(dfs).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="first")]
    print(f"  Total records loaded: {len(all_df):,}")
    
    # 期間フィルタ
    all_df = all_df[(all_df.index >= DATE_START) & (all_df.index <= DATE_END)]
    print(f"  Filtered records ({DATE_START.date()} to {DATE_END.date()}): {len(all_df):,}")
    
    # 出力
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(out_path)
    print(f"  Saved integrated data to: {out_path}")

if __name__ == "__main__":
    main()
