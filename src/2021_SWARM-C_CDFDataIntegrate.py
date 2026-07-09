"""
2021_SWARM-C_CDFDataIntegrate.py

Purpose:
    Integrate SWARM-C density CDF files for the 2021 SSW period (2020-12-25 to 2021-02-05).
    Filters nominal data (validity_flag == 0) and outputs a merged Parquet dataset.

Output:
    integrateddata/2021/swarm_dnscpod_2021_integrated.parquet
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from cdflib import CDF, cdfepoch

FILL = 0.99900e33

DATE_START = pd.Timestamp("2020-12-25", tz=None)
DATE_END   = pd.Timestamp("2021-02-05 23:59:59", tz=None)

def read_swarm_dnscpod(path: str) -> pd.DataFrame:
    cdf = CDF(path)
    t_raw = cdf.varget("time")
    # cdfepoch.to_datetime translates cdf epoch to UTC datetime objects
    t = pd.to_datetime(cdfepoch.to_datetime(t_raw), utc=True).tz_convert(None)

    df = pd.DataFrame({
        "density": cdf.varget("density"),
        "density_orbitmean": cdf.varget("density_orbitmean"),
        "validity_flag": cdf.varget("validity_flag"),
        "altitude_m": cdf.varget("altitude"),
        "lat": cdf.varget("latitude"),
        "lon": cdf.varget("longitude"),
        "lst_h": cdf.varget("local_solar_time"),
    }, index=t).sort_index()

    for col in ["density", "density_orbitmean", "altitude_m", "lat", "lon", "lst_h"]:
        df.loc[np.isclose(df[col], FILL), col] = np.nan

    df = df[df["validity_flag"] == 0].copy()
    return df

def main():
    data_dir = Path("data/SSW2021/SWARM_C")
    out_path = Path("integrateddata/2021/swarm_dnscpod_2021_integrated.parquet")
    
    print("Collecting SWARM-C CDF files...")
    files = sorted(list(data_dir.glob("SW_OPER_DNSCPOD_2__*.cdf")))
    if not files:
        raise FileNotFoundError(f"No CDF files found in {data_dir}")

    dfs = []
    for fp in files:
        # ファイル日付が範囲内か大まかにチェック
        # ファイル名形式例: SW_OPER_DNSCPOD_2__20210107T000000_20210107T235930_0301.cdf
        parts = fp.name.split("__")
        if len(parts) >= 2:
            time_part = parts[1].split("_")[0] # e.g. 20210107T000000
            file_date = pd.to_datetime(time_part[:8], format="%Y%m%d")
            # 余裕を見て 1日前後も許容
            if file_date < DATE_START - pd.Timedelta(days=2) or file_date > DATE_END + pd.Timedelta(days=2):
                continue
        
        print(f"  Reading {fp.name}...")
        try:
            df = read_swarm_dnscpod(str(fp))
            df["source_file"] = fp.name
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to read {fp.name}: {e}")

    if not dfs:
        raise ValueError("No data extracted from CDF files.")

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
