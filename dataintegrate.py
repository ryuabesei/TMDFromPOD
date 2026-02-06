from pathlib import Path
import re
import numpy as np
import pandas as pd
from cdflib import CDF, cdfepoch

FILL = 0.99900e33

def read_swarm_dnscpod(path: str) -> pd.DataFrame:
    cdf = CDF(path)

    t_raw = cdf.varget("time")
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

    # FILLVAL -> NaN
    for col in ["density", "density_orbitmean", "altitude_m", "lat", "lon", "lst_h"]:
        df.loc[np.isclose(df[col], FILL), col] = np.nan

    # validity flag: 0 nominal, 1 anomalous
    df = df[df["validity_flag"] == 0].copy()

    return df

def collect_cdfs(data_dir="data/SWARM_C", pattern="SW_OPER_DNSCPOD_2__*.cdf") -> pd.DataFrame:
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CDF files found in {data_dir} with pattern {pattern}")

    dfs = []
    for fp in files:
        df = read_swarm_dnscpod(str(fp))
        df["source_file"] = fp.name  # どのファイル由来か追跡用（要らなければ消す）
        dfs.append(df)

    all_df = pd.concat(dfs).sort_index()
    # 同一時刻が重複してたら（まれに）まとめる：平均 or 最初を採用
    all_df = all_df[~all_df.index.duplicated(keep="first")]

    return all_df

# 実行
all_df = collect_cdfs(data_dir="data/SWARM_C")
print(all_df.head(), "\n")
print(all_df.index.min(), "->", all_df.index.max(), "N=", len(all_df))

# 保存（おすすめ：parquet）
all_df.to_parquet("swarm_dnscpod_2018.parquet")
print("Saved: swarm_dnscpod_2018.parquet")
