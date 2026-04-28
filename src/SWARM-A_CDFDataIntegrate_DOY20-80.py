# =========================
# SWARM-A: CDF統合スクリプト（DOY 20-80版）
# 対象期間: 2018-01-20 (DOY20) ～ 2018-03-21 (DOY80)
# =========================

from pathlib import Path
import numpy as np
import pandas as pd
from cdflib import CDF, cdfepoch

FILL = 0.99900e33

DATE_START = pd.Timestamp("2018-01-20", tz=None)  # DOY 20
DATE_END   = pd.Timestamp("2018-03-21 23:59:59", tz=None)  # DOY 80


def read_swarm_dnsapod(path: str) -> pd.DataFrame:
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

    for col in ["density", "density_orbitmean", "altitude_m", "lat", "lon", "lst_h"]:
        df.loc[np.isclose(df[col], FILL), col] = np.nan

    df = df[df["validity_flag"] == 0].copy()
    return df


def collect_cdfs(data_dir="data/SWARM_A", pattern="SW_OPER_DNSAPOD_2__*.cdf") -> pd.DataFrame:
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CDF files found in {data_dir} with pattern {pattern}")

    dfs = []
    for fp in files:
        df = read_swarm_dnsapod(str(fp))
        df["source_file"] = fp.name
        dfs.append(df)

    all_df = pd.concat(dfs).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="first")]
    return all_df


# 統合
print("Reading all CDF files for SWARM-A ...")
all_df = collect_cdfs(data_dir="data/SWARM_A")
print(f"  Total before filter: {len(all_df):,}  ({all_df.index.min()} -> {all_df.index.max()})")

# DOY 20~80 でフィルタ
all_df = all_df[(all_df.index >= DATE_START) & (all_df.index <= DATE_END)]
print(f"  After DOY20-80 filter: {len(all_df):,}  ({all_df.index.min()} -> {all_df.index.max()})")

# 保存
out_path = Path("integrateddata/swarm_dnsapod_2018_DOY20-80.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)
all_df.to_parquet(out_path)
print(f"Saved: {out_path}")
