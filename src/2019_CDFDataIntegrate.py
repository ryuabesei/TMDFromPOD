# =========================
# SWARM-A/B/C: CDF統合スクリプト (2019年SSW期間版)
# 対象期間: 2019-09-09 (DOY 252) ～ 2019-09-23 (DOY 266)
# =========================

from pathlib import Path
import numpy as np
import pandas as pd
from cdflib import CDF, cdfepoch

FILL = 0.99900e33

DATE_START = pd.Timestamp("2019-09-09", tz=None)
DATE_END   = pd.Timestamp("2019-09-23 23:59:59", tz=None)


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


def collect_cdfs(data_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(pattern))
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


def process_satellite(sat_char: str):
    sat_char_upper = sat_char.upper()
    print(f"\nProcessing SWARM-{sat_char_upper}...")

    data_dir = Path(f"data/SSW2019/SWARM_{sat_char_upper}")
    pattern = f"SW_OPER_DNS{sat_char_upper}POD_2__*.cdf"

    # 収集
    all_df = collect_cdfs(data_dir, pattern)
    print(f"  Total before filter: {len(all_df):,} ({all_df.index.min()} -> {all_df.index.max()})")

    # 期間フィルタ
    all_df = all_df[(all_df.index >= DATE_START) & (all_df.index <= DATE_END)]
    print(f"  After Sep 9-23 filter: {len(all_df):,} ({all_df.index.min()} -> {all_df.index.max()})")

    # 出力
    out_path = Path(f"integrateddata/2019/swarm_dns{sat_char.lower()}pod_2019_SSW.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(out_path)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    for sat in ["a", "b", "c"]:
        process_satellite(sat)
    print("\n✅ All 2019 satellites data integration completed.")
