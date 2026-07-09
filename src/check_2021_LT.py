"""
check_2021_LT.py
Check the distribution and drift of Local Solar Time (LST) for Swarm-C and GRACE-FO in 2021 SSW.
"""
from pathlib import Path
import pandas as pd

JOBS = [
    ("SWARM-C", Path("normalizeddata/2021/swarm_dnscpod_2021_normalized.parquet")),
    ("GRACE-FO", Path("normalizeddata/2021/grace_fo_dns_2021_normalized.parquet")),
]

for label, fp in JOBS:
    print(f"\n======================================")
    print(f"{label}")
    print(f"======================================")
    df = pd.read_parquet(fp)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    print("lst_h describe:")
    print(df["lst_h"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    
    # 期間の最初と最後での LST
    t_min, t_max = df["datetime"].min(), df["datetime"].max()
    df_early = df[df["datetime"] <= t_min + pd.Timedelta(days=2)]
    df_late = df[df["datetime"] >= t_max - pd.Timedelta(days=2)]
    print(f"LST at start ({t_min.date()}): {df_early['lst_h'].median():.2f} h")
    print(f"LST at end ({t_max.date()}): {df_late['lst_h'].median():.2f} h")
