"""
check_2021_data_range.py
Check the min/max datetime and availability of variables in 2021 normalized datasets.
"""
from pathlib import Path
import pandas as pd

DN_DIR = Path("normalizeddata/2021")
for fp in DN_DIR.glob("*.parquet"):
    print(f"\n======================================")
    print(f"File: {fp.name}")
    print(f"======================================")
    df = pd.read_parquet(fp)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    print("Columns:", list(df.columns))
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
