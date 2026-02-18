import pandas as pd
from pathlib import Path

FILES = {
    "SWARM-A": Path("normalizeddata/swarm_dnsapod_2018_normalized.parquet"),
    "SWARM-B": Path("normalizeddata/swarm_dnsbpod_2018_normalized.parquet"),
    "SWARM-C": Path("normalizeddata/swarm_dnscpod_2018_normalized.parquet"),
}

for name, fp in FILES.items():
    print("="*60)
    print(f"{name}")
    print("File:", fp)

    df = pd.read_parquet(fp)

    print("\n■ Columns:")
    print(df.columns.tolist())

    print("\n■ dtypes:")
    print(df.dtypes)

    print("\n■ Head:")
    print(df.head())

    print("\n■ Index type:")
    print(type(df.index))
