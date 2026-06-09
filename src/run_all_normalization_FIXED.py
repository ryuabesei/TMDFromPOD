"""
run_all_normalization_FIXED.py

修正版正規化を全衛星・全出力ファイルに対して実行する。

対象:
  1. SWARM-A DOY20-80       → swarm_dnsapod_2018_normalized_DOY20-80.parquet
  2. SWARM-B DOY20-80       → swarm_dnsbpod_2018_normalized_DOY20-80.parquet
  3. SWARM-B DOY20-80 450km → swarm_dnsbpod_2018_normalized_DOY20-80(450km).parquet
  4. SWARM-C DOY20-80       → swarm_dnscpod_2018_normalized_DOY20-80.parquet
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from NormalizationMISIS_FIXED import normalize

KPINDEX_CSV = Path("data/Kpindex/SW-20180120_20180320.csv")

JOBS = [
    dict(
        swarm_parquet = Path("integrateddata/swarm_dnsapod_2018_DOY20-80.parquet"),
        out_parquet   = Path("normalizeddata/swarm_dnsapod_2018_normalized_DOY20-80.parquet"),
        alt_ref_km    = 450.0,
        label         = "SWARM-A DOY20-80",
    ),
    dict(
        swarm_parquet = Path("integrateddata/swarm_dnsbpod_2018_DOY20-80.parquet"),
        out_parquet   = Path("normalizeddata/swarm_dnsbpod_2018_normalized_DOY20-80.parquet"),
        alt_ref_km    = 450.0,
        label         = "SWARM-B DOY20-80 (alt_ref=450km)",
    ),
    dict(
        swarm_parquet = Path("integrateddata/swarm_dnsbpod_2018_DOY20-80.parquet"),
        out_parquet   = Path("normalizeddata/swarm_dnsbpod_2018_normalized_DOY20-80(450km).parquet"),
        alt_ref_km    = 450.0,
        label         = "SWARM-B DOY20-80 (450km parquet)",
    ),
    dict(
        swarm_parquet = Path("integrateddata/swarm_dnscpod_2018_DOY20-80.parquet"),
        out_parquet   = Path("normalizeddata/swarm_dnscpod_2018_normalized_DOY20-80.parquet"),
        alt_ref_km    = 450.0,
        label         = "SWARM-C DOY20-80",
    ),
]

if __name__ == "__main__":
    for job in JOBS:
        print(f"\n>>> {job['label']}")
        normalize(
            swarm_parquet = job["swarm_parquet"],
            kpindex_csv   = KPINDEX_CSV,
            out_parquet   = job["out_parquet"],
            alt_ref_km    = job["alt_ref_km"],
        )

    print("\n" + "="*60)
    print("✅ 全衛星の正規化が完了しました。")
    print("="*60)
