# ==============================================================================
# 2019_NormalizationMISIS.py
# 目的: 2019年SSW期間（9/9〜9/23）の Swarm A, B, C データを MSIS モデルで正規化する。
# ==============================================================================

import sys
import os
from pathlib import Path

# src ディレクトリを path に追加して NormalizationMISIS_FIXED.py をインポート可能にする
sys.path.insert(0, str(Path(__file__).parent))

from NormalizationMISIS_FIXED import normalize

KPINDEX_CSV = Path("data/SSW2019/Kpindex/SW-All_2019-09-09_to_2019-09-23.csv")

JOBS = [
    # --- SWARM-A ---
    dict(
        swarm_parquet = Path("integrateddata/2019/swarm_dnsapod_2019_SSW.parquet"),
        out_parquet   = Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_SSW.parquet"),
        alt_ref_km    = 450.0,
        label         = "SWARM-A 2019 SSW (alt_ref=450km)",
    ),
    # --- SWARM-B (標準 510km) ---
    dict(
        swarm_parquet = Path("integrateddata/2019/swarm_dnsbpod_2019_SSW.parquet"),
        out_parquet   = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_SSW.parquet"),
        alt_ref_km    = 510.0,
        label         = "SWARM-B 2019 SSW (alt_ref=510km)",
    ),
    # --- SWARM-B (比較用 450km) ---
    dict(
        swarm_parquet = Path("integrateddata/2019/swarm_dnsbpod_2019_SSW.parquet"),
        out_parquet   = Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_SSW(450km).parquet"),
        alt_ref_km    = 450.0,
        label         = "SWARM-B 2019 SSW (alt_ref=450km)",
    ),
    # --- SWARM-C ---
    dict(
        swarm_parquet = Path("integrateddata/2019/swarm_dnscpod_2019_SSW.parquet"),
        out_parquet   = Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_SSW.parquet"),
        alt_ref_km    = 450.0,
        label         = "SWARM-C 2019 SSW (alt_ref=450km)",
    ),
]

if __name__ == "__main__":
    if not KPINDEX_CSV.exists():
        print(f"❌ Kpindex CSV が見つかりません: {KPINDEX_CSV}")
        sys.exit(1)

    for job in JOBS:
        print(f"\n>>> Running job: {job['label']}")
        normalize(
            swarm_parquet = job["swarm_parquet"],
            kpindex_csv   = KPINDEX_CSV,
            out_parquet   = job["out_parquet"],
            alt_ref_km    = job["alt_ref_km"],
        )

    print("\n" + "="*60)
    print("✅ 2019年データのすべての正規化処理が完了しました。")
    print("="*60)
