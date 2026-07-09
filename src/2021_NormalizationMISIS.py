"""
2021_NormalizationMISIS.py

Purpose:
    Perform NRLMSIS 2.1 normalization for Swarm-C and GRACE-FO integrated datasets
    covering the period 2020-12-25 to 2021-02-05 (2021 SSW).
    Outputs both standard and LT-removed normalized datasets.

    Reference altitude for both satellites is set to 450.0 km (F107_ref=70.0, Ap_ref=4.0).

Output:
    normalizeddata/2021/swarm_dnscpod_2021_normalized.parquet
    normalizeddata/2021/grace_fo_dns_2021_normalized.parquet
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add src dir to path to import common normalization library
sys.path.insert(0, str(Path(__file__).parent))
from NormalizationMISIS_FIXED import normalize

KPINDEX_CSV = Path("data/SSW2021/Kpindex/SW-20201220_20210228.csv")

JOBS = [
    # --- SWARM-C ---
    dict(
        swarm_parquet          = Path("integrateddata/2021/swarm_dnscpod_2021_integrated.parquet"),
        out_parquet            = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized.parquet"),
        out_parquet_lt_removed = Path("normalizeddata/2021/swarm_dnscpod_2021_normalized_with_LT_removed.parquet"),
        alt_ref_km             = 450.0,
        label                  = "SWARM-C 2021 SSW (alt_ref=450km)",
    ),
    # --- GRACE-FO ---
    dict(
        swarm_parquet          = Path("integrateddata/2021/grace_fo_dns_2021_integrated.parquet"),
        out_parquet            = Path("normalizeddata/2021/grace_fo_dns_2021_normalized.parquet"),
        out_parquet_lt_removed = Path("normalizeddata/2021/grace_fo_dns_2021_normalized_with_LT_removed.parquet"),
        alt_ref_km             = 450.0,
        label                  = "GRACE-FO 2021 SSW (alt_ref=450km)",
    ),
]

def main():
    if not KPINDEX_CSV.exists():
        print(f"❌ Kpindex CSV not found: {KPINDEX_CSV}")
        sys.exit(1)

    for job in JOBS:
        print(f"\n>>> Running normalization job: {job['label']}")
        normalize(
            swarm_parquet          = job["swarm_parquet"],
            kpindex_csv            = KPINDEX_CSV,
            out_parquet            = job["out_parquet"],
            alt_ref_km             = job["alt_ref_km"],
            out_parquet_lt_removed = job["out_parquet_lt_removed"],
        )

    print("\n" + "="*60)
    print("✅ 2021 normalization pipeline completed successfully.")
    print("="*60)

if __name__ == "__main__":
    main()
