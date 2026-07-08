"""
calc_non_ssw_medians_2019.py
Calculate the median of density_ratio_msis during non-SSW reference periods (08/20-08/26 & 09/20-09/23) for 2019.
[Updated] Swarm-A/C LT sector updated to 2.5-8.5h and 14.5-20.5h.
"""
from pathlib import Path
import pandas as pd
import numpy as np

LT_DAWN_DUSK = [
    {"label": "Dawn (LT 2.5-8.5h)", "min": 2.5, "max": 8.5},
    {"label": "Dusk (LT 14.5-20.5h)", "min": 14.5, "max": 20.5}
]
LT_MIDNIGHT_NOON = [
    {"label": "Midnight (LT 0-4h)", "min": 0, "max": 4},
    {"label": "Noon (LT 12-16h)", "min": 12, "max": 16}
]

SATELLITES = [
    {
        "label": "SWARM-A",
        "parquet": Path("normalizeddata/2019/swarm_dnsapod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        "lt_sectors": LT_DAWN_DUSK
    },
    {
        "label": "SWARM-B",
        "parquet": Path("normalizeddata/2019/swarm_dnsbpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        "lt_sectors": LT_MIDNIGHT_NOON
    },
    {
        "label": "SWARM-C",
        "parquet": Path("normalizeddata/2019/swarm_dnscpod_2019_normalized_with_LT_removed_SSW_extended.parquet"),
        "lt_sectors": LT_DAWN_DUSK
    }
]

LAT_BANDS = [
    ("High (40-60°)", 40.0, 60.0),
    ("Mid (20-40°)", 20.0, 40.0),
    ("Low (0-20°)", 0.0, 20.0),
]

DATE_REF1_START = pd.Timestamp("2019-08-20", tz="UTC")
DATE_REF1_END   = pd.Timestamp("2019-08-26", tz="UTC")
DATE_REF2_START = pd.Timestamp("2019-09-20", tz="UTC")
DATE_REF2_END   = pd.Timestamp("2019-09-23", tz="UTC")

VALUE_COL = "density_ratio_msis"

def main():
    results = []
    for sat in SATELLITES:
        df = pd.read_parquet(sat["parquet"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        
        # 参照期間フィルタリング
        mask_ref = (
            ((df["datetime"] >= DATE_REF1_START) & (df["datetime"] <= DATE_REF1_END)) |
            ((df["datetime"] >= DATE_REF2_START) & (df["datetime"] <= DATE_REF2_END))
        )
        df_ref = df[mask_ref].copy()
        df_ref["date"] = df_ref["datetime"].dt.normalize()
        
        for lt in sat["lt_sectors"]:
            df_lt = df_ref[(df_ref["lst_h"] >= lt["min"]) & (df_ref["lst_h"] < lt["max"])]
            
            for band_name, lat_lo, lat_hi in LAT_BANDS:
                mask_lat = (df_lt["lat"].abs() >= lat_lo) & (df_lt["lat"].abs() < lat_hi)
                sub = df_lt[mask_lat]
                
                # Daily median first, then median of those (matching plot logic)
                daily_medians = sub.groupby("date")[VALUE_COL].median()
                if len(daily_medians) > 0:
                    ref_val = daily_medians.median()
                else:
                    ref_val = np.nan
                
                results.append({
                    "Satellite": sat["label"],
                    "LT Sector": lt["label"],
                    "Latitude Band": band_name,
                    "Ref Median (rho_ratio)": ref_val
                })
                
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
