"""
check_daily_lt_SWARM-A.py

Purpose:
    Compute and display the representative Local Time (LT) of Swarm-A 
    for each Day of Year (DOY 20–80) under the extended LT sectors:
    - Morning (04–14 LT)
    - Evening (16–24 LT)
"""

from pathlib import Path
import pandas as pd

PARQUET = Path("normalizeddata/swarm_dnsapod_2018_normalized_DOY20-80.parquet")
DOY_START, DOY_END = 20, 80
LAT_MIN, LAT_MAX = -60.0, 60.0

SECTOR_MORNING = (4.0, 14.0)
SECTOR_EVENING = (16.0, 24.0)

def main():
    print(f"Loading data from {PARQUET}...")
    df = pd.read_parquet(PARQUET)
    
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["DOY_int"] = df["datetime"].dt.dayofyear
    
    # Apply standard filters
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)]
    df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]
    
    rows = []
    for doy in range(DOY_START, DOY_END + 1):
        df_day = df[df["DOY_int"] == doy]
        
        # Morning LT
        m_data = df_day[(df_day["lst_h"] >= SECTOR_MORNING[0]) & (df_day["lst_h"] < SECTOR_MORNING[1])]
        if len(m_data) > 0:
            m_lt_median = m_data["lst_h"].median()
            m_count = len(m_data)
        else:
            m_lt_median = float("nan")
            m_count = 0
            
        # Evening LT
        e_data = df_day[(df_day["lst_h"] >= SECTOR_EVENING[0]) & (df_day["lst_h"] < SECTOR_EVENING[1])]
        if len(e_data) > 0:
            e_lt_median = e_data["lst_h"].median()
            e_count = len(e_data)
        else:
            e_lt_median = float("nan")
            e_count = 0
            
        rows.append({
            "DOY": doy,
            "Morning_LT": m_lt_median,
            "Morning_Count": m_count,
            "Evening_LT": e_lt_median,
            "Evening_Count": e_count
        })
        
    df_lt = pd.DataFrame(rows)
    
    print("\n" + "="*70)
    print(f"{'DOY':^5} | {'Morning LT (Median)':^20} | {'Morning N':^10} | {'Evening LT (Median)':^20} | {'Evening N':^10}")
    print("="*70)
    for _, r in df_lt.iterrows():
        m_lt_str = f"{r['Morning_LT']:.2f} h" if not pd.isna(r["Morning_LT"]) else "N/A"
        e_lt_str = f"{r['Evening_LT']:.2f} h" if not pd.isna(r["Evening_LT"]) else "N/A"
        print(f"{int(r['DOY']):^5} | {m_lt_str:^20} | {int(r['Morning_Count']):^10} | {e_lt_str:^20} | {int(r['Evening_Count']):^10}")
    print("="*70)

    # Save to a CSV for easy reference
    out_csv = Path("Figure/swarm-a_daily_lt_doy20-80.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_lt.to_csv(out_csv, index=False)
    print(f"\n✅ Saved daily LT values to: {out_csv}")

if __name__ == "__main__":
    main()
