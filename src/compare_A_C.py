import pandas as pd
import numpy as np

def main():
    print("Loading Swarm-A data...")
    df_a = pd.read_parquet("normalizeddata/swarm_dnsapod_2018_normalized_DOY20-80.parquet")
    print("Loading Swarm-C data...")
    df_c = pd.read_parquet("normalizeddata/swarm_dnscpod_2018_normalized_DOY20-80.parquet")
    
    # Filter for the same time period and lat
    df_a["datetime"] = pd.to_datetime(df_a["datetime"], utc=True)
    df_c["datetime"] = pd.to_datetime(df_c["datetime"], utc=True)
    
    # Let's check mean, std
    print("\n--- Basic Stats (Normalized Density) ---")
    print(f"Swarm-A Mean: {df_a['density_norm'].mean():.4e}, Std: {df_a['density_norm'].std():.4e}")
    print(f"Swarm-C Mean: {df_c['density_norm'].mean():.4e}, Std: {df_c['density_norm'].std():.4e}")
    
    # Merge by datetime to compare point-by-point
    # They should have very similar timestamps, but not perfectly identical.
    # Let's round to nearest minute
    df_a["dt_min"] = df_a["datetime"].dt.floor("Min")
    df_c["dt_min"] = df_c["datetime"].dt.floor("Min")
    
    # Group by minute just in case
    g_a = df_a.groupby("dt_min")["density_norm"].mean().rename("dens_A")
    g_c = df_c.groupby("dt_min")["density_norm"].mean().rename("dens_C")
    
    merged = pd.concat([g_a, g_c], axis=1).dropna()
    print(f"\n--- Point-by-point Comparison (per minute) ---")
    print(f"Matched points: {len(merged):,}")
    
    diff = merged["dens_A"] - merged["dens_C"]
    rel_diff = (diff / merged["dens_A"]).abs() * 100
    
    print(f"Mean Absolute Diff: {diff.abs().mean():.4e} kg/m³")
    print(f"Mean Relative Diff: {rel_diff.mean():.2f} %")
    print(f"Correlation: {merged['dens_A'].corr(merged['dens_C']):.4f}")

if __name__ == "__main__":
    main()
