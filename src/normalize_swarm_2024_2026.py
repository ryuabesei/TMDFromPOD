"""
normalize_swarm_2024_2026.py

Purpose:
    Apply MSIS normalization to SWARM-A/B/C DNS POD data for 2024 and 2026 SSW events.
    Follows the exact same pipeline as SWARM-A_NormalizationMISIS_DOY20-80.py.

    rho_norm = rho_obs * (rho_MSIS(ref_cond) / rho_MSIS(real_cond))

    ref_cond : alt=450km, F10.7=70, Ap=4 (≈Kp=1)
    real_cond: obs alt, real F10.7 (81-day centered), real Ap history

Input:
    integrateddata/2024/swarm_dns{a,b,c}pod_2024_SSW.parquet
    data/SSW2024/Kpindex/SW-20231201_20240331.csv
    (same pattern for 2026)

Output:
    normalizeddata/2024/swarm_dns{a,b,c}pod_2024_normalized_with_LT.parquet
    normalizeddata/2026/swarm_dns{a,b,c}pod_2026_normalized_with_LT.parquet
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ─── Normalization constants (same as existing pipeline) ─────────────────────
F107_REF    = 70.0
AP_REF      = 4.0
ALT_REF_KM  = 450.0
FILL        = 0.99900e33

# ─── Event + Satellite definitions ───────────────────────────────────────────
EVENTS = [
    dict(
        label   = "2024",
        kp_csv  = Path("data/SSW2024/Kpindex/SW-20231201_20240331.csv"),
        sats = [
            dict(label="SWARM-A",  prefix="swarm_dnsapod"),
            dict(label="SWARM-B",  prefix="swarm_dnsbpod"),
            dict(label="SWARM-C",  prefix="swarm_dnscpod"),
        ],
    ),
    dict(
        label   = "2026",
        kp_csv  = Path("data/SSW2026/Kpindex/SW-20251201_20260430.csv"),
        sats = [
            dict(label="SWARM-A",  prefix="swarm_dnsapod"),
            dict(label="SWARM-B",  prefix="swarm_dnsbpod"),
            dict(label="SWARM-C",  prefix="swarm_dnscpod"),
        ],
    ),
]

# ─── Utilities (same as existing scripts) ────────────────────────────────────

def build_ap7_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build ap[0..6] array for MSIS input from 3-hourly Ap columns."""
    hours   = df["datetime"].dt.hour.to_numpy()
    k       = (hours // 3).astype(int)
    ap_today = df[[f"AP{i}" for i in range(1, 9)]].to_numpy(float)
    ap_prev  = df[[f"AP{i}_prev" for i in range(1, 9)]].to_numpy(float)
    ap_avg   = df["AP_AVG"].to_numpy(float)
    N        = len(df)
    idxN     = np.arange(N)

    def get_ap_shift(shift):
        idx      = k - shift
        use_today = idx >= 0
        val_today = ap_today[idxN, np.clip(idx, 0, 7)]
        idx_prev  = idx + 8
        val_prev  = ap_prev[idxN, np.clip(idx_prev, 0, 7)]
        return np.where(use_today, val_today, val_prev)

    ap7 = np.column_stack([
        ap_avg,
        get_ap_shift(0), get_ap_shift(1), get_ap_shift(2),
        get_ap_shift(3), get_ap_shift(4), get_ap_shift(5),
    ]).astype(float)
    return ap7


def msis_density(time_utc, lat, lon, alt_km, f107s, f107as, aps) -> np.ndarray:
    from pymsis import msis
    out = msis.run(time_utc, lon, lat, alt_km, f107s=f107s, f107as=f107as, aps=aps, version=2.1)
    out = np.asarray(out)
    rho = out[:, 0].astype(float) if out.ndim >= 2 else out.astype(float)
    if np.any(rho <= 0) or np.any(~np.isfinite(rho)):
        raise ValueError("MSIS returned invalid density values.")
    return rho


def load_kp(kp_csv: Path) -> pd.DataFrame:
    ap_cols  = [f"AP{i}" for i in range(1, 9)]
    keep     = ["DATE", "F10.7_ADJ", "F10.7_ADJ_CENTER81", "AP_AVG"] + ap_cols
    df_geo   = pd.read_csv(kp_csv, parse_dates=["DATE"])[keep].copy()
    df_geo.rename(columns={"F10.7_ADJ": "F107", "F10.7_ADJ_CENTER81": "F107A"}, inplace=True)
    df_geo["DATE"] = pd.to_datetime(df_geo["DATE"], utc=True).dt.floor("D").dt.tz_localize(None)
    return df_geo


def normalize_one(year: str, sat: dict, kp_csv: Path) -> None:
    prefix  = sat["prefix"]
    label   = sat["label"]
    int_path = Path(f"integrateddata/{year}/{prefix}_{year}_SSW.parquet")
    out_path = Path(f"normalizeddata/{year}/{prefix}_{year}_normalized_with_LT.parquet")

    if not int_path.exists():
        print(f"  SKIP {label}: {int_path} not found")
        return

    print(f"  {label} ... ", end="", flush=True)

    # Load SWARM data
    df = pd.read_parquet(int_path).copy()

    # Ensure datetime column
    if "datetime" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "datetime"})
        else:
            raise KeyError("No datetime column found")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Column mapping: viresclient uses different names
    rename_map = {
        "Height_GD":    "altitude_m",
        "Latitude_GD":  "lat",
        "Longitude_GD": "lon",
        "local_solar_time": "lst_h",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Altitude: viresclient returns meters
    if "altitude_m" in df.columns:
        df["alt_km"] = df["altitude_m"].astype(float) / 1000.0
    elif "alt_km" in df.columns:
        pass
    else:
        raise KeyError("No altitude column found")

    # Filter validity flag
    if "validity_flag" in df.columns:
        df = df[df["validity_flag"] == 0].copy()

    # Filter high latitudes (>60°)
    if "lat" in df.columns:
        df = df[df["lat"].abs() <= 60.0].copy()

    # Load Kp/F10.7
    df_geo  = load_kp(kp_csv)

    # Prev-day Ap
    ap_cols = [f"AP{i}" for i in range(1, 9)]
    df_geo_prev = df_geo[["DATE", "AP_AVG"] + ap_cols].copy()
    df_geo_prev["DATE"] = df_geo_prev["DATE"] + pd.Timedelta(days=1)
    df_geo_prev = df_geo_prev.rename(columns={c: f"{c}_prev" for c in ["AP_AVG"] + ap_cols})

    # Merge
    df["DATE"] = df["datetime"].dt.floor("D").dt.tz_localize(None)
    df = df.merge(df_geo,      on="DATE", how="left")
    df = df.merge(df_geo_prev, on="DATE", how="left")

    for i in range(1, 9):
        c = f"AP{i}"
        df[f"{c}_prev"] = df[f"{c}_prev"].fillna(df[c])

    # Drop rows where F107 / AP_AVG is missing (outside index range)
    df = df.dropna(subset=["F107", "F107A", "AP_AVG"]).copy()

    if df.empty:
        print("no valid rows after merge — skipping")
        return

    # Build MSIS inputs
    rho_obs     = df["density"].astype(float).to_numpy()
    lat         = df["lat"].astype(float).to_numpy()
    lon         = df["lon"].astype(float).to_numpy()
    alt_km      = df["alt_km"].astype(float).to_numpy()
    time_utc    = pd.to_datetime(df["datetime"], utc=True).to_numpy()
    f107s_real  = df["F107"].astype(float).to_numpy()
    f107as_real = df["F107A"].astype(float).to_numpy()
    aps_real    = build_ap7_matrix(df)

    f107s_ref   = np.full_like(f107s_real,  F107_REF, dtype=float)
    f107as_ref  = np.full_like(f107as_real, F107_REF, dtype=float)
    aps_ref     = np.full_like(aps_real,    AP_REF,   dtype=float)
    alt_ref_km  = np.full_like(alt_km,      ALT_REF_KM, dtype=float)

    rho_model_real = msis_density(time_utc, lat, lon, alt_km,    f107s_real,  f107as_real, aps_real)
    rho_model_ref  = msis_density(time_utc, lat, lon, alt_ref_km, f107s_ref,   f107as_ref,  aps_ref)

    ratio = rho_model_ref / rho_model_real
    df["density_norm"]                  = rho_obs * ratio
    df["norm_ratio_model_ref_over_real"] = ratio
    df["rho_ratio"]                     = rho_obs * ratio   # alias used in some scripts
    df["density_ratio_msis"]            = rho_obs / rho_model_real  # dimensionless ratio
    df["date"]                          = df["datetime"].dt.date

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"done  ({len(df):,} rows) → {out_path}")


def main():
    print("=== MSIS Normalization: 2024 & 2026 SSW ===\n")
    for event in EVENTS:
        year   = event["label"]
        kp_csv = event["kp_csv"]
        if not kp_csv.exists():
            print(f"[{year}] Kp CSV not found: {kp_csv} — run download_kpindex_2024_2026.py first")
            continue
        print(f"[{year}]")
        for sat in event["sats"]:
            normalize_one(year, sat, kp_csv)
    print("\nAll done.")


if __name__ == "__main__":
    main()
