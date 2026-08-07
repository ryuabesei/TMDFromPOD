"""
normalize_swarm_2024_SH.py

Purpose:
    Apply MSIS normalization to SWARM-A/B/C DNS POD data for the
    2024 Southern Hemisphere SSW (June–August 2024).
    Follows the same pipeline as normalize_swarm_2024_2026.py.

    rho_norm = rho_obs * (rho_MSIS(ref_cond) / rho_MSIS(real_cond))

    ref_cond : alt=450 km, F10.7=70, Ap=4 (≈Kp=1)
    real_cond: obs alt, real F10.7 (81-day centered), real Ap history

Input:
    integrateddata/2024_SH/swarm_dns{a,b,c}pod_2024_SH_SSW.parquet
    data/SSW2024_SH/Kpindex/*.csv  (or existing 2024 Kpindex CSV)

Output:
    normalizeddata/2024_SH/swarm_dns{a,b,c}pod_2024_SH_normalized_with_LT.parquet
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ─── Normalization constants ────────────────────────────────────────────────
F107_REF   = 70.0
AP_REF     = 4.0
ALT_REF_KM = 450.0

# ─── Kp/F10.7 source: reuse existing 2024 index if SH-specific one absent ──
KP_CANDIDATES = [
    Path("data/SSW2024_SH/Kpindex/SW-20240601_20240831.csv"),
    Path("data/SSW2024/Kpindex/SW-20231201_20240331.csv"),   # fallback (covers some months)
]

SATS = [
    dict(label="SWARM-A", prefix="swarm_dnsapod"),
    dict(label="SWARM-B", prefix="swarm_dnsbpod"),
    dict(label="SWARM-C", prefix="swarm_dnscpod"),
]


# ─── Utilities ───────────────────────────────────────────────────────────────

def build_ap7_matrix(df: pd.DataFrame) -> np.ndarray:
    hours    = df["datetime"].dt.hour.to_numpy()
    k        = (hours // 3).astype(int)
    ap_today = df[[f"AP{i}" for i in range(1, 9)]].to_numpy(float)
    ap_prev  = df[[f"AP{i}_prev" for i in range(1, 9)]].to_numpy(float)
    ap_avg   = df["AP_AVG"].to_numpy(float)
    N        = len(df)
    idxN     = np.arange(N)

    def get_ap_shift(shift: int) -> np.ndarray:
        idx       = k - shift
        use_today = idx >= 0
        val_today = ap_today[idxN, np.clip(idx, 0, 7)]
        idx_prev  = idx + 8
        val_prev  = ap_prev[idxN, np.clip(idx_prev, 0, 7)]
        return np.where(use_today, val_today, val_prev)

    return np.column_stack([
        ap_avg,
        get_ap_shift(0), get_ap_shift(1), get_ap_shift(2),
        get_ap_shift(3), get_ap_shift(4), get_ap_shift(5),
    ]).astype(float)


def msis_density(time_utc, lat, lon, alt_km, f107s, f107as, aps) -> np.ndarray:
    from pymsis import msis
    out = msis.run(time_utc, lon, lat, alt_km, f107s=f107s, f107as=f107as, aps=aps, version=2.1)
    out = np.asarray(out)
    rho = out[:, 0].astype(float) if out.ndim >= 2 else out.astype(float)
    if np.any(rho <= 0) or np.any(~np.isfinite(rho)):
        raise ValueError("MSIS returned invalid density values.")
    return rho


def load_kp(kp_csv: Path) -> pd.DataFrame:
    ap_cols = [f"AP{i}" for i in range(1, 9)]
    keep    = ["DATE", "F10.7_ADJ", "F10.7_ADJ_CENTER81", "AP_AVG"] + ap_cols
    df_geo  = pd.read_csv(kp_csv, parse_dates=["DATE"])[keep].copy()
    df_geo.rename(columns={"F10.7_ADJ": "F107", "F10.7_ADJ_CENTER81": "F107A"}, inplace=True)
    df_geo["DATE"] = pd.to_datetime(df_geo["DATE"], utc=True).dt.floor("D").dt.tz_localize(None)
    return df_geo


def download_kp_if_needed() -> Path | None:
    """Download Kp/F10.7 for 2024-06 to 2024-09 if not already available."""
    for p in KP_CANDIDATES:
        if p.exists():
            print(f"  Using Kp index: {p}")
            return p

    # Try to download via the existing script pattern
    print("  [INFO] No SH-period Kp file found. Downloading 2024-06 to 2024-09 ...")
    try:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "src/download_kpindex_2024_SH.py"],
            capture_output=True, text=True, cwd=Path(".").resolve()
        )
        if result.returncode == 0:
            for p in KP_CANDIDATES:
                if p.exists():
                    return p
    except Exception as e:
        print(f"  [WARN] Auto-download failed: {e}")

    print("  [ERROR] No Kp CSV found. Cannot normalize. Please provide:")
    for p in KP_CANDIDATES:
        print(f"    {p}")
    return None


def normalize_one(sat: dict, kp_csv: Path) -> None:
    prefix   = sat["prefix"]
    label    = sat["label"]
    int_path = Path(f"integrateddata/2024_SH/{prefix}_2024_SH_SSW.parquet")
    out_path = Path(f"normalizeddata/2024_SH/{prefix}_2024_SH_normalized_with_LT.parquet")

    if not int_path.exists():
        print(f"  SKIP {label}: {int_path} not found")
        return

    print(f"  {label} ... ", end="", flush=True)

    df = pd.read_parquet(int_path).copy()

    if "datetime" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "datetime"})
        else:
            raise KeyError("No datetime column found")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    rename_map = {
        "Height_GD":        "altitude_m",
        "Latitude_GD":      "lat",
        "Longitude_GD":     "lon",
        "local_solar_time": "lst_h",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    if "altitude_m" in df.columns:
        df["alt_km"] = df["altitude_m"].astype(float) / 1000.0
    elif "alt_km" not in df.columns:
        raise KeyError("No altitude column found")

    if "validity_flag" in df.columns:
        df = df[df["validity_flag"] == 0].copy()

    if "lat" in df.columns:
        df = df[df["lat"].abs() <= 60.0].copy()

    # Kp / F10.7
    df_geo = load_kp(kp_csv)

    ap_cols     = [f"AP{i}" for i in range(1, 9)]
    df_geo_prev = df_geo[["DATE", "AP_AVG"] + ap_cols].copy()
    df_geo_prev["DATE"] = df_geo_prev["DATE"] + pd.Timedelta(days=1)
    df_geo_prev = df_geo_prev.rename(columns={c: f"{c}_prev" for c in ["AP_AVG"] + ap_cols})

    df["DATE"] = df["datetime"].dt.floor("D").dt.tz_localize(None)
    df = df.merge(df_geo,      on="DATE", how="left")
    df = df.merge(df_geo_prev, on="DATE", how="left")

    for i in range(1, 9):
        df[f"AP{i}_prev"] = df[f"AP{i}_prev"].fillna(df[f"AP{i}"])

    df = df.dropna(subset=["F107", "F107A", "AP_AVG"]).copy()

    if df.empty:
        print("no valid rows after merge — skipping")
        return

    rho_obs     = df["density"].astype(float).to_numpy()
    lat         = df["lat"].astype(float).to_numpy()
    lon         = df["lon"].astype(float).to_numpy()
    alt_km      = df["alt_km"].astype(float).to_numpy()
    time_utc    = pd.to_datetime(df["datetime"], utc=True).to_numpy()
    f107s_real  = df["F107"].astype(float).to_numpy()
    f107as_real = df["F107A"].astype(float).to_numpy()
    aps_real    = build_ap7_matrix(df)

    f107s_ref  = np.full_like(f107s_real,  F107_REF,   dtype=float)
    f107as_ref = np.full_like(f107as_real, F107_REF,   dtype=float)
    aps_ref    = np.full_like(aps_real,    AP_REF,     dtype=float)
    alt_ref_km = np.full_like(alt_km,      ALT_REF_KM, dtype=float)

    rho_model_real = msis_density(time_utc, lat, lon, alt_km,     f107s_real, f107as_real, aps_real)
    rho_model_ref  = msis_density(time_utc, lat, lon, alt_ref_km, f107s_ref,  f107as_ref,  aps_ref)

    ratio = rho_model_ref / rho_model_real
    df["density_norm"]                   = rho_obs * ratio
    df["norm_ratio_model_ref_over_real"] = ratio
    df["rho_ratio"]                      = rho_obs * ratio
    df["density_ratio_msis"]             = rho_obs / rho_model_real
    df["date"]                           = df["datetime"].dt.date

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"done  ({len(df):,} rows) → {out_path}")


def main() -> None:
    print("=== MSIS Normalization: 2024 SH SSW ===\n")
    kp_csv = download_kp_if_needed()
    if kp_csv is None:
        return
    for sat in SATS:
        normalize_one(sat, kp_csv)
    print("\nAll done.")


if __name__ == "__main__":
    main()
