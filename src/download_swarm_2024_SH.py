"""
download_swarm_2024_SH.py

Purpose:
    Download SWARM-A/B/C DNS POD (Level 2) data for the
    2024 Southern Hemisphere SSW events (July–August 2024):
      - Analysis window: 2024-06-15 to 2024-08-25

    Uses viresclient to fetch data from the ESA VirES server.
    Saves daily parquet cache files, then combines per satellite.

Output:
    data/SSW2024_SH/SWARM_A/swarm_dnsapod_YYYYMMDD.parquet  (etc.)
    integrateddata/2024_SH/swarm_dns{a,b,c}pod_2024_SH_SSW.parquet
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from viresclient import SwarmRequest

SERVER = "https://vires.services/ows"
FILL   = 0.99900e33

# ─── Event definition ──────────────────────────────────────────────────────────
EVENT = dict(
    label   = "2024 SH SSW",
    start   = "2024-06-15",
    end     = "2024-08-25",
    out_dir = "data/SSW2024_SH",
    int_dir = "integrateddata/2024_SH",
)

# ─── Satellite definitions ─────────────────────────────────────────────────────
SATELLITES = [
    dict(label="SWARM-A", collection="SW_OPER_DNSAPOD_2_",
         subdir="SWARM_A", out_prefix="swarm_dnsapod"),
    dict(label="SWARM-B", collection="SW_OPER_DNSBPOD_2_",
         subdir="SWARM_B", out_prefix="swarm_dnsbpod"),
    dict(label="SWARM-C", collection="SW_OPER_DNSCPOD_2_",
         subdir="SWARM_C", out_prefix="swarm_dnscpod"),
]

MEASUREMENTS = [
    "Height_GD",
    "Latitude_GD",
    "Longitude_GD",
    "local_solar_time",
    "density",
    "density_orbitmean",
    "validity_flag",
]


def fetch_one_day(req: SwarmRequest, collection: str, date: pd.Timestamp) -> pd.DataFrame | None:
    start = date.strftime("%Y-%m-%dT00:00:00")
    end   = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    try:
        req.set_collection(collection)
        req.set_products(measurements=MEASUREMENTS, sampling_step="PT1S")
        result = req.get_between(start_time=start, end_time=end, asynchronous=False)
        df = result.as_dataframe()
        if df.empty:
            return None

        df = df.rename(columns={
            "Height_GD":        "altitude_m",
            "Latitude_GD":      "lat",
            "Longitude_GD":     "lon",
            "local_solar_time": "lst_h",
        })

        for col in ["density", "density_orbitmean", "altitude_m", "lat", "lon", "lst_h"]:
            if col in df.columns:
                df.loc[np.isclose(df[col].values.astype(float), FILL, rtol=1e-3), col] = np.nan

        if "validity_flag" in df.columns:
            df = df[df["validity_flag"] == 0].copy()

        df.index.name = "datetime"
        df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df

    except Exception as e:
        print(f"    WARNING: {date.date()} failed: {e}")
        return None


def download_satellite(req: SwarmRequest, sat: dict) -> None:
    raw_dir = Path(EVENT["out_dir"]) / sat["subdir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(EVENT["start"], EVENT["end"], freq="D")
    all_dfs = []
    n_ok = 0

    print(f"\n  [{sat['label']}]  {EVENT['label']}  ({EVENT['start']} → {EVENT['end']})")
    print(f"  Fetching {len(dates)} days from VirES ...")

    for date in dates:
        cache = raw_dir / f"{sat['out_prefix']}_{date.strftime('%Y%m%d')}.parquet"
        if cache.exists():
            df_day = pd.read_parquet(cache)
            print(f"    {date.strftime('%Y-%m-%d')}  [cached] rows={len(df_day):,}", end="\r")
        else:
            df_day = fetch_one_day(req, sat["collection"], date)
            if df_day is not None and not df_day.empty:
                df_day.to_parquet(cache, index=False)

        if df_day is not None and not df_day.empty:
            all_dfs.append(df_day)
            n_ok += 1

    print(f"\n  Done: {n_ok}/{len(dates)} days with data")

    if not all_dfs:
        print(f"  ERROR: No data for {sat['label']}")
        return

    combined = pd.concat(all_dfs, ignore_index=True).sort_values("datetime")
    combined = combined.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    int_dir = Path(EVENT["int_dir"])
    int_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = int_dir / f"{sat['out_prefix']}_2024_SH_SSW.parquet"
    combined.to_parquet(out_parquet, index=False)
    print(f"  Saved: {out_parquet}  ({len(combined):,} rows)")


def main():
    print("=== SWARM DNS POD Download: 2024 SH SSW ===")
    print(f"Server: {SERVER}")
    print(f"Period: {EVENT['start']} → {EVENT['end']}\n")

    req = SwarmRequest(SERVER)
    for sat in SATELLITES:
        download_satellite(req, sat)

    print("\n\nAll downloads complete.")


if __name__ == "__main__":
    main()
