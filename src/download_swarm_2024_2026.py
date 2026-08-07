"""
download_swarm_2024_2026.py

Purpose:
    Download SWARM-A/B/C DNS POD (Level 2) data for:
      - 2024 NH SSW (peak ~2024-01-16)  : 2023-12-22 to 2024-02-29
      - 2026 NH SSW (peak ~2026-02-10)  : 2025-12-16 to 2026-03-31

    Uses viresclient to fetch data from the ESA VirES server.
    Saves daily CDF-equivalent data as individual daily parquet files,
    then combines them per satellite per event.

Output:
    data/SSW2024/SWARM_A/sw_dnsapod_YYYYMMDD.parquet  (etc.)
    data/SSW2026/SWARM_A/sw_dnsapod_YYYYMMDD.parquet  (etc.)
    integrateddata/2024/swarm_dns{a,b,c}pod_2024_SSW.parquet
    integrateddata/2026/swarm_dns{a,b,c}pod_2026_SSW.parquet
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from viresclient import SwarmRequest

SERVER = "https://vires.services/ows"
FILL   = 0.99900e33

# ─── Event definitions ────────────────────────────────────────────────────────
EVENTS = [
    dict(
        label     = "2024 NH SSW",
        start     = "2023-12-22",
        end       = "2024-02-29",
        out_dir   = "data/SSW2024",
        int_dir   = "integrateddata/2024",
    ),
    dict(
        label     = "2026 NH SSW",
        start     = "2025-12-16",
        end       = "2026-03-31",
        out_dir   = "data/SSW2026",
        int_dir   = "integrateddata/2026",
    ),
]

# ─── Satellite definitions ────────────────────────────────────────────────────
SATELLITES = [
    dict(
        label      = "SWARM-A",
        collection = "SW_OPER_DNSAPOD_2_",
        subdir     = "SWARM_A",
        out_prefix = "swarm_dnsapod",
    ),
    dict(
        label      = "SWARM-B",
        collection = "SW_OPER_DNSBPOD_2_",
        subdir     = "SWARM_B",
        out_prefix = "swarm_dnsbpod",
    ),
    dict(
        label      = "SWARM-C",
        collection = "SW_OPER_DNSCPOD_2_",
        subdir     = "SWARM_C",
        out_prefix = "swarm_dnscpod",
    ),
]

MEASUREMENTS = [
    "Height_GD",        # altitude [m]
    "Latitude_GD",      # geographic latitude
    "Longitude_GD",     # geographic longitude
    "local_solar_time", # LT [h]
    "density",          # thermospheric mass density [kg/m³]
    "density_orbitmean",
    "validity_flag",
]


def fetch_one_day(req: SwarmRequest, collection: str, date: pd.Timestamp) -> pd.DataFrame | None:
    """Fetch one day of data from VirES and return as DataFrame."""
    start = date.strftime("%Y-%m-%dT00:00:00")
    end   = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")

    try:
        req.set_collection(collection)
        req.set_products(measurements=MEASUREMENTS, sampling_step="PT1S")
        result = req.get_between(start_time=start, end_time=end, asynchronous=False)
        df = result.as_dataframe()
        if df.empty:
            return None

        # Rename columns to match existing pipeline
        df = df.rename(columns={
            "Height_GD":        "altitude_m",
            "Latitude_GD":      "lat",
            "Longitude_GD":     "lon",
            "local_solar_time": "lst_h",
        })

        # Replace fill values with NaN
        for col in ["density", "density_orbitmean", "altitude_m", "lat", "lon", "lst_h"]:
            if col in df.columns:
                df.loc[np.isclose(df[col].values.astype(float), FILL, rtol=1e-3), col] = np.nan

        # Keep only valid data
        if "validity_flag" in df.columns:
            df = df[df["validity_flag"] == 0].copy()

        # Add datetime column
        df.index.name = "datetime"
        df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        return df

    except Exception as e:
        print(f"    WARNING: {date.date()} failed: {e}")
        return None


def download_event_satellite(
    req: SwarmRequest,
    event: dict,
    sat: dict,
) -> None:
    """Download and save all daily data for one event and satellite."""

    raw_dir = Path(event["out_dir"]) / sat["subdir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(event["start"], event["end"], freq="D")
    all_dfs = []
    n_ok = 0

    print(f"\n  [{sat['label']}]  {event['label']}  ({event['start']} → {event['end']})")
    print(f"  Fetching {len(dates)} days from VirES ...")

    for date in dates:
        # Check if daily cache file exists
        cache_file = raw_dir / f"{sat['out_prefix']}_{date.strftime('%Y%m%d')}.parquet"
        if cache_file.exists():
            df_day = pd.read_parquet(cache_file)
        else:
            df_day = fetch_one_day(req, sat["collection"], date)
            if df_day is not None and not df_day.empty:
                df_day.to_parquet(cache_file, index=False)

        if df_day is not None and not df_day.empty:
            all_dfs.append(df_day)
            n_ok += 1
            print(f"    {date.strftime('%Y-%m-%d')}  rows={len(df_day):,}", end="\r")

    print(f"  Done: {n_ok}/{len(dates)} days with data")

    if not all_dfs:
        print(f"  ERROR: No data fetched for {sat['label']} {event['label']}")
        return

    # Combine and save integrated parquet
    combined = pd.concat(all_dfs, ignore_index=True).sort_values("datetime")
    combined = combined.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    int_dir = Path(event["int_dir"])
    int_dir.mkdir(parents=True, exist_ok=True)

    # Derive year tag from event label
    year = event["start"][:4] if event["start"][:4] != "2023" else "2024"
    year = "2024" if "2024" in event["label"] else "2026"
    out_parquet = int_dir / f"{sat['out_prefix']}_{year}_SSW.parquet"
    combined.to_parquet(out_parquet, index=False)

    print(f"  Saved: {out_parquet}  ({len(combined):,} rows)")


def main():
    print("=== SWARM DNS POD Download: 2024 & 2026 SSW ===")
    print(f"Server: {SERVER}\n")

    req = SwarmRequest(SERVER)

    for event in EVENTS:
        print(f"\n{'='*60}")
        print(f"EVENT: {event['label']}")
        print(f"{'='*60}")
        for sat in SATELLITES:
            download_event_satellite(req, event, sat)

    print("\n\nAll downloads complete.")


if __name__ == "__main__":
    main()
