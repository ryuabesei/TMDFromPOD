"""
download_kpindex_2024_2026.py

Purpose:
    Generate Kp/Ap/F10.7 index CSV files for 2024 and 2026 SSW analysis periods.
    Uses the same format as existing data/SSW2021/Kpindex/SW-20201220_20210228.csv.

Source:
    GFZ Potsdam: https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt

Output:
    data/SSW2024/Kpindex/SW-20231201_20240331.csv
    data/SSW2026/Kpindex/SW-20251201_20260430.csv
"""

from __future__ import annotations
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"

EVENTS = [
    dict(
        label    = "2024 NH SSW",
        start    = "2023-11-01",   # extra margin for 81-day F10.7 average
        end      = "2024-03-31",
        out_csv  = Path("data/SSW2024/Kpindex/SW-20231201_20240331.csv"),
    ),
    dict(
        label    = "2026 NH SSW",
        start    = "2025-10-01",   # extra margin
        end      = "2026-04-30",
        out_csv  = Path("data/SSW2026/Kpindex/SW-20251201_20260430.csv"),
    ),
]


def download_gfz(url: str, dest: Path) -> None:
    print(f"Downloading GFZ index file from:\n  {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to: {dest}")


def parse_gfz(path: Path) -> pd.DataFrame:
    """Parse GFZ Kp/Ap/F10.7 text file into DataFrame."""
    data_rows = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 26:
                data_rows.append(parts)

    print(f"  Parsed {len(data_rows):,} daily rows")

    processed = []
    for r in data_rows:
        try:
            year  = int(r[0])
            month = int(r[1])
            day   = int(r[2])
            row = {
                "DATE":     f"{year:04d}-{month:02d}-{day:02d}",
                "KP1":      float(r[7]),
                "KP2":      float(r[8]),
                "KP3":      float(r[9]),
                "KP4":      float(r[10]),
                "KP5":      float(r[11]),
                "KP6":      float(r[12]),
                "KP7":      float(r[13]),
                "KP8":      float(r[14]),
                "AP1":      float(r[15]),
                "AP2":      float(r[16]),
                "AP3":      float(r[17]),
                "AP4":      float(r[18]),
                "AP5":      float(r[19]),
                "AP6":      float(r[20]),
                "AP7":      float(r[21]),
                "AP8":      float(r[22]),
                "AP_AVG":   float(r[23]),
                "F10.7_OBS": float(r[25]),
            }
            processed.append(row)
        except Exception:
            continue

    df = pd.DataFrame(processed)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    df = df.replace(-1.0, np.nan)
    df["F10.7_OBS"] = df["F10.7_OBS"].ffill()
    df["F10.7_ADJ"] = df["F10.7_OBS"]

    # 81-day moving averages
    df["F10.7_ADJ_CENTER81"] = df["F10.7_ADJ"].rolling(window=81, center=True,  min_periods=1).mean()
    df["F10.7_OBS_CENTER81"] = df["F10.7_OBS"].rolling(window=81, center=True,  min_periods=1).mean()
    df["F10.7_ADJ_LAST81"]   = df["F10.7_ADJ"].rolling(window=81, center=False, min_periods=1).mean()
    df["F10.7_OBS_LAST81"]   = df["F10.7_OBS"].rolling(window=81, center=False, min_periods=1).mean()

    return df


def make_event_csv(df_full: pd.DataFrame, event: dict) -> None:
    out_csv = event["out_csv"]
    start, end = event["start"], event["end"]

    df_slice = df_full.loc[start:end].copy().reset_index()
    df_slice["DATE"] = df_slice["DATE"].dt.strftime("%Y-%m-%d")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_slice.to_csv(out_csv, index=False)
    print(f"  ✅ Saved {event['label']}: {out_csv}  ({len(df_slice)} rows)")


def main():
    print("=== Kp/Ap/F10.7 Index Preparation for 2024 & 2026 SSW ===\n")

    # Download once
    temp_file = Path("data/SSW2024/Kpindex/gfz_kp_temp.txt")
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    download_gfz(URL, temp_file)

    # Parse
    print("\nParsing data ...")
    df_full = parse_gfz(temp_file)
    print(f"  Date range: {df_full.index.min()} → {df_full.index.max()}")

    # Slice and save for each event
    print("\nGenerating event CSVs ...")
    for event in EVENTS:
        make_event_csv(df_full, event)

    # Cleanup temp
    temp_file.unlink()
    print("\nAll done.")


if __name__ == "__main__":
    main()
