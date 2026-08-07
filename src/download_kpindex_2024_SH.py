"""
download_kpindex_2024_SH.py

Purpose:
    Download Kp/Ap/F10.7 index CSV for the 2024 SH SSW analysis period.
    (2024-04-01 to 2024-09-30, with extra margin for 81-day F10.7 average)

Source:
    GFZ Potsdam: https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt

Output:
    data/SSW2024_SH/Kpindex/SW-20240601_20240831.csv
"""

from __future__ import annotations
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
GFZ_CACHE = Path("data/SSW2024_SH/Kpindex/gfz_kp_full.txt")

EVENT = dict(
    label   = "2024 SH SSW",
    start   = "2024-03-01",   # extra margin for 81-day F10.7
    end     = "2024-09-30",
    out_csv = Path("data/SSW2024_SH/Kpindex/SW-20240601_20240831.csv"),
)


def download_gfz(url: str, dest: Path) -> None:
    print(f"Downloading GFZ index from:\n  {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Cached to: {dest}")


def parse_gfz(path: Path) -> pd.DataFrame:
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
            year, month, day = int(r[0]), int(r[1]), int(r[2])
            row = {
                "DATE":              f"{year:04d}-{month:02d}-{day:02d}",
                "KP1":               float(r[7]),  "KP2": float(r[8]),
                "KP3":               float(r[9]),  "KP4": float(r[10]),
                "KP5":               float(r[11]), "KP6": float(r[12]),
                "KP7":               float(r[13]), "KP8": float(r[14]),
                "AP1":               float(r[15]), "AP2": float(r[16]),
                "AP3":               float(r[17]), "AP4": float(r[18]),
                "AP5":               float(r[19]), "AP6": float(r[20]),
                "AP7":               float(r[21]), "AP8": float(r[22]),
                "AP_AVG":            float(r[23]),
                "F10.7_OBS":         float(r[25]) if len(r) > 25 else np.nan,
                "F10.7_ADJ":         float(r[26]) if len(r) > 26 else np.nan,
                "F10.7_ADJ_CENTER81":np.nan,   # filled below
            }
            processed.append(row)
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(processed)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    # Compute 81-day centered rolling mean for F10.7_ADJ
    df["F10.7_ADJ_CENTER81"] = (
        df["F10.7_ADJ"]
        .rolling(window=81, min_periods=40, center=True)
        .mean()
    )
    return df


def save_event(df_full: pd.DataFrame, event: dict) -> None:
    t0 = pd.Timestamp(event["start"])
    t1 = pd.Timestamp(event["end"])
    df_ev = df_full[(df_full["DATE"] >= t0) & (df_full["DATE"] <= t1)].copy()
    df_ev["DATE"] = df_ev["DATE"].dt.strftime("%Y-%m-%d")
    event["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    df_ev.to_csv(event["out_csv"], index=False)
    print(f"  Saved: {event['out_csv']}  ({len(df_ev)} rows)")


def main() -> None:
    print("=== Kp/F10.7 Download: 2024 SH SSW ===\n")

    if not GFZ_CACHE.exists():
        download_gfz(URL, GFZ_CACHE)
    else:
        print(f"  Using cached GFZ file: {GFZ_CACHE}")

    df_full = parse_gfz(GFZ_CACHE)
    save_event(df_full, EVENT)
    print("\nDone.")


if __name__ == "__main__":
    main()
