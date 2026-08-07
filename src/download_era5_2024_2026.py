"""
download_era5_2024_2026.py

Purpose:
    Download ERA5 temperature at 10 hPa over 60°N–90°N from Copernicus Climate Data Store (CDS).
    Saves NetCDF files into data/SSW2024/ERA5/ and data/SSW2026/ERA5/.

Dataset: reanalysis-era5-pressure-levels
Variable: temperature
Pressure Level: 10 hPa
Area: [90, -180, 60, 180] (North, West, South, East)
"""

from __future__ import annotations
from pathlib import Path
import cdsapi

EVENTS = [
    dict(
        label    = "2024 NH SSW",
        out_dir  = Path("data/SSW2024/ERA5"),
        years    = ["2023", "2024"],
        months   = ["12", "01", "02", "03"],
        out_name = "era5_t10hpa_2024.nc",
    ),
    dict(
        label    = "2026 NH SSW",
        out_dir  = Path("data/SSW2026/ERA5"),
        years    = ["2025", "2026"],
        months   = ["12", "01", "02", "03"],
        out_name = "era5_t10hpa_2026.nc",
    ),
]

DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(0, 24)]


def download_event(c: cdsapi.Client, event: dict) -> None:
    out_dir = event["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / event["out_name"]

    if out_file.exists():
        print(f"[{event['label']}] Already exists: {out_file}")
        return

    print(f"\n[{event['label']}] Requesting ERA5 data from CDS ...")
    print(f"  Years: {event['years']}, Months: {event['months']}")
    print(f"  Saving to: {out_file}")

    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["temperature"],
            "pressure_level": ["10"],
            "year": event["years"],
            "month": event["months"],
            "day": DAYS,
            "time": TIMES,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [90, -180, 60, 180],  # N, W, S, E
        },
        str(out_file),
    )
    print(f"  ✅ Download complete: {out_file}")


def main():
    print("=== ERA5 10 hPa Temperature Download (CDS API) ===\n")
    try:
        c = cdsapi.Client()
    except Exception as e:
        print(f"ERROR: Could not initialize CDS API Client.\n{e}")
        return

    for event in EVENTS:
        download_event(c, event)

    print("\nAll downloads finished.")


if __name__ == "__main__":
    main()
