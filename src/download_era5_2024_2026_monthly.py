"""
download_era5_2024_2026_monthly.py

Purpose:
    Download ERA5 temperature at 10 hPa (60°N–90°N) month-by-month for faster CDS processing.
    Existing load_era5_temp() will automatically aggregate all *.nc files in data/SSW2024/ERA5/ and data/SSW2026/ERA5/.

Dataset: reanalysis-era5-pressure-levels
Variable: temperature
Pressure Level: 10 hPa
Area: [90, -180, 60, 180] (North, West, South, East)
"""

from __future__ import annotations
from pathlib import Path
import cdsapi

# Month-by-month tasks
TASKS = [
    # 2024 SSW (2023-12 to 2024-03)
    dict(year="2023", month="12", out_dir=Path("data/SSW2024/ERA5")),
    dict(year="2024", month="01", out_dir=Path("data/SSW2024/ERA5")),
    dict(year="2024", month="02", out_dir=Path("data/SSW2024/ERA5")),
    dict(year="2024", month="03", out_dir=Path("data/SSW2024/ERA5")),
    # 2026 SSW (2025-12 to 2026-03)
    dict(year="2025", month="12", out_dir=Path("data/SSW2026/ERA5")),
    dict(year="2026", month="01", out_dir=Path("data/SSW2026/ERA5")),
    dict(year="2026", month="02", out_dir=Path("data/SSW2026/ERA5")),
    dict(year="2026", month="03", out_dir=Path("data/SSW2026/ERA5")),
]

DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(0, 24)]


def download_month(c: cdsapi.Client, task: dict) -> None:
    out_dir = task["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"era5_t10hpa_{task['year']}{task['month']}.nc"

    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    print(f"\nRequesting {task['year']}-{task['month']} ...")
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["temperature"],
            "pressure_level": ["10"],
            "year": [task["year"]],
            "month": [task["month"]],
            "day": DAYS,
            "time": TIMES,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [90, -180, 60, 180],
        },
        str(out_file),
    )
    print(f"  ✅ Complete: {out_file}")


def main():
    print("=== ERA5 10 hPa Temperature Monthly Download ===\n")
    c = cdsapi.Client()

    for task in TASKS:
        download_month(c, task)

    print("\nAll monthly downloads finished.")


if __name__ == "__main__":
    main()
