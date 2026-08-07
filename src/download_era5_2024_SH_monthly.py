"""
download_era5_2024_SH_monthly.py

Purpose:
    Download ERA5 10 hPa temperature NetCDF files for the
    2024 Southern Hemisphere SSW period (June–September 2024).
    Southern Hemisphere polar region: 60°S–90°S.

Source:
    Copernicus CDS (https://cds.climate.copernicus.eu)
    Dataset: reanalysis-era5-pressure-levels

Output:
    data/SSW2024_SH/ERA5/era5_T10hPa_2024_{MM}.nc
"""

from __future__ import annotations
import cdsapi
from pathlib import Path

OUT_DIR = Path("data/SSW2024_SH/ERA5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Months needed (with 1-month margin on each side)
MONTHS = [
    ("2024", "05"),  # margin
    ("2024", "06"),  # ref period 1 starts Jun 15
    ("2024", "07"),  # SSW peak 1: Jul 7
    ("2024", "08"),  # SSW peak 2: Aug 5
    ("2024", "09"),  # margin
]

def download_month(year: str, month: str) -> None:
    out_file = OUT_DIR / f"era5_T10hPa_{year}_{month}.nc"
    if out_file.exists():
        print(f"  [SKIP] {out_file.name} already exists")
        return

    print(f"  Downloading {year}-{month} ...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable":     "temperature",
            "pressure_level": ["10"],
            "year":   year,
            "month":  month,
            "day":    [f"{d:02d}" for d in range(1, 32)],
            "time":   ["00:00", "06:00", "12:00", "18:00"],
            "area":   [-60, -180, -90, 180],  # [N, W, S, E] → 60°S to 90°S
            "format": "netcdf",
        },
        str(out_file),
    )
    print(f"  Saved: {out_file}")


def main() -> None:
    print("=== ERA5 10 hPa T Download: 2024 SH SSW ===")
    print(f"Output dir: {OUT_DIR}\n")
    for year, month in MONTHS:
        download_month(year, month)
    print("\nAll ERA5 months done.")


if __name__ == "__main__":
    main()
