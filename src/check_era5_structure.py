"""
check_era5_structure.py
Check the variable names, dimensions, and attributes of the 2021 ERA5 NetCDF files.
"""
from pathlib import Path
import xarray as xr

ERA5_DIR = Path("data/SSW2021/ERA5")

for fp in ERA5_DIR.glob("*.nc"):
    print(f"\n======================================")
    print(f"File: {fp.name}")
    print(f"======================================")
    try:
        ds = xr.open_dataset(fp)
        print(ds)
        ds.close()
    except Exception as e:
        print(f"Error opening file: {e}")
