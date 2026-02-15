from netCDF4 import Dataset
from pathlib import Path

# 代表ファイル1つ（あなたのパスに合わせて）
fp = Path("data/COSMIC-1_atmPrf_Data/atmPrf_repro2021_2018_032") \
     / "atmPrf_C001.2018.032.02.15.G05_2021.0390_nc"

with Dataset(fp, "r") as ds:
    print("===== VARIABLES =====")
    for var in ds.variables:
        print(var)

    print("\n===== GLOBAL ATTRIBUTES =====")
    for attr in ds.ncattrs():
        print(attr, ":", getattr(ds, attr))
