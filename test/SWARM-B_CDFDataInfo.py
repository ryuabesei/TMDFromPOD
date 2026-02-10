from cdflib import CDF
from cdflib import cdfepoch
import numpy as np

path = "data/SWARM_B/SW_OPER_DNSBPOD_2__20180201T000000_20180201T235930_0301.cdf"
cdf = CDF(path)

info = cdf.cdf_info()

print("zVariables:", info.zVariables)
print("rVariables:", info.rVariables)
print("Attributes:", info.Attributes)

zvars = cdf.cdf_info().zVariables

print("=== zVariables の基本情報 ===")
for v in zvars:
    data = cdf.varget(v)

    print(f"\n--- {v} ---")
    print("type:", type(data))
    print("dtype:", data.dtype if hasattr(data, "dtype") else "N/A")
    print("shape:", np.shape(data))

    # 数値データの場合は代表値を見る
    if np.issubdtype(data.dtype, np.number):
        print("min:", np.nanmin(data))
        print("max:", np.nanmax(data))
        print("mean:", np.nanmean(data))

    # 先頭5個だけ表示（巨大配列対策）
    print("first 5 values:", data[:5])


time = cdf.varget("time")

# 人間が読める UTC に変換
time_utc = cdfepoch.to_datetime(time)

print("time shape:", time.shape)
print("start time:", time_utc[0])
print("end time  :", time_utc[-1])
rho = cdf.varget("density")

print("density shape:", rho.shape)
print("min [kg/m3]:", rho.min())
print("max [kg/m3]:", rho.max())
print("example:", rho[:5])

rho_orb = cdf.varget("density_orbitmean")

print("orbit-mean density min:", rho_orb.min())
print("orbit-mean density max:", rho_orb.max())

alt = cdf.varget("altitude")

print("altitude shape:", alt.shape)
print("min altitude:", alt.min())
print("max altitude:", alt.max())
print("example:", alt[:5])

lat = cdf.varget("latitude")
lon = cdf.varget("longitude")

print("latitude range:", lat.min(), lat.max())
print("longitude range:", lon.min(), lon.max())

lst = cdf.varget("local_solar_time")

print("LST range:", lst.min(), lst.max())
print("example:", lst[:10])

flag = cdf.varget("validity_flag")

unique, counts = np.unique(flag, return_counts=True)
print("validity_flag counts:")
for u, c in zip(unique, counts):
    print(u, c)
