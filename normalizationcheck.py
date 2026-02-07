import pandas as pd
import numpy as np

df = pd.read_parquet("normalizeddata/swarm_dnscpod_2018_normalized.parquet")

# 相関（ざっくり確認）
corr_before = df["density"].corr(df["alt_km"])
corr_after  = df["density_norm"].corr(df["alt_km"])

print("corr(density, alt_km)      =", corr_before)
print("corr(density_norm, alt_km) =", corr_after)
print("corr(density, F107)      =", df["density"].corr(df["F107"]))
print("corr(density_norm, F107) =", df["density_norm"].corr(df["F107"]))
print("corr(density, AP_AVG)      =", df["density"].corr(df["AP_AVG"]))
print("corr(density_norm, AP_AVG) =", df["density_norm"].corr(df["AP_AVG"]))

df["date"] = pd.to_datetime(df["datetime"], utc=True).dt.floor("D")

daily = df.groupby("date")[["density", "density_norm"]].mean()

import matplotlib.pyplot as plt

print(daily.head())
plt.figure(figsize=(10,4))
sc = plt.scatter(
    df["datetime"],
    df["density_norm"],
    c=df["lst_h"],
    s=1,
    cmap="twilight"
)
plt.colorbar(sc, label="local time [h]")
plt.tight_layout()
plt.show()
