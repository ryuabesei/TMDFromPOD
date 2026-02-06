import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("integrateddata/swarm_dnscpod_2018.parquet")

plt.figure(figsize=(12,4))
plt.plot(df.index, df["density_orbitmean"], lw=0.8)
plt.yscale("log")
plt.grid(alpha=0.3)
plt.xlabel("UTC")
plt.ylabel("Density [kg/m³]")
plt.title("Swarm-C density (orbit-mean, no LT / no latitude filtering)")
plt.tight_layout()
plt.show()
