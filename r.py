import pandas as pd
import numpy as np

df = pd.read_parquet("normalizeddata/swarm_dnscpod_2018_normalized.parquet")
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

# 期間抽出
df = df[(df["datetime"] >= "2018-02-05") & (df["datetime"] <= "2018-02-20 23:59:59")].copy()

print("columns contain lat/lst_h?", "lat" in df.columns, "lst_h" in df.columns)
print("N total (in period):", len(df))

# lst_hの統計
print("\n=== lst_h describe ===")
print(df["lst_h"].describe(percentiles=[0.01,0.05,0.1,0.5,0.9,0.95,0.99]))

# 16-18 が本当に0か確認
n_16_18 = ((df["lst_h"] >= 16) & (df["lst_h"] < 18)).sum()
print("\nN in 16–18 LT:", n_16_18)

# どのLT帯に多いか（ざっくりヒスト）
bins = np.arange(0, 24.1, 1.0)
counts, edges = np.histogram(df["lst_h"].to_numpy(), bins=bins)
print("\n=== hourly counts (0-24) ===")
for i, c in enumerate(counts):
    if c > 0:
        print(f"{int(edges[i]):02d}-{int(edges[i+1]):02d}:", c)
