import pandas as pd

df = pd.read_parquet("normalizeddata/swarm_dnscpod_2018_normalized.parquet")

print("datetime dtype:", df["datetime"].dtype)
print("N rows:", len(df))
print("N duplicated datetime:", df["datetime"].duplicated().sum())

# 重複が多い時刻トップ10
dup_counts = df["datetime"].value_counts()
print(dup_counts.head(10))
