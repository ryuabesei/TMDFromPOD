import pandas as pd
df = pd.read_parquet("integrateddata/swarm_dnscpod_2018.parquet")
print(df.shape, df.index.min(), df.index.max())
print(df.isna().mean().sort_values(ascending=False).head(10))
print(df["validity_flag"].value_counts(dropna=False))
df60 = df[df["lat"].between(-60, 60)]
print("N(all) =", len(df), "N(|lat|<=60) =", len(df60))

# LSTの分布
print(df60["lst_h"].describe())
