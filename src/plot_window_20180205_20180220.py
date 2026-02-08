import pandas as pd
import numpy as np

PARQUET = "normalizeddata/swarm_dnscpod_2018_normalized.parquet"
t0 = pd.Timestamp("2018-02-05 00:00:00", tz="UTC")
t1 = pd.Timestamp("2018-02-21 00:00:00", tz="UTC")

df = pd.read_parquet(PARQUET).copy()
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

w = df[(df["datetime"]>=t0) & (df["datetime"]<t1)].copy()

# ---- 1) lst_h 自体の中身を“境界付近”で見る ----
# 15-19に入る行を抽出して、lst_h のユニーク値（丸め）を見る
x = w[(w["lst_h"]>=15) & (w["lst_h"]<19)]["lst_h"].to_numpy()
print("15-19 count:", len(x))
if len(x) > 0:
    print("15-19 min/max:", float(x.min()), float(x.max()))
    rounded = pd.Series(np.round(x, 3)).value_counts().head(40)
    print("\nTop lst_h values (rounded to 0.001) within 15-19:")
    print(rounded)

# 16-18が本当に0か確認
y = w[(w["lst_h"]>=16) & (w["lst_h"]<18)]["lst_h"].to_numpy()
print("\n16-18 count:", len(y))

# 16に“近い値”が存在するか
z = w[(w["lst_h"]>=15.8) & (w["lst_h"]<16.2)]["lst_h"].to_numpy()
print("15.8-16.2 count:", len(z))
if len(z)>0:
    print("sample near 16:", np.sort(z)[:30])

# ---- 2) そもそも lst_h が hour として妥当か ----
# 0-24に収まってるか、負の値が無いか
print("\nSanity check:")
print("lst_h overall min/max:", float(w['lst_h'].min()), float(w['lst_h'].max()))
print("any lst_h < 0?", bool((w["lst_h"] < 0).any()))
print("any lst_h >= 24?", bool((w["lst_h"] >= 24).any()))
