import pandas as pd
import numpy as np

NORM = "normalizeddata/swarm_dnsbpod_2018_normalized.parquet"
RAW  = "integrateddata/swarm_dnsbpod_2018.parquet"

df = pd.read_parquet(NORM)
raw = pd.read_parquet(RAW)

print("=== normalized basic ===")
print("shape:", df.shape)
print("columns head:", list(df.columns)[:20])
print()

# -------------------------
# 1) 必須列チェック
# -------------------------
required = [
    "datetime", "density", "density_orbitmean", "alt_km", "lat", "lon",
    "F107", "F107A", "AP_AVG",
    "density_norm", "norm_ratio_model_ref_over_real",
    "norm_ref_alt_km", "norm_ref_F107", "norm_ref_AP"
]
missing = [c for c in required if c not in df.columns]
print("=== required columns ===")
print("missing:", missing)
print()

# -------------------------
# 2) 範囲・サンプリング（期間と30秒）
# -------------------------
print("=== time / cadence ===")
t = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
print("t min/max:", t.min(), "->", t.max())
dt = t.sort_values().diff().dropna()
print("mode dt:", dt.mode().iloc[0])
print("fraction dt != 30s:", (dt != pd.Timedelta(seconds=30)).mean())
print("gaps >= 5min:", (dt >= pd.Timedelta(minutes=5)).sum())
print()

# -------------------------
# 3) 基準条件が一定か
# -------------------------
print("=== reference conditions constant? ===")
for c in ["norm_ref_alt_km", "norm_ref_F107", "norm_ref_AP"]:
    u = df[c].unique()
    print(c, "unique:", u[:10], "n_unique:", len(u))
print()

# -------------------------
# 4) 欠損・inf・負値チェック
# -------------------------
print("=== NaN / inf / sign check ===")
check_cols = ["density", "alt_km", "F107", "AP_AVG", "density_norm", "norm_ratio_model_ref_over_real"]
for c in check_cols:
    x = pd.to_numeric(df[c], errors="coerce")
    print(f"{c:30s} NaN={x.isna().mean():.6f}  inf={np.isinf(x).mean():.6f}  <=0={(x<=0).mean():.6f}")
print()

# -------------------------
# 5) ratio の物理的妥当性（極端値がないか）
# -------------------------
print("=== ratio stats ===")
ratio = df["norm_ratio_model_ref_over_real"].astype(float)
print(ratio.describe(percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
print()

# -------------------------
# 6) 正規化の効果：相関が減っているか（超重要）
#    raw density vs alt/F107/AP と、 density_norm vs alt/F107/AP を比較
# -------------------------
print("=== correlation comparison (raw vs normalized) ===")

# rawにも alt_km を作る（integratedは altitude_m のはず）
raw2 = raw.copy()
raw2["alt_km"] = raw2["altitude_m"].astype(float) / 1000.0

# normalized 側と時間を合わせて結合（innerで共通時刻だけ）
t_norm = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
df2 = df.copy()
df2["time"] = t_norm
raw2 = raw2.copy()
raw2["time"] = raw2.index

m = pd.merge(raw2[["time","density","alt_km"]], df2[["time","density_norm","F107","AP_AVG","alt_km"]], on="time", how="inner", suffixes=("_raw","_norm"))
print("merged rows:", len(m))

def corr(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a.corr(b)

print("raw density  vs alt_km :", corr(m["density"], m["alt_km_raw"]))
print("norm density vs alt_km :", corr(m["density_norm"], m["alt_km_norm"]))

print("norm density vs F107   :", corr(m["density_norm"], m["F107"]))
print("norm density vs AP_AVG :", corr(m["density_norm"], m["AP_AVG"]))
print()

# 期待：正規化後の相関（絶対値）が、正規化前より明確に小さい（特に高度）
# -------------------------
# 7) “基準高度に換算できている感”の確認
#    高度をバケットに分けて中央値を見る（正規化後は高度依存が弱まるはず）
# -------------------------
print("=== altitude-binned medians ===")
m["alt_bin"] = pd.cut(m["alt_km_raw"], bins=[430, 440, 450, 460, 470, 480], include_lowest=True)
g = m.groupby("alt_bin")[["density", "density_norm"]].median()
print(g)
print()

print("✅ done")
