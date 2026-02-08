import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 入力（あなたのパス）=====
# PARQUET = "integrateddata/swarm_dnscpod_2018.parquet"
# 正規化後のファイルで alt_km が入っているなら、こっちでもOK:
PARQUET = "normalizeddata/swarm_dnscpod_2018_normalized.parquet"

df = pd.read_parquet(PARQUET).copy()

# ===== datetime を確実に作る =====
if "datetime" in df.columns:
    t = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
elif isinstance(df.index, pd.DatetimeIndex):
    t = pd.to_datetime(df.index, utc=True, errors="coerce")
else:
    raise KeyError("datetime列 or DatetimeIndex が見つかりません")

if t.isna().any():
    raise ValueError("datetime の変換に失敗しました（NaNあり）")

df["datetime"] = t

# ===== 高度列を km に統一 =====
if "alt_km" in df.columns:
    df["alt_km"] = df["alt_km"].astype(float)
elif "altitude_km" in df.columns:
    df["alt_km"] = df["altitude_km"].astype(float)
elif "altitude_m" in df.columns:
    df["alt_km"] = df["altitude_m"].astype(float) / 1000.0
else:
    raise KeyError("高度列が見つかりません（alt_km / altitude_km / altitude_m）")

# 緯度（あるなら使う）
lat_col = None
for c in ["lat", "latitude"]:
    if c in df.columns:
        lat_col = c
        break

# ====== 1) 高度の時系列（全点）======
plt.figure(figsize=(18,4))
plt.plot(df["datetime"], df["alt_km"], lw=0.6)
plt.grid(alpha=0.3)
plt.xlabel("UTC")
plt.ylabel("Altitude [km]")
plt.title("Swarm-C altitude time series (all points)")
plt.tight_layout()
plt.show()

# ====== 2) 高度分布（ヒストグラム + 分位点）======
alt = df["alt_km"].to_numpy()

p1, p5, p25, p50, p75, p95, p99 = np.percentile(alt, [1,5,25,50,75,95,99])
mean = float(np.mean(alt))
std  = float(np.std(alt))

print("Altitude summary [km]")
print(f"mean={mean:.3f}, std={std:.3f}")
print(f"p01={p1:.3f}, p05={p5:.3f}, p25={p25:.3f}, p50={p50:.3f}, p75={p75:.3f}, p95={p95:.3f}, p99={p99:.3f}")

plt.figure(figsize=(8,4))
plt.hist(df["alt_km"], bins=60)
plt.axvline(p50, linewidth=1.5, label=f"median={p50:.2f} km")
plt.axvline(mean, linewidth=1.5, label=f"mean={mean:.2f} km")
plt.axvline(p25, linewidth=1.0, linestyle="--", label=f"p25={p25:.2f}")
plt.axvline(p75, linewidth=1.0, linestyle="--", label=f"p75={p75:.2f}")
plt.axvline(p5,  linewidth=1.0, linestyle=":",  label=f"p05={p5:.2f}")
plt.axvline(p95, linewidth=1.0, linestyle=":",  label=f"p95={p95:.2f}")
plt.grid(alpha=0.3)
plt.xlabel("Altitude [km]")
plt.ylabel("Count")
plt.title("Swarm-C altitude distribution")
plt.legend()
plt.tight_layout()
plt.show()

# ====== 3) DoY × 緯度 の高度マップ（緯度がある場合）======
if lat_col is not None:
    # DoY（小数日）
    dt = df["datetime"]
    doy = dt.dt.dayofyear + (dt.dt.hour + dt.dt.minute/60 + dt.dt.second/3600)/24.0
    df["doy"] = doy

    # ビン設定
    doy_bin = 0.25   # 6時間
    lat_bin = 2.0    # 2度

    doy_edges = np.arange(np.floor(df["doy"].min()/doy_bin)*doy_bin,
                          np.ceil(df["doy"].max()/doy_bin)*doy_bin + doy_bin,
                          doy_bin)
    lat_edges = np.arange(-60, 60+lat_bin, lat_bin)

    # ビン化
    di = np.digitize(df["doy"].to_numpy(), doy_edges) - 1
    li = np.digitize(df[lat_col].to_numpy(), lat_edges) - 1

    ok = (di>=0)&(di<len(doy_edges)-1)&(li>=0)&(li<len(lat_edges)-1)
    tmp = pd.DataFrame({
        "di": di[ok],
        "li": li[ok],
        "alt": df.loc[ok, "alt_km"].to_numpy(float)
    })

    g = tmp.groupby(["li","di"], as_index=False)["alt"].mean()

    Z = np.full((len(lat_edges)-1, len(doy_edges)-1), np.nan)
    Z[g["li"].to_numpy(int), g["di"].to_numpy(int)] = g["alt"].to_numpy(float)

    doy_cent = (doy_edges[:-1] + doy_edges[1:]) / 2
    lat_cent = (lat_edges[:-1] + lat_edges[1:]) / 2
    X, Y = np.meshgrid(doy_cent, lat_cent)

    plt.figure(figsize=(12,4))
    m = plt.pcolormesh(doy_edges, lat_edges, Z, shading="auto")
    plt.colorbar(m, label="Mean altitude [km]")
    plt.xlabel("DoY (2018)")
    plt.ylabel("GLAT (deg)")
    plt.title("Swarm-C altitude map (mean altitude in DoY×lat bins)")
    plt.tight_layout()
    plt.show()
else:
    print("lat列が見つからないので、DoY×緯度の高度マップはスキップしました。")
