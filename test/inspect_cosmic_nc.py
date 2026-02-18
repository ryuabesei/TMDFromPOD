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


# COSMIC_Daily_T10hPa_and_DensityPlot.py
# 目的:
# - COSMIC atmPrf（展開済み *_nc / *.nc）から「10 hPa (=10 mb) の成層圏温度」を抽出
# - （任意）緯度帯フィルタ（例: 60–90N）
# - 1日平均 T(10hPa) を作成してCSV保存
# - すでに作れている Swarm 正規化密度の日平均と同じDOY軸で重ね描き（右軸をピンク線）
#
# 使い方:
# - 下の「入力パラメータ」をあなたの環境に合わせて変更して、そのまま実行
# - まずは DOY_START=32, DOY_END=50 で動作確認 → 問題なければ 69 まで伸ばす

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# =========================
# 入力パラメータ（ここだけ編集すればOK）
# =========================
# COSMIC 展開済みデータのルート
COSMIC_ROOT = Path("data/COSMIC-1_atmPrf_Data")

# COSMIC フォルダ命名（あなたの例に合わせて）
# 例: data/COSMIC-1_atmPrf_Data/atmPrf_repro2021_2018_032/...
COSMIC_DIR_TEMPLATE = "atmPrf_repro2021_2018_{doy:03d}"

# 読み込む DOY 範囲（例: 032〜050）
DOY_START = 32
DOY_END = 52

# 10 hPa (=10 mb)
P_TARGET_MB = 10.0

# 緯度フィルタ（None にすると無効）
LAT_RANGE = (60.0, 90.0)  # 例: 60–90N
# LAT_RANGE = None        # ← フィルタなしにしたい場合

# QC: bad=1 を除外するか
EXCLUDE_BAD_QC = True

# Swarm 正規化密度 parquet
SWARM_PARQUET = Path("normalizeddata/swarm_dnsapod_2018_normalized.parquet")

# 表示したい期間（UTC）
START_DATE = "2018-02-05"
END_DATE = "2018-02-21"

# 出力（COSMIC 日平均）
OUT_CSV = Path("cosmic_T10hPa_daily_2018_DOY032_050_lat60_90N.csv")


# =========================
# ユーティリティ
# =========================
def list_cosmic_files(doy_dir: Path) -> list[Path]:
    """
    COSMICファイルは末尾が "_nc"（拡張子なし）だったり ".nc" だったりするので両対応
    """
    files = list(doy_dir.rglob("*_nc")) + list(doy_dir.rglob("*.nc"))
    # 重複排除しつつソート
    return sorted({f.resolve() for f in files})


def interp_temp_at_pressure_logp(p_mb: np.ndarray, t_c: np.ndarray, p_target_mb: float) -> float | None:
    """
    Pres[mb] と Temp[℃] の鉛直プロファイルから、p_target の温度を log(P) 空間で線形補間して返す
    """
    p = np.asarray(p_mb, dtype=float)
    t = np.asarray(t_c, dtype=float)

    # 欠損値除去（-999 など）
    m = (p > 0) & np.isfinite(p) & np.isfinite(t) & (p != -999) & (t != -999)
    p, t = p[m], t[m]
    if p.size < 2:
        return None

    # 圧力でソート
    idx = np.argsort(p)
    p, t = p[idx], t[idx]

    # ターゲットが範囲外なら None
    if not (p.min() <= p_target_mb <= p.max()):
        return None

    # logP補間（標準的）
    return float(np.interp(np.log(p_target_mb), np.log(p), t))


def read_profile_time_lat_qc(nc_path: Path) -> tuple[pd.Timestamp | None, float | None, int | None]:
    """
    このCOSMIC atmPrfは、year/month/.../bad が「GLOBAL ATTRIBUTES」(ncattrs) に入っている。
    ここから観測時刻・緯度・QC(bad)を取得する。
    """
    with Dataset(nc_path, "r") as ds:
        # QC (bad) は属性
        bad = int(ds.getncattr("bad")) if "bad" in ds.ncattrs() else 0

        # 時刻は属性 year/month/day/hour/minute/second
        need = ["year", "month", "day", "hour", "minute", "second"]
        if not all(k in ds.ncattrs() for k in need):
            return None, None, bad

        y = int(ds.getncattr("year"))
        mo = int(ds.getncattr("month"))
        d = int(ds.getncattr("day"))
        h = int(ds.getncattr("hour"))
        mi = int(ds.getncattr("minute"))
        sec = float(ds.getncattr("second"))

        sec_int = int(np.floor(sec))
        usec = int(round((sec - sec_int) * 1e6))

        ts = pd.Timestamp(
            year=y, month=mo, day=d, hour=h, minute=mi, second=sec_int, microsecond=usec, tz="UTC"
        )

        # 緯度は属性 lat を優先（なければ変数 Lat の先頭）
        if "lat" in ds.ncattrs():
            lat = float(ds.getncattr("lat"))
        elif "Lat" in ds.variables:
            arr = ds.variables["Lat"][:]
            lat = float(arr[0]) if np.size(arr) else None
        else:
            lat = None

        return ts, lat, bad


def read_T10_from_nc(nc_path: Path, p_target_mb: float) -> float | None:
    """
    Pres / Temp から 10 hPa の温度(℃) を抽出
    """
    with Dataset(nc_path, "r") as ds:
        if ("Pres" not in ds.variables) or ("Temp" not in ds.variables):
            return None
        pres = ds.variables["Pres"][:]  # mb (=hPa)
        temp = ds.variables["Temp"][:]  # ℃
    return interp_temp_at_pressure_logp(pres, temp, p_target_mb)


# =========================
# COSMIC: 日平均T(10hPa)を作る
# =========================
rows: list[tuple[pd.Timestamp, float, float, int]] = []

bad0 = 0
bad1 = 0

for doy in range(DOY_START, DOY_END + 1):
    doy_dir = COSMIC_ROOT / COSMIC_DIR_TEMPLATE.format(doy=doy)
    if not doy_dir.exists():
        print(f"{doy:03d} skip (dir not found): {doy_dir}")
        continue

    files = list_cosmic_files(doy_dir)
    print(f"{doy:03d} files: {len(files)}")

    for fp in files:
        ts, lat, bad = read_profile_time_lat_qc(fp)

        if bad is not None:
            if bad == 0:
                bad0 += 1
            elif bad == 1:
                bad1 += 1

        # QC除外
        if EXCLUDE_BAD_QC and (bad == 1):
            continue

        if ts is None or lat is None:
            continue

        # 緯度フィルタ
        if LAT_RANGE is not None:
            lat_min, lat_max = LAT_RANGE
            if not (lat_min <= lat <= lat_max):
                continue

        t10_c = read_T10_from_nc(fp, P_TARGET_MB)
        if t10_c is None:
            continue

        rows.append((ts, lat, t10_c, bad if bad is not None else -1))

print("QC count (seen): bad=0", bad0, "bad=1", bad1)
print("profiles used (after filters):", len(rows))

if len(rows) == 0:
    raise RuntimeError(
        "No profiles were read after filters.\n"
        "確認ポイント:\n"
        "- EXCLUDE_BAD_QC=True で弾きすぎてないか\n"
        "- LAT_RANGE が厳しすぎないか（例: 60–90N）\n"
        "- Pres/Temp が欠けているファイルが多くないか\n"
    )

df_cos = pd.DataFrame(rows, columns=["datetime", "lat", "T10_C", "bad"]).set_index("datetime").sort_index()

# 日平均（℃→K）
daily_T10_K = (df_cos["T10_C"] + 273.15).resample("D").mean()
out = pd.DataFrame({"T10_K": daily_T10_K})
out["DOY"] = out.index.dayofyear
out.to_csv(OUT_CSV, index=True)
print("Saved COSMIC daily T10hPa:", OUT_CSV)

# =========================
# Swarm: 既存コード相当（密度の日平均）
# =========================
df_sw = pd.read_parquet(SWARM_PARQUET)
df_sw["datetime"] = pd.to_datetime(df_sw["datetime"], utc=True)
df_sw = df_sw.set_index("datetime").sort_index()

df_sw_period = df_sw.loc[START_DATE:END_DATE]
daily_density = df_sw_period["density_norm"].resample("D").mean()

# =========================
# 同じ期間で COSMIC T10 を切り出し
# =========================
t10_period = out.loc[START_DATE:END_DATE, "T10_K"]

# =========================
# プロット（左:密度 log、右:T10 ピンク）
# =========================
fig, ax1 = plt.subplots(figsize=(8, 4))

# x軸（DOY）
x_doy = daily_density.index.dayofyear

ax1.plot(x_doy, daily_density.values, marker="o", markersize=4, linewidth=1.5)
ax1.set_yscale("log")
ax1.grid(alpha=0.3)
ax1.set_xlabel("Day of Year (DOY)")
ax1.set_ylabel("Daily mean normalized density [kg/m³]")
ax1.set_title("Swarm-A Density + COSMIC T(10 hPa)")

ax2 = ax1.twinx()
ax2.plot(t10_period.index.dayofyear, t10_period.values, linewidth=1.8, color="hotpink")
ax2.set_ylabel("T (10 hPa) [K]")

plt.tight_layout()
plt.show()
