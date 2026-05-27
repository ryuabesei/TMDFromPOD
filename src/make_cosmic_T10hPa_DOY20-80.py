"""
make_cosmic_T10hPa_DOY20-80.py

目的:
    COSMIC atmPrf NetCDF から 10 hPa の温度を抽出し、
    DOY 20-80、緯度 60-90N の日平均 T(10hPa) [K] を CSV で保存する。

方法:
    - 時刻・緯度・QC は グローバル属性 (ncattrs) から読む
    - Pres / Temp から log(P) 補間で 10 hPa の温度 [℃] を抽出
    - ℃ → K 変換して日平均
    - bad=1 のプロファイルは除外

出力:
    cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv
    (columns: datetime, T10_K, DOY)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# =========================
# 設定
# =========================
COSMIC_ROOT         = Path("data/COSMIC-1_atmPrf_Data")
COSMIC_DIR_TEMPLATE = "atmPrf_repro2021_2018_{doy:03d}"

DOY_START = 20
DOY_END   = 80

P_TARGET_MB    = 10.0        # 10 hPa
LAT_RANGE      = (60.0, 90.0)
EXCLUDE_BAD_QC = True

OUT_CSV = Path("cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv")


# =========================
# ユーティリティ
# =========================
def list_cosmic_files(doy_dir: Path) -> list[Path]:
    """拡張子なし _nc と .nc の両方に対応"""
    files = list(doy_dir.rglob("*_nc")) + list(doy_dir.rglob("*.nc"))
    return sorted({f.resolve() for f in files})


def interp_temp_at_pressure_logp(p_mb: np.ndarray, t_c: np.ndarray,
                                  p_target_mb: float) -> float | None:
    """log(P) 空間で線形補間して p_target_mb [hPa] の温度 [℃] を返す"""
    p = np.asarray(p_mb, dtype=float)
    t = np.asarray(t_c, dtype=float)

    m = (p > 0) & np.isfinite(p) & np.isfinite(t) & (p != -999) & (t != -999)
    p, t = p[m], t[m]
    if p.size < 2:
        return None

    idx = np.argsort(p)
    p, t = p[idx], t[idx]

    if not (p.min() <= p_target_mb <= p.max()):
        return None

    return float(np.interp(np.log(p_target_mb), np.log(p), t))


def read_profile_time_lat_qc(nc_path: Path) -> tuple[pd.Timestamp | None, float | None, int]:
    """グローバル属性から時刻・緯度・QC(bad) を取得"""
    with Dataset(nc_path, "r") as ds:
        bad = int(ds.getncattr("bad")) if "bad" in ds.ncattrs() else 0

        need = ["year", "month", "day", "hour", "minute", "second"]
        if not all(k in ds.ncattrs() for k in need):
            return None, None, bad

        y   = int(ds.getncattr("year"))
        mo  = int(ds.getncattr("month"))
        d   = int(ds.getncattr("day"))
        h   = int(ds.getncattr("hour"))
        mi  = int(ds.getncattr("minute"))
        sec = float(ds.getncattr("second"))

        sec_int = int(np.floor(sec))
        usec    = int(round((sec - sec_int) * 1e6))

        ts = pd.Timestamp(
            year=y, month=mo, day=d, hour=h, minute=mi,
            second=sec_int, microsecond=usec, tz="UTC"
        )

        if "lat" in ds.ncattrs():
            lat = float(ds.getncattr("lat"))
        elif "Lat" in ds.variables:
            arr = ds.variables["Lat"][:]
            lat = float(arr[0]) if np.size(arr) else None
        else:
            lat = None

        return ts, lat, bad


def read_T10_from_nc(nc_path: Path, p_target_mb: float) -> float | None:
    """Pres / Temp から p_target_mb の温度 [℃] を抽出"""
    with Dataset(nc_path, "r") as ds:
        if ("Pres" not in ds.variables) or ("Temp" not in ds.variables):
            return None
        pres = ds.variables["Pres"][:]
        temp = ds.variables["Temp"][:]
    return interp_temp_at_pressure_logp(pres, temp, p_target_mb)


# =========================
# メイン処理
# =========================
rows: list[tuple[pd.Timestamp, float, float]] = []
bad0_total = bad1_total = 0

for doy in range(DOY_START, DOY_END + 1):
    doy_dir = COSMIC_ROOT / COSMIC_DIR_TEMPLATE.format(doy=doy)
    if not doy_dir.exists():
        print(f"  DOY {doy:03d}: directory not found, skip")
        continue

    files = list_cosmic_files(doy_dir)
    n_used = 0

    for fp in files:
        try:
            ts, lat, bad = read_profile_time_lat_qc(fp)
        except Exception as e:
            continue

        if bad == 0:
            bad0_total += 1
        elif bad == 1:
            bad1_total += 1

        if EXCLUDE_BAD_QC and bad == 1:
            continue
        if ts is None or lat is None:
            continue

        lat_min, lat_max = LAT_RANGE
        if not (lat_min <= lat <= lat_max):
            continue

        try:
            t10_c = read_T10_from_nc(fp, P_TARGET_MB)
        except Exception:
            continue

        if t10_c is None:
            continue

        rows.append((ts, lat, t10_c))
        n_used += 1

    print(f"  DOY {doy:03d}: {len(files)} files, {n_used} profiles used")

print(f"\nQC: bad=0 → {bad0_total}, bad=1 → {bad1_total}")
print(f"Total profiles used: {len(rows)}")

if len(rows) == 0:
    raise RuntimeError("No profiles read. Check COSMIC data directory and filters.")

# DataFrame 化 → 日平均
df_cos = (
    pd.DataFrame(rows, columns=["datetime", "lat", "T10_C"])
    .set_index("datetime")
    .sort_index()
)

# ℃ → K に変換して日平均
daily_T10_K = (df_cos["T10_C"] + 273.15).resample("D").mean()

out = pd.DataFrame({"T10_K": daily_T10_K})
out["DOY"] = out.index.dayofyear

out.to_csv(OUT_CSV, index=True)
print(f"\n✅ Saved: {OUT_CSV}")
print(out.to_string())
