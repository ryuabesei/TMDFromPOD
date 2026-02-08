import pandas as pd
import numpy as np

PARQUET = "integrateddata/swarm_dnscpod_2018.parquet"

# =========================
# 1) 読み込み＆基本情報
# =========================
df = pd.read_parquet(PARQUET)

print("=== basic ===")
print("shape:", df.shape)
print("columns:", list(df.columns))
print("index type:", type(df.index))
print("index min/max:", df.index.min(), "->", df.index.max())
print("is_monotonic_increasing:", df.index.is_monotonic_increasing)
print()

# 期待する列が揃っているか（必要ならここで追加）
expected_cols = ["density", "density_orbitmean", "validity_flag", "altitude_m", "lat", "lon", "lst_h", "source_file"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print("Missing columns:", missing)
print()

# =========================
# 2) 期間が正しいか（端点チェック）
# =========================
expected_start = pd.Timestamp("2018-02-01 00:00:00")
expected_end   = pd.Timestamp("2018-03-10 23:59:30")

print("=== range check ===")
print("expected:", expected_start, "->", expected_end)
print("actual  :", df.index.min(), "->", df.index.max())

if df.index.min() != expected_start:
    print("⚠ start mismatch")
if df.index.max() != expected_end:
    print("⚠ end mismatch")
print()

# =========================
# 3) indexの重複が残っていないか
# =========================
print("=== duplicate index check ===")
dup_count = df.index.duplicated().sum()
print("duplicated index count:", dup_count)

# もし重複があったら、どの時刻が重複しているか少し見る
if dup_count > 0:
    dup_times = df.index[df.index.duplicated(keep=False)]
    print("examples of duplicated times:", dup_times[:10].tolist())
print()

# =========================
# 4) サンプリング間隔（30秒）とギャップ検出
# =========================
print("=== cadence & gap check ===")
dt = df.index.to_series().diff().dropna()

# 代表的な差分（最頻値）を見る
mode_dt = dt.mode().iloc[0]
print("mode dt:", mode_dt)

# 30秒以外の割合を見る（多少の例外があってもOKだが、多いと統合ミスの疑い）
non30 = (dt != pd.Timedelta(seconds=30)).mean()
print("fraction of dt != 30s:", non30)

# 大きいギャップ（例：5分以上）を列挙
gaps = dt[dt >= pd.Timedelta(minutes=5)]
print("num gaps >= 5min:", len(gaps))
if len(gaps) > 0:
    print("gap examples:")
    print(gaps.head(10))
print()

# =========================
# 5) 日ごとの点数チェック（1日=2880点が理想）
# =========================
print("=== daily count check ===")
daily_counts = df.resample("D").size()

print("days:", len(daily_counts))
print("min/max daily count:", daily_counts.min(), daily_counts.max())

# 2880 から外れている日（欠け・重複除去・欠損ファイルを疑う）
bad_days = daily_counts[daily_counts != 2880]
print("num days != 2880:", len(bad_days))
if len(bad_days) > 0:
    print("days with abnormal counts:")
    print(bad_days)
print()

# 期待日数（2018/02/01〜03/10 は 38日）もチェック
expected_days = (expected_end.normalize() - expected_start.normalize()).days + 1
print("expected number of days:", expected_days)
print("actual   number of days:", len(daily_counts))
if len(daily_counts) != expected_days:
    print("⚠ day count mismatch (some days missing or extra)")
print()

# =========================
# 6) NaN と FILL の混入チェック
# =========================
print("=== NaN / FILL check ===")
nan_rate = df[["density", "density_orbitmean", "altitude_m", "lat", "lon", "lst_h"]].isna().mean()
print("NaN rate per column:")
print(nan_rate)

# FILL値が残っていないか（あなたのFILL=0.999e33）
FILL = 0.99900e33
fill_left = {}
for col in ["density", "density_orbitmean", "altitude_m", "lat", "lon", "lst_h"]:
    fill_left[col] = int(np.isclose(df[col].to_numpy(), FILL, equal_nan=False).sum())
print("FILL count per column:", fill_left)
print()

# =========================
# 7) validity_flag の確認（統合時に 0 に絞っているはず）
# =========================
print("=== validity_flag check ===")
vc = df["validity_flag"].value_counts(dropna=False)
print(vc)
if len(vc) != 1 or vc.index[0] != 0:
    print("⚠ validity_flag contains non-zero values (filter may not have worked)")
print()

# =========================
# 8) source_file の整合性（ざっくり）
# =========================
print("=== source_file sanity ===")
if "source_file" in df.columns:
    # 何ファイル分あるか
    print("num unique source_file:", df["source_file"].nunique())
    print("example source_file:", df["source_file"].dropna().unique()[:5])

    # 1日が複数source_fileに跨ってないか（跨るのが普通ならOKだが、異常も検出できる）
    daily_sources = df.groupby(pd.Grouper(freq="D"))["source_file"].nunique()
    many = daily_sources[daily_sources > 1]
    print("days with >1 source_file:", len(many))
    if len(many) > 0:
        print(many.head(10))
print()

print("✅ done")
