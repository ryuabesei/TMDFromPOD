# ============================================
# Swarm latitudinal density profile (Afternoon sector)
# 2018 SSW representative days
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# 設定
# -------------------------

FILES = {
    "SWARM-A": "normalizeddata/swarm_dnsapod_2018_normalized.parquet",
    "SWARM-B": "normalizeddata/swarm_dnsbpod_2018_normalized.parquet",
    "SWARM-C": "normalizeddata/swarm_dnscpod_2018_normalized.parquet",
}

REP_DATES = ["2018-02-08", "2018-02-11", "2018-02-13", "2018-02-15"]

LAT_RANGE = (-60, 60)
LAT_BIN = 2.0
LT_WINDOW = (7, 9)   # Afternoon sector

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------
# 緯度プロファイル作成関数
# -------------------------

def make_lat_profile(df_day):
    df_day = df_day[(df_day["lat"] >= LAT_RANGE[0]) &
                    (df_day["lat"] <= LAT_RANGE[1])]

    edges = np.arange(LAT_RANGE[0], LAT_RANGE[1] + LAT_BIN, LAT_BIN)
    centers = (edges[:-1] + edges[1:]) / 2

    df_day["lat_bin"] = pd.cut(
        df_day["lat"],
        bins=edges,
        labels=centers,
        include_lowest=True
    )

    prof = df_day.groupby("lat_bin")["density_norm"].median().reset_index()
    prof.columns = ["lat_center", "density_norm"]
    prof["lat_center"] = prof["lat_center"].astype(float)

    return prof.sort_values("lat_center")

# -------------------------
# メイン処理
# -------------------------

for sat, fp in FILES.items():

    print(f"\nProcessing {sat}")

    df = pd.read_parquet(fp)

    # datetime を index に
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    # LTフィルタ（Afternoon）
    df = df[(df["lst_h"] >= LT_WINDOW[0]) &
            (df["lst_h"] < LT_WINDOW[1])]

    fig, ax = plt.subplots(figsize=(8, 5))

    for d in REP_DATES:

        d0 = pd.Timestamp(d, tz="UTC")
        d1 = d0 + pd.Timedelta(days=1)

        df_day = df[(df.index >= d0) & (df.index < d1)]

        if len(df_day) == 0:
            print(f"No data for {sat} {d}")
            continue

        prof = make_lat_profile(df_day)

        ax.plot(prof["lat_center"],
                prof["density_norm"],
                lw=2,
                label=d)

    ax.set_title(f"{sat} Afternoon (16–18 LT)")
    ax.set_xlabel("Geographic Latitude [deg]")
    ax.set_ylabel("Normalized Density [kg m$^{-3}$]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_yscale("log")

    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{sat}_lat_profile_afternoon.png"
    plt.savefig(out_path, dpi=300)
    print("Saved:", out_path)

    plt.show()
