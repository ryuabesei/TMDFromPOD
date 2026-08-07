import numpy as np
import pandas as pd
from pymsis import msis
import matplotlib.pyplot as plt


def compute_msis_ref_density(df: pd.DataFrame) -> np.ndarray:
    """
    観測の lat, lon, time(index) を使い、
    ref条件（alt=450km, F10.7=70, Ap=4）でMSIS密度を計算
    """

    # =========================
    # 必須列チェック
    # =========================
    required_cols = ["lat", "lon"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"{c} 列が必要です")

    # =========================
    # numpy配列へ変換
    # =========================
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()

    # ✅ 時間は index を使う
    time_utc = pd.to_datetime(df.index, utc=True).to_numpy()

    N = len(df)

    # =========================
    # ref条件
    # =========================
    alt_km = np.full(N, 450.0)

    f107s  = np.full(N, 70.0)
    f107as = np.full(N, 70.0)

    aps = np.full((N, 7), 4.0)

    # =========================
    # MSIS実行
    # =========================
    out = msis.run(
        time_utc,
        lon,
        lat,
        alt_km,
        f107s=f107s,
        f107as=f107as,
        aps=aps,
        version=2.1
    )

    out = np.asarray(out)

    # =========================
    # 密度取り出し
    # =========================
    rho = out[:, 0] if out.ndim >= 2 else out

    # =========================
    # 安全チェック
    # =========================
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("MSISが不正な値を返しました")

    return rho


def prepare_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    density_norm を作る（無ければ）
    """

    if "density_norm" not in df.columns:
        if "density_orbitmean" in df.columns:
            df["density_norm"] = df["density"] / df["density_orbitmean"]
        else:
            print("⚠️ density_normなし → densityをそのまま使用")
            df["density_norm"] = df["density"]

    return df


def plot_density_vs_doy(df: pd.DataFrame):
    """
    DOY vs Density プロット
    """

    # =========================
    # DOY計算（indexから）
    # =========================
    dt = pd.to_datetime(df.index, utc=True)

    doy = dt.dayofyear.values
    hour = dt.hour.values
    minute = dt.minute.values

    doy_frac = doy + (hour + minute / 60.0) / 24.0

    # =========================
    # データ
    # =========================
    rho_norm = df["density_norm"].values
    rho_msis = df["rho_msis_ref"].values

    # =========================
    # プロット
    # =========================
    plt.figure(figsize=(10, 5))

    plt.scatter(doy_frac, rho_norm, s=5, alpha=0.5, label="SWARM (normalized)")
    plt.scatter(doy_frac, rho_msis, s=5, alpha=0.5, label="MSIS (ref)")

    plt.xlabel("Day of Year (DOY)")
    plt.ylabel("Density [kg/m³]")
    plt.title("Density vs DOY (SWARM vs MSIS)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# 実行部分
# =========================
df = pd.read_parquet("integrateddata/2018/swarm_dnsapod_2018.parquet")

# density_norm準備
df = prepare_density(df)

# MSIS計算
df["rho_msis_ref"] = compute_msis_ref_density(df)

# 比（超重要）
df["ratio"] = df["density"] / df["rho_msis_ref"]

# プロット
plot_density_vs_doy(df)