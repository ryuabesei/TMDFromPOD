# =========================
# Swarm DNSPOD データの
# 軌道平均密度（density_orbitmean）を
# 時系列で可視化する簡単なプロット
# =========================

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1. parquet ファイルの読み込み
# -------------------------
# integrateddata/swarm_dnscpod_2018.parquet は、
# すでに CDF から読み込んで前処理した Swarm の密度データ
# index は UTC の DatetimeIndex になっている想定
df = pd.read_parquet("integrateddata/swarm_dnsbpod_2018.parquet")


# -------------------------
# 2. figure 
# -------------------------
plt.figure(figsize=(12, 4))


# -------------------------
# 3. 密度の時系列プロット
# -------------------------
# x軸：観測時刻（df.index）
# y軸：軌道平均密度 density_orbitmean
# lw=0.8 で線を細めにして、点が多くても潰れにくくする
plt.plot(df.index, df["density_orbitmean"], lw=0.8)


# -------------------------
# 4. y軸を対数スケールに変更
# -------------------------
# 熱圏密度は数桁オーダーで変動するため、
# logスケールの方が変動構造を見やすい
plt.yscale("log")


# -------------------------
# 5. グリッド線を追加
# -------------------------
# alpha=0.3 で薄く表示し、データの邪魔をしないようにする
plt.grid(alpha=0.3)


# -------------------------
# 6. 軸ラベルの設定
# -------------------------
# x軸：UTC時刻
# y軸：密度（単位はDNSCPOD仕様に依存、ここでは kg/m^3 表記）
plt.xlabel("UTC")
plt.ylabel("Density [kg/m³]")


# -------------------------
# 7. 図のタイトル
# -------------------------
# 「LT や緯度でのフィルタを一切していない」
# 生データに近い軌道平均密度の時系列であることを明示
plt.title("Swarm-B density (orbit-mean, no LT / no latitude filtering)")


# -------------------------
# 8. レイアウト調整
# -------------------------
# ラベルやタイトルが図からはみ出さないように自動調整
plt.tight_layout()


# -------------------------
# 9. 図を画面に表示
# -------------------------
plt.show()
