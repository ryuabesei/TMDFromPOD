"""
plot_ref_medians_comparison_2019.py

Purpose:
    Create a clean, presentation-ready bar chart of the 2019 non-SSW reference medians
    for SWARM-A, B, and C (with updated LT sectors).
    This chart can be directly pasted into slides.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 計算された2019年基準中央値データ
data = {
    "SWARM-A": {
        "Dawn (LT 2.5-8.5h)": [0.453, 0.436, 0.444],
        "Dusk (LT 14.5-20.5h)": [0.502, 0.450, 0.407]
    },
    "SWARM-C": {
        "Dawn (LT 2.5-8.5h)": [0.431, 0.426, 0.427],
        "Dusk (LT 14.5-20.5h)": [0.497, 0.451, 0.409]
    },
    "SWARM-B": {
        "Midnight (LT 0-4h)": [0.471, 0.344, 0.263],
        "Noon (LT 12-16h)": [0.489, 0.421, 0.362]
    }
}

categories = ["High\n(40-60°)", "Mid\n(20-40°)", "Low\n(0-20°)"]
x = np.arange(len(categories))
width = 0.35

# スライド用にクリアなスタイルに設定
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
fig.patch.set_facecolor('white')

# サテライトごとのカラー設定
colors = {
    "Dawn / Midnight": "#1f77b4",  # 青
    "Dusk / Noon": "#e07b39"       # オレンジ
}

for i, sat_name in enumerate(["SWARM-A", "SWARM-C", "SWARM-B"]):
    ax = axes[i]
    sat_data = data[sat_name]
    
    # キー名を取得（LTセクター名）
    lt_keys = list(sat_data.keys())
    lt1, lt2 = lt_keys[0], lt_keys[1]
    
    # 棒グラフの描画
    rects1 = ax.bar(x - width/2, sat_data[lt1], width, label=lt1, color=colors["Dawn / Midnight"], alpha=0.9, edgecolor='none')
    rects2 = ax.bar(x + width/2, sat_data[lt2], width, label=lt2, color=colors["Dusk / Noon"], alpha=0.9, edgecolor='none')
    
    # 数値ラベルを棒の上に追加
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    # 軸の装飾
    ax.set_title(sat_name, fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 0.6)  # 2019年は全体的に値が小さいため上限を0.6に調整
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.legend(loc='lower center', fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.2))

axes[0].set_ylabel("Reference Median of \u03c1_ratio", fontsize=14, fontweight='bold')

plt.suptitle("2019 non-SSW Reference Medians (08/20-08/26 & 09/20-09/23)", fontsize=18, fontweight='bold', y=1.02)

# 画像として保存
out_png = Path("Figure/2019/reference_medians_comparison.png")
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
plt.close()
