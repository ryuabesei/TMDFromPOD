"""
plot_LT_coverage_2018.py

Purpose:
    Visualize the Local Time (LT) sector coverage of SWARM-A, B, C
    during DOY 30-65, 2018, using a clock-style polar plot.

Output:
    Figure/2018/LT_coverage_SWARM-ABC_2018.png
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUT_PNG = Path("Figure/2018/LT_coverage_SWARM-ABC_2018.png")

# ── LT sectors ────────────────────────────────────────────────────────────────
# (label, lt_start, lt_end, color, satellite, linestyle_ring)
SECTORS = [
    ("SWARM-B\nNightside",  0,  6, "#4a90d9", "SWARM-B", 0),   # 深夜
    ("SWARM-A\nMorning",    6, 12, "#1f77b4", "SWARM-A", 1),   # 朝
    ("SWARM-C\nMorning",    6, 12, "#17becf", "SWARM-C", 2),   # 朝（タンデム）
    ("SWARM-B\nDayside",   12, 18, "#e07b39", "SWARM-B", 0),   # 昼
    ("SWARM-A\nEvening",   18, 24, "#d62728", "SWARM-A", 1),   # 夕
    ("SWARM-C\nEvening",   18, 24, "#ff7f0e", "SWARM-C", 2),   # 夕（タンデム）
]

# Ring radii for each satellite (inner, outer)
RING_PARAMS = {
    "SWARM-B": {"r_inner": 0.55, "r_outer": 0.75, "lw": 0},
    "SWARM-A": {"r_inner": 0.78, "r_outer": 0.93, "lw": 0},
    "SWARM-C": {"r_inner": 0.96, "r_outer": 1.10, "lw": 0},
}

SAT_COLORS = {
    "SWARM-A": "#1f77b4",
    "SWARM-B": "#4a90d9",
    "SWARM-C": "#17becf",
}

SAT_ALT = {
    "SWARM-A": "~460 km",
    "SWARM-B": "~530 km",
    "SWARM-C": "~460 km",
}


def lt_to_rad(lt: float) -> float:
    """LT [0,24] → angle [rad], 0h at top, clockwise."""
    return (lt / 24.0) * 2 * np.pi - np.pi / 2


def draw_arc_sector(ax, r_inner, r_outer, lt_start, lt_end,
                    color, alpha=0.80, n=300):
    """Draw a filled arc sector between two radii."""
    theta1 = lt_to_rad(lt_start)
    theta2 = lt_to_rad(lt_end)
    thetas = np.linspace(theta1, theta2, n)

    xs = (np.concatenate([r_inner * np.cos(thetas),
                           r_outer * np.cos(thetas[::-1])]))
    ys = (np.concatenate([r_inner * np.sin(thetas),
                           r_outer * np.sin(thetas[::-1])]))
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=3)
    ax.plot(np.append(r_inner * np.cos(thetas), r_inner * np.cos(thetas[0])),
            np.append(r_inner * np.sin(thetas), r_inner * np.sin(thetas[0])),
            color="white", lw=0.6, zorder=4)
    ax.plot(np.append(r_outer * np.cos(thetas), r_outer * np.cos(thetas[0])),
            np.append(r_outer * np.sin(thetas), r_outer * np.sin(thetas[0])),
            color="white", lw=0.6, zorder=4)


def draw_ring_outline(ax, r_inner, r_outer, color, label, n=500):
    """Draw thin ring outline for a satellite."""
    thetas = np.linspace(0, 2 * np.pi, n)
    for r in [r_inner, r_outer]:
        ax.plot(r * np.cos(thetas), r * np.sin(thetas),
                color=color, lw=1.0, ls="--", alpha=0.4, zorder=2)


def main():
    fig = plt.figure(figsize=(10, 9), facecolor="#0d1117")
    ax = fig.add_subplot(111, aspect="equal", facecolor="#0d1117")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")

    # ── Background circles ─────────────────────────────────────────────────
    for r, alpha in [(0.50, 0.12), (0.75, 0.08), (0.93, 0.06), (1.10, 0.05)]:
        circle = plt.Circle((0, 0), r, color="white", fill=False,
                             lw=0.5, alpha=alpha, zorder=1)
        ax.add_patch(circle)

    # Day/Night background
    thetas = np.linspace(0, 2 * np.pi, 500)
    # Dayside (6h–18h): right half
    th_day = np.linspace(lt_to_rad(6), lt_to_rad(18), 300)
    xs = np.concatenate([[0], 1.30 * np.cos(th_day), [0]])
    ys = np.concatenate([[0], 1.30 * np.sin(th_day), [0]])
    ax.fill(xs, ys, color="#fffde7", alpha=0.04, zorder=0)

    # ── Draw sectors ───────────────────────────────────────────────────────
    for label, lt_s, lt_e, color, sat, _ in SECTORS:
        rp = RING_PARAMS[sat]
        draw_arc_sector(ax, rp["r_inner"], rp["r_outer"],
                        lt_s, lt_e, color, alpha=0.82)

    # ── Ring outlines ──────────────────────────────────────────────────────
    for sat, rp in RING_PARAMS.items():
        draw_ring_outline(ax, rp["r_inner"], rp["r_outer"],
                          SAT_COLORS[sat], sat)

    # ── LT hour ticks & labels ─────────────────────────────────────────────
    for lt in range(0, 24, 3):
        angle = lt_to_rad(lt)
        r_tick_in  = 0.48
        r_tick_out = 1.14
        # tick line
        ax.plot([r_tick_in  * np.cos(angle), r_tick_out * np.cos(angle)],
                [r_tick_in  * np.sin(angle), r_tick_out * np.sin(angle)],
                color="white", lw=0.7, alpha=0.30, zorder=1, ls=":")
        # LT label
        r_label = 1.22
        lx = r_label * np.cos(angle)
        ly = r_label * np.sin(angle)
        ax.text(lx, ly, f"{lt:02d}h",
                ha="center", va="center", fontsize=9,
                color="white", alpha=0.75,
                fontfamily="monospace", fontweight="bold")

    # ── Center label ───────────────────────────────────────────────────────
    ax.text(0, 0.10, "LOCAL", ha="center", va="center",
            fontsize=11, color="white", alpha=0.5, fontweight="bold")
    ax.text(0, -0.08, "TIME", ha="center", va="center",
            fontsize=11, color="white", alpha=0.5, fontweight="bold")

    # Sun symbol (right side = noon)
    sun_angle = lt_to_rad(12)
    sx, sy = 1.38 * np.cos(sun_angle), 1.38 * np.sin(sun_angle)
    ax.text(sx, sy, "☀", ha="center", va="center", fontsize=18,
            color="#FFD700", zorder=5)

    # Moon symbol (left side = midnight)
    moon_angle = lt_to_rad(0)
    mx, my = 1.38 * np.cos(moon_angle), 1.38 * np.sin(moon_angle)
    ax.text(mx, my, "🌙", ha="center", va="center", fontsize=16, zorder=5)

    # ── Satellite ring labels (right side) ─────────────────────────────────
    label_x = 1.42
    ring_label_info = [
        ("SWARM-B", RING_PARAMS["SWARM-B"], "#4a90d9",
         "Nightside / Dayside\n(LT 0–6h & 12–18h)"),
        ("SWARM-A", RING_PARAMS["SWARM-A"], "#1f77b4",
         "Morning / Evening\n(LT 6–12h & 18–24h)"),
        ("SWARM-C", RING_PARAMS["SWARM-C"], "#17becf",
         "Morning / Evening\n(LT 6–12h & 18–24h, +~1h offset)"),
    ]

    # ── Legend (bottom) ────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#4a90d9", label=f"SWARM-B  ({SAT_ALT['SWARM-B']})  Nightside & Dayside"),
        mpatches.Patch(color="#1f77b4", label=f"SWARM-A  ({SAT_ALT['SWARM-A']})  Morning & Evening"),
        mpatches.Patch(color="#17becf", label=f"SWARM-C  ({SAT_ALT['SWARM-C']})  Morning & Evening  (+~1h)"),
    ]
    leg = ax.legend(handles=legend_items,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.07),
                    fontsize=9.5, framealpha=0.15,
                    facecolor="#1c2333", edgecolor="#444",
                    labelcolor="white", ncol=1,
                    handlelength=1.5, handleheight=1.2)

    # ── Ring radius annotations ────────────────────────────────────────────
    ann_angle = lt_to_rad(9)   # 9h方向に注記
    for sat, rp in RING_PARAMS.items():
        r_mid = (rp["r_inner"] + rp["r_outer"]) / 2
        ax.text(r_mid * np.cos(ann_angle),
                r_mid * np.sin(ann_angle),
                sat.replace("SWARM-", "S-"),
                ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold",
                alpha=0.90, zorder=6)

    # ── Title ──────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97,
             "SWARM-A / B / C  Local Time Coverage",
             ha="center", va="top",
             fontsize=15, fontweight="bold", color="white")
    fig.text(0.5, 0.935,
             "DOY 30–65, 2018  |  Inner ring: SWARM-B  ·  Middle: SWARM-A  ·  Outer: SWARM-C",
             ha="center", va="top",
             fontsize=9, color="#aaaaaa")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {OUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
