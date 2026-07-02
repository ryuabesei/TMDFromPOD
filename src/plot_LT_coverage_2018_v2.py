"""
plot_LT_coverage_2018_v2.py

Purpose:
    Visualize SWARM-A/B/C Local Time coverage (DOY 30-65, 2018).
    - Top:    Clock-style polar plot  (1 color per satellite)
    - Bottom: LT histogram from actual observations

Output:
    Figure/2018/LT_coverage_SWARM-ABC_2018_v2.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_PNG = Path("Figure/2018/LT_coverage_SWARM-ABC_2018_v2.png")

# ── Satellite settings (1 color per satellite) ────────────────────────────────
SATELLITES = {
    "SWARM-A": {
        "parquet": Path("normalizeddata/2018/swarm_dnsapod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        "color":   "#e74c3c",   # vivid red
        "lt_sectors": [(6, 12), (18, 24)],
        "label": "SWARM-A  Morning & Evening\n(LT 6–12h & 18–24h, ~460 km)",
        "ring": (0.78, 0.93),
    },
    "SWARM-B": {
        "parquet": Path("normalizeddata/2018/swarm_dnsbpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        "color":   "#2ecc71",   # vivid green
        "lt_sectors": [(0, 6), (12, 18)],
        "label": "SWARM-B  Nightside & Dayside\n(LT 0–6h & 12–18h, ~530 km)",
        "ring": (0.55, 0.75),
    },
    "SWARM-C": {
        "parquet": Path("normalizeddata/2018/swarm_dnscpod_2018_normalized_with_LT_removed_DOY20-80.parquet"),
        "color":   "#a855f7",   # vivid purple
        "lt_sectors": [(6, 12), (18, 24)],
        "label": "SWARM-C  Morning & Evening\n(LT 6–12h & 18–24h, +~1h, ~460 km)",
        "ring": (0.96, 1.10),
    },
}

DOY_START, DOY_END = 30, 65


def lt_to_rad(lt: float) -> float:
    """LT [0,24] → angle [rad]. 12h=top, clockwise.
    12h → +π/2 (top), 18h → 0 (right), 00h → -π/2 (bottom), 06h → π (left)
    """
    return -((lt - 12) / 24.0) * 2 * np.pi + np.pi / 2


def draw_arc(ax, r_in, r_out, lt_s, lt_e, color, alpha=0.82, n=300):
    """Filled arc sector on a Cartesian axis."""
    a1 = lt_to_rad(lt_s)
    a2 = lt_to_rad(lt_e)
    # make sure we go clockwise
    if a2 > a1:
        a2 -= 2 * np.pi
    thetas = np.linspace(a1, a2, n)
    xs = np.concatenate([r_in  * np.cos(thetas),
                         r_out * np.cos(thetas[::-1])])
    ys = np.concatenate([r_in  * np.sin(thetas),
                         r_out * np.sin(thetas[::-1])])
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=3)
    for r in [r_in, r_out]:
        ax.plot(r * np.cos(thetas), r * np.sin(thetas),
                color="white", lw=0.8, alpha=0.5, zorder=4)


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    lt_data: dict[str, np.ndarray] = {}
    for sat, cfg in SATELLITES.items():
        df = pd.read_parquet(cfg["parquet"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["DOY_int"] = df["datetime"].dt.dayofyear
        df = df[(df["DOY_int"] >= DOY_START) & (df["DOY_int"] <= DOY_END)]
        df = df.dropna(subset=["lst_h"])
        lt_data[sat] = df["lst_h"].values
        print(f"{sat}: {len(df):,} obs")

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 12), facecolor="#0d1117")
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.6, 1],
                            hspace=0.05, left=0.08, right=0.92,
                            top=0.94, bottom=0.07)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 1: Clock-style polar plot
    # ══════════════════════════════════════════════════════════════════════════
    ax_clock = fig.add_subplot(gs[0], aspect="equal", facecolor="#0d1117")
    ax_clock.set_xlim(-1.35, 1.35)
    ax_clock.set_ylim(-1.35, 1.35)
    ax_clock.axis("off")

    # Background guide circles
    thetas_full = np.linspace(0, 2 * np.pi, 500)
    for r, a in [(0.50, 0.10), (0.75, 0.08), (0.93, 0.07), (1.10, 0.06)]:
        ax_clock.plot(r * np.cos(thetas_full), r * np.sin(thetas_full),
                      color="white", lw=0.5, alpha=a, zorder=1)

    # LT tick lines and hour labels
    for lt in range(0, 24, 3):
        angle = lt_to_rad(lt)
        ax_clock.plot([0.48 * np.cos(angle), 1.13 * np.cos(angle)],
                      [0.48 * np.sin(angle), 1.13 * np.sin(angle)],
                      color="white", lw=0.5, alpha=0.20, ls=":", zorder=1)
        r_lbl = 1.23
        ax_clock.text(r_lbl * np.cos(angle), r_lbl * np.sin(angle),
                      f"{lt:02d}h",
                      ha="center", va="center", fontsize=9.5,
                      color="white", alpha=0.70, fontfamily="monospace",
                      fontweight="bold")

    # Sun / Night markers
    ax_clock.text(0, 1.33, "☀  Noon", ha="center", va="center",
                  fontsize=12, color="#FFD700", fontweight="bold")
    ax_clock.text(0, -1.33, "Midnight", ha="center", va="center",
                  fontsize=10, color="#aaaacc", alpha=0.7)

    # Draw sectors (1 color per satellite)
    for sat, cfg in SATELLITES.items():
        r_in, r_out = cfg["ring"]
        for (lt_s, lt_e) in cfg["lt_sectors"]:
            draw_arc(ax_clock, r_in, r_out, lt_s, lt_e,
                     cfg["color"], alpha=0.85)
        # Ring name label at 45° position
        ann_angle = lt_to_rad(9)
        r_mid = (r_in + r_out) / 2
        ax_clock.text(r_mid * np.cos(ann_angle),
                      r_mid * np.sin(ann_angle),
                      sat.replace("SWARM-", "S-"),
                      ha="center", va="center",
                      fontsize=7.5, color="white",
                      fontweight="bold", alpha=0.95, zorder=6)

    # Center label
    ax_clock.text(0, 0.08, "LOCAL", ha="center", va="center",
                  fontsize=12, color="white", alpha=0.40, fontweight="bold")
    ax_clock.text(0, -0.10, "TIME", ha="center", va="center",
                  fontsize=12, color="white", alpha=0.40, fontweight="bold")

    # Legend (clock panel) — placed inside the figure to the right
    patches = [mpatches.Patch(color=cfg["color"], label=cfg["label"])
               for cfg in SATELLITES.values()]
    ax_clock.legend(handles=patches, loc="lower right",
                    bbox_to_anchor=(1.35, 0.02), fontsize=9,
                    framealpha=0.20, facecolor="#1c2333",
                    edgecolor="#555", labelcolor="white",
                    ncol=1, handlelength=1.4, handleheight=1.1)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 2: LT histogram (actual observation distribution)
    # ══════════════════════════════════════════════════════════════════════════
    ax_hist = fig.add_subplot(gs[1], facecolor="#0d1117")

    bins = np.arange(0, 24.5, 0.5)   # 0.5h bins
    for sat, cfg in SATELLITES.items():
        lts = lt_data[sat]
        counts, edges = np.histogram(lts, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax_hist.fill_between(centers, counts, alpha=0.35, color=cfg["color"],
                             step="mid", zorder=2)
        ax_hist.step(centers, counts, color=cfg["color"], lw=1.8,
                     where="mid", label=sat, zorder=3)

    ax_hist.set_xlim(0, 24)
    ax_hist.set_xticks(range(0, 25, 3))
    ax_hist.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 3)],
                             fontsize=9, color="white")
    ax_hist.set_xlabel("Local Time (h)", fontsize=11, color="white", labelpad=6)
    ax_hist.set_ylabel("Observation count\n(0.5h bin)", fontsize=10,
                        color="white", labelpad=6)
    ax_hist.tick_params(colors="white", labelsize=9)
    for spine in ax_hist.spines.values():
        spine.set_edgecolor("#444")
    ax_hist.grid(axis="y", alpha=0.15, color="white", lw=0.6)
    ax_hist.grid(axis="x", alpha=0.10, color="white", lw=0.5)

    ax_hist.legend(fontsize=9.5, framealpha=0.20, facecolor="#1c2333",
                   edgecolor="#555", labelcolor="white",
                   loc="upper left", ncol=1)
    ax_hist.set_title("Observation LT distribution  (DOY 30–65, 2018)",
                      fontsize=10, color="#aaaaaa", pad=5, loc="right")

    # ── Suptitle ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97,
             "SWARM-A / B / C  —  Local Time Coverage",
             ha="center", va="top",
             fontsize=15, fontweight="bold", color="white")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {OUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
