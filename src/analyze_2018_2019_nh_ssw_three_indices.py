"""Analyze the 2018/19 NH SSW with daily Ap, 3-hour ap, and 1-hour ap60."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_2d_detrend_only_3years import compute_2d_grid, load_temp


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "standardizeddata" / "2018_2019_NH_SSW"
GFZ = ROOT / "data" / "geomagnetic_gfz_high_cadence"
OUT = ROOT / "reports" / "2018_2019_NH_SSW" / "three_index_linear_detrending"
EVENT = "2018_2019_NH"
START = pd.Timestamp("2018-12-10", tz="UTC")
END = pd.Timestamp("2019-02-10 23:59:59", tz="UTC")
CENTRAL = pd.Timestamp("2019-01-02", tz="UTC")
SSW_END = pd.Timestamp("2019-01-20 23:59:59", tz="UTC")
METHODS = {
    "daily_Ap": dict(label="daily Ap", cadence=24.0, window=24.0, lag=0.0),
    "ap_3hour": dict(label="3-hour ap", cadence=3.0, window=24.0, lag=0.0),
    "ap60_1hour": dict(label="1-hour ap60", cadence=1.0, window=24.0, lag=1.0),
}


def load_driver(tag: str) -> pd.Series:
    data = json.loads((GFZ / f"{EVENT}_{tag}_gfz.json").read_text())
    key = {"daily_Ap": "Ap", "ap_3hour": "ap", "ap60_1hour": "ap60"}[tag]
    return pd.Series(
        data[key], index=pd.to_datetime(data["datetime"], utc=True), dtype=float
    ).replace(-1, np.nan).sort_index()


def attach_driver(times: pd.Series, tag: str) -> np.ndarray:
    cfg = METHODS[tag]
    series = load_driver(tag)
    if tag == "daily_Ap":
        by_day = series.copy()
        by_day.index = by_day.index.normalize()
        return by_day.reindex(times.dt.normalize()).to_numpy()
    # GFZ timestamps mark interval starts. Only completed intervals are available.
    series.index += pd.to_timedelta(cfg["cadence"], unit="h")
    rolling = series.rolling(
        pd.Timedelta(hours=cfg["window"]), closed="right",
        min_periods=max(1, math.ceil(cfg["window"] / cfg["cadence"])),
    ).mean()
    left = pd.DataFrame({
        "cutoff": times - pd.to_timedelta(cfg["lag"], unit="h"),
        "order": np.arange(len(times)),
    }).sort_values("cutoff")
    right = pd.DataFrame({"available": rolling.index, "driver": rolling.values}).dropna()
    merged = pd.merge_asof(
        left, right.sort_values("available"), left_on="cutoff", right_on="available",
        direction="backward",
    )
    return merged.sort_values("order")["driver"].to_numpy()


def temperature() -> pd.Series:
    return load_temp(dict(
        temp_type="era5", temp_dir=ROOT / "data/SSW2018_2019_NH/ERA5",
        lat_range=(60.0, 90.0), date_start=START, date_end=END,
    ))


def qc_rows(df: pd.DataFrame, satellite: str) -> list[dict]:
    finite = np.isfinite(df["density"]) & np.isfinite(df["rho_model_real"]) & np.isfinite(df["density_ratio_msis"])
    physical = finite & (df["density"] > 0) & (df["rho_model_real"] > 0) & (df["density_ratio_msis"] >= 0)
    rows = []
    for level, low, high in (("QC0", None, None), ("QC1", .1, 3.), ("QC2", .01, 5.)):
        keep = physical.copy()
        if low is not None:
            keep &= df["density_ratio_msis"].between(low, high)
        rows.append(dict(
            event=EVENT, satellite=satellite, qc_level=level, n_input=len(df),
            n_retained=int(keep.sum()), n_excluded=int((~keep).sum()),
            ratio_min=low, ratio_max=high,
        ))
    return rows


def main() -> None:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    temp = temperature()
    metrics, qc = [], []

    for path in sorted(INPUT.glob("swarm_*_msis_normalized.parquet")):
        satellite = path.name.split("_")[1].upper()
        raw = pd.read_parquet(path)
        raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
        qc.extend(qc_rows(raw, satellite))
        use = (
            np.isfinite(raw["density_ratio_msis"]) & np.isfinite(raw["density"])
            & np.isfinite(raw["rho_model_real"]) & (raw["density"] > 0)
            & (raw["rho_model_real"] > 0) & (raw["density_ratio_msis"] >= 0)
        )
        raw = raw[use & raw["datetime"].between(START, END)].copy()
        raw["block"] = raw["datetime"].dt.floor("90min")
        blocks = raw.groupby("block").agg(
            datetime=("datetime", "median"),
            density_ratio_msis=("density_ratio_msis", "median"),
        ).reset_index(drop=True)

        for tag, cfg in METHODS.items():
            blocks["driver"] = attach_driver(blocks["datetime"], tag)
            fit = blocks.dropna(subset=["driver", "density_ratio_msis"])
            slope, intercept = np.polyfit(fit["driver"], fit["density_ratio_msis"], 1)
            corr = float(np.corrcoef(fit["driver"], fit["density_ratio_msis"])[0, 1])
            raw["driver"] = attach_driver(raw["datetime"], tag)
            raw["residual"] = raw["density_ratio_msis"] - (slope * raw["driver"] + intercept)
            raw["date"] = raw["datetime"].dt.normalize()
            daily = raw.groupby("date").agg(
                residual=("residual", "median"), driver=("driver", "mean"),
            )
            pre = daily[daily.index < CENTRAL]["residual"]
            during = daily[daily.index.to_series().between(CENTRAL, SSW_END)]["residual"]
            metrics.append(dict(
                event=EVENT, satellite=satellite, geomagnetic_index=cfg["label"],
                window_h=cfg["window"], lag_h=cfg["lag"], fit_unit="90-minute block median",
                slope=slope, intercept=intercept, correlation=corr,
                pre_days=len(pre), ssw_days=len(during),
                residual_pre_median=pre.median(), residual_ssw_median=during.median(),
                ssw_minus_pre_residual_pp=100 * (during.median() - pre.median()),
            ))

            edges = pd.date_range(START.normalize(), END.normalize() + pd.Timedelta(days=1), freq="D")
            lat_edges = np.arange(-60, 63, 3)
            z = compute_2d_grid(raw, "residual", edges, lat_edges)
            fig = plt.figure(figsize=(14, 8), layout="constrained")
            gs = fig.add_gridspec(2, 2, width_ratios=[1, .045], height_ratios=[1, 1.55])
            ax = fig.add_subplot(gs[0, 0]); spacer = fig.add_subplot(gs[0, 1]); spacer.axis("off")
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax); cax = fig.add_subplot(gs[1, 1])
            tax = ax.twinx(); gax = ax.twinx(); gax.spines["right"].set_position(("axes", 1.08))
            l1 = ax.plot(daily.index, daily["residual"], "o-", lw=2, ms=3.5, color="#1f77b4", label=f"ratio residual after {cfg['label']}")
            l2 = tax.plot(temp.index, temp.values, "^-", lw=2, ms=3.5, color="#dc143c", label="T(10 hPa) 60–90°N")
            if tag == "daily_Ap":
                detail = "same-UTC-day assignment"
            else:
                detail = f"{cfg['window']:g} h causal trailing; lag {cfg['lag']:g} h"
            l3 = gax.plot(daily.index, daily["driver"], "s-", lw=1.4, ms=3.5, color="#e67e22", label=f"{cfg['label']}: {detail}")
            ax.axhline(0, color=".4", ls=":"); ax.axvline(CENTRAL, color="red", ls="--")
            ax.set_ylabel("Density-ratio residual", color="#1f77b4", fontweight="bold")
            tax.set_ylabel("T(10 hPa) [K]", color="#dc143c", fontweight="bold")
            gax.set_ylabel(cfg["label"], color="#e67e22", fontweight="bold")
            ax.tick_params(axis="y", labelcolor="#1f77b4"); tax.tick_params(axis="y", labelcolor="#dc143c"); gax.tick_params(axis="y", labelcolor="#e67e22")
            ax.text(.012, .96, f"slope a={slope:.5f}  intercept b={intercept:.5f}  corr(R,G)={corr:.3f}", transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round,pad=.25", fc="white", ec=".25", alpha=.88))
            lines = l1 + l2 + l3; ax.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)
            ax.grid(ls=":", alpha=.35); plt.setp(ax.get_xticklabels(), visible=False)
            im = ax2.pcolormesh(mdates.date2num(edges), lat_edges, z, cmap="RdBu_r", vmin=-.2, vmax=.2, shading="flat")
            ax2.axvline(CENTRAL, color="red", ls="--"); ax2.set_ylim(-60, 60); ax2.set_xlim(START, END)
            ax2.set_ylabel("Geographic latitude [deg]", fontweight="bold"); ax2.set_xlabel("Date (UTC)", fontweight="bold")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); ax2.grid(ls=":", alpha=.25)
            fig.colorbar(im, cax=cax, label="density-ratio residual")
            fig.suptitle(f"2018/2019 NH major SSW (central date 2019-01-02) — SWARM-{satellite}\nstandardized MSIS ratio linearly detrended with {cfg['label']}; {detail}", fontsize=14, fontweight="bold")
            fig.savefig(figure_dir / f"SSW_2018_2019_NH_SWARM_{satellite}_{tag}.png", dpi=190)
            plt.close(fig)

    pd.DataFrame(metrics).to_csv(OUT / "linear_detrending_comparison.csv", index=False)
    pd.DataFrame(qc).to_csv(OUT / "qc_level_summary.csv", index=False)


if __name__ == "__main__":
    main()
