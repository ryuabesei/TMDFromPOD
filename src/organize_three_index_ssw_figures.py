"""Collect all daily-Ap/ap/ap60 SSW figures under publication-safe names."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "reports" / "daily_Ap_ap_ap60_linear_detrending" / "figures"
NEW = ROOT / "reports" / "2018_2019_NH_SSW" / "three_index_linear_detrending" / "figures"
GRACE = ROOT / "reports" / "2020_2021_NH_GRACE_FO_three_index_linear_detrending" / "figures"
OUT = ROOT / "reports" / "three_index_linear_detrending_all_ssw"
FIGURES = OUT / "figures"

EVENTS = {
    "2017-2018_NH_SSW": {
        "display": "2017/2018 NH SSW",
        "source_dir": OLD,
        "source_event": "2018_NH",
        "central_date": "2018-02-12",
    },
    "2018-2019_NH_SSW": {
        "display": "2018/2019 NH SSW",
        "source_dir": NEW,
        "source_event": "2018_2019_NH",
        "central_date": "2019-01-02",
    },
    "2019_SH_SSW": {
        "display": "2019 SH SSW",
        "source_dir": OLD,
        "source_event": "2019_SH",
        "central_date": "2019-09-19",
    },
    "2020-2021_NH_SSW": {
        "display": "2020/2021 NH SSW",
        "source_dir": OLD,
        "source_event": "2021_NH",
        "central_date": "2021-01-04",
    },
}

METHODS = {
    "daily-Ap": "daily_Ap",
    "ap-3hour": "ap_3hour",
    "ap60-1hour": "ap60_1hour",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    for event_tag, event in EVENTS.items():
        for satellite in "ABC":
            for method_tag, source_method in METHODS.items():
                source = event["source_dir"] / (
                    f"SSW_{event['source_event']}_SWARM_{satellite}_{source_method}.png"
                )
                target = FIGURES / f"{event_tag}_SWARM-{satellite}_{method_tag}_linear-detrending.png"
                if not source.exists():
                    raise FileNotFoundError(source)
                if target.exists():
                    if digest(target) != digest(source):
                        raise FileExistsError(f"Existing collected figure differs: {target}")
                else:
                    shutil.copy2(source, target)
                rows.append({
                    "event_display": event["display"],
                    "event_file_tag": event_tag,
                    "central_date": event["central_date"],
                    "satellite": f"Swarm-{satellite}",
                    "geomagnetic_method": method_tag,
                    "source_figure": str(source.relative_to(ROOT)),
                    "collected_figure": str(target.relative_to(ROOT)),
                    "sha256": digest(target),
                })
    for method_tag, source_method in METHODS.items():
        source = GRACE / f"SSW_2021_NH_GRACE_FO_{source_method}.png"
        target = FIGURES / f"2020-2021_NH_SSW_GRACE-FO_{method_tag}_linear-detrending.png"
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists():
            if digest(target) != digest(source):
                raise FileExistsError(f"Existing collected figure differs: {target}")
        else:
            shutil.copy2(source, target)
        rows.append({
            "event_display": "2020/2021 NH SSW",
            "event_file_tag": "2020-2021_NH_SSW",
            "central_date": "2021-01-04",
            "satellite": "GRACE-FO",
            "geomagnetic_method": method_tag,
            "source_figure": str(source.relative_to(ROOT)),
            "collected_figure": str(target.relative_to(ROOT)),
            "sha256": digest(target),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "figure_manifest.csv", index=False)

    old_metrics = pd.read_csv(
        ROOT / "reports/daily_Ap_ap_ap60_linear_detrending/linear_detrending_comparison.csv"
    ).rename(columns={"index": "method_key", "corr": "correlation"})
    old_metrics["event_display"] = old_metrics["event"].map({
        "2018_NH": "2017/2018 NH SSW",
        "2019_SH": "2019 SH SSW",
        "2021_NH": "2020/2021 NH SSW",
    })
    new_metrics = pd.read_csv(
        ROOT / "reports/2018_2019_NH_SSW/three_index_linear_detrending/linear_detrending_comparison.csv"
    ).rename(columns={"geomagnetic_index": "method_key"})
    new_metrics["event_display"] = "2018/2019 NH SSW"
    method_names = {
        "Ap": "daily-Ap", "daily Ap": "daily-Ap",
        "ap": "ap-3hour", "3-hour ap": "ap-3hour",
        "ap60": "ap60-1hour", "1-hour ap60": "ap60-1hour",
    }
    grace_metrics = pd.read_csv(
        ROOT / "reports/2020_2021_NH_GRACE_FO_three_index_linear_detrending/linear_detrending_comparison.csv"
    ).rename(columns={"index": "method_key", "corr": "correlation"})
    grace_metrics["event_display"] = "2020/2021 NH SSW"
    combined = pd.concat([old_metrics, new_metrics, grace_metrics], ignore_index=True, sort=False)
    combined["geomagnetic_method"] = combined["method_key"].map(method_names)
    metric_columns = [
        "event_display", "satellite", "geomagnetic_method", "window_h", "lag_h",
        "slope", "intercept", "correlation", "pre_days", "ssw_days",
        "residual_pre_median", "residual_ssw_median", "ssw_minus_pre_residual_pp",
    ]
    combined[metric_columns].sort_values(
        ["event_display", "satellite", "geomagnetic_method"]
    ).to_csv(OUT / "linear_detrending_metrics_all_events.csv", index=False)

    lines = [
        "# Daily Ap / 3-hour ap / ap60 linear-detrending figures",
        "",
        "This folder is the single entry point for the four analyzed SSW events.",
        "The original figures remain in their source report directories.",
        "",
        "## Naming",
        "",
        "- `2017-2018_NH_SSW`: central date 2018-02-12",
        "- `2018-2019_NH_SSW`: central date 2019-01-02",
        "- `2019_SH_SSW`: current plot marker 2019-09-19",
        "- `2020-2021_NH_SSW`: central date 2021-01-04",
        "- methods: `daily-Ap`, `ap-3hour`, `ap60-1hour`",
        "",
        "There are 36 Swarm figures plus 3 GRACE-FO figures for 2020/2021 NH SSW = 39 figures.",
        "See `figure_manifest.csv` for the exact source, destination, and SHA-256.",
        "See `linear_detrending_metrics_all_events.csv` for the harmonized regression results.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Collected {len(manifest)} figures in {FIGURES}")


if __name__ == "__main__":
    main()
