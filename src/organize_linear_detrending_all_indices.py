"""Collect daily Ap, 3-hour ap, ap60, and ap30 plots under one directory."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "linear_detrending"
FIGURES = OUT / "figures"
OLD = ROOT / "reports/daily_Ap_ap_ap60_linear_detrending/figures"
NH1819 = ROOT / "reports/2018_2019_NH_SSW/three_index_linear_detrending/figures"
GRACE = ROOT / "reports/2020_2021_NH_GRACE_FO_three_index_linear_detrending/figures"
AP30 = ROOT / "reports/ap30_linear_detrending_all_ssw/figures"

EVENTS = {
    "2017-2018_NH_SSW": dict(display="2017/2018 NH SSW", key="2018_NH", central="2018-02-12", source=OLD),
    "2018-2019_NH_SSW": dict(display="2018/2019 NH SSW", key="2018_2019_NH", central="2019-01-02", source=NH1819),
    "2019_SH_SSW": dict(display="2019 SH SSW", key="2019_SH", central="2019-09-19", source=OLD),
    "2020-2021_NH_SSW": dict(display="2020/2021 NH SSW", key="2021_NH", central="2021-01-04", source=OLD),
}
METHODS = {
    "daily-Ap": "daily_Ap",
    "ap-3hour": "ap_3hour",
    "ap60-1hour": "ap60_1hour",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_checked(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists():
        if sha256(source) != sha256(target):
            raise FileExistsError(f"Existing target differs: {target}")
    else:
        shutil.copy2(source, target)


def standard_source(event: dict, satellite: str, method: str) -> Path:
    if event["key"] == "2018_2019_NH":
        return event["source"] / f"SSW_2018_2019_NH_SWARM_{satellite}_{method}.png"
    return event["source"] / f"SSW_{event['key']}_SWARM_{satellite}_{method}.png"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    for event_tag, event in EVENTS.items():
        platforms = ["SWARM-A", "SWARM-B", "SWARM-C"]
        if event_tag == "2020-2021_NH_SSW":
            platforms.append("GRACE-FO")
        for platform in platforms:
            sat = platform[-1] if platform.startswith("SWARM-") else None
            for method_tag, method_source_tag in METHODS.items():
                if platform == "GRACE-FO":
                    source = GRACE / f"SSW_2021_NH_GRACE_FO_{method_source_tag}.png"
                else:
                    source = standard_source(event, sat, method_source_tag)
                target = FIGURES / f"{event_tag}_{platform}_{method_tag}_linear-detrending.png"
                copy_checked(source, target)
                rows.append(dict(event=event["display"],event_file_tag=event_tag,central_date=event["central"],platform=platform,geomagnetic_index=method_tag,window_h={"daily-Ap":24,"ap-3hour":24,"ap60-1hour":24}[method_tag],lag_h={"daily-Ap":0,"ap-3hour":0,"ap60-1hour":1}[method_tag],source_figure=str(source.relative_to(ROOT)),collected_figure=str(target.relative_to(ROOT)),sha256=sha256(target)))
            source = AP30 / f"{event_tag}_{platform}_ap30_linear_detrending.png"
            target = FIGURES / f"{event_tag}_{platform}_ap30-30min_linear-detrending.png"
            copy_checked(source, target)
            rows.append(dict(event=event["display"],event_file_tag=event_tag,central_date=event["central"],platform=platform,geomagnetic_index="ap30-30min",window_h=6,lag_h=6,source_figure=str(source.relative_to(ROOT)),collected_figure=str(target.relative_to(ROOT)),sha256=sha256(target)))

    manifest = pd.DataFrame(rows).sort_values(["event","platform","geomagnetic_index"])
    manifest.to_csv(OUT/"figure_manifest.csv", index=False)

    base = pd.read_csv(ROOT/"reports/three_index_linear_detrending_all_ssw/linear_detrending_metrics_all_events.csv")
    base = base.rename(columns={"event_display":"event","satellite":"platform"})
    base["platform"] = base["platform"].astype(str).map({"A":"SWARM-A","B":"SWARM-B","C":"SWARM-C","GRACE-FO":"GRACE-FO"}).fillna(base["platform"])
    ap30 = pd.read_csv(AP30.parent/"ap30_linear_detrending_metrics.csv")
    ap30["event_name"] = ap30["event_display"].replace({"2018/2019 NH major SSW":"2018/2019 NH SSW"})
    ap30["geomagnetic_method"] = "ap30-30min"
    ap30["platform"] = ap30["satellite"]
    ap30_metrics = ap30[["event_name","platform","geomagnetic_method","window_h","lag_h","slope","intercept","correlation","pre_days","ssw_days","residual_pre_median","residual_ssw_median","ssw_minus_pre_residual_pp"]].rename(columns={"event_name":"event"})
    metric_cols = ["event","platform","geomagnetic_method","window_h","lag_h","slope","intercept","correlation","pre_days","ssw_days","residual_pre_median","residual_ssw_median","ssw_minus_pre_residual_pp"]
    combined = pd.concat([base[metric_cols],ap30_metrics],ignore_index=True).sort_values(["event","platform","geomagnetic_method"])
    combined.to_csv(OUT/"linear_detrending_metrics.csv",index=False)

    readme = """# Linear detrending: all SSW events and geomagnetic indices

This is the single entry point for the standardized MSIS density-ratio linear-detrending plots.
Existing source figures remain unchanged in their original report directories.

## Events

- 2017/2018 NH SSW (central date: 2018-02-12)
- 2018/2019 NH SSW (central date: 2019-01-02)
- 2019 SH SSW (plot marker: 2019-09-19)
- 2020/2021 NH SSW (central date: 2021-01-04)

## Geomagnetic drivers

- `daily-Ap`: same-UTC-day daily Ap, lag 0 h
- `ap-3hour`: 3-hour ap, 24 h causal trailing mean, lag 0 h
- `ap60-1hour`: 1-hour ap60, 24 h causal trailing mean, lag 1 h
- `ap30-30min`: 30-minute ap30, 6 h causal trailing mean, lag 6 h

There are 13 platform-event cases x 4 drivers = 52 figures. The 13 cases are Swarm A/B/C for all four events plus GRACE-FO for 2020/2021 NH SSW.

See `figure_manifest.csv` for exact provenance and SHA-256 hashes. See `linear_detrending_metrics.csv` for harmonized regression and pre/SSW summaries.
"""
    (OUT/"README.md").write_text(readme)
    print(f"Collected {len(manifest)} figures in {FIGURES}")


if __name__ == "__main__":
    main()
