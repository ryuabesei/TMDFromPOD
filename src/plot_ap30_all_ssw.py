"""Create true ap30 (30-minute) linear-detrending plots for all SSW cases."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_2d_detrend_only_3years import compute_2d_grid, load_temp
from run_high_cadence_geomagnetic_analysis import ROOT, attach_driver


OUT = ROOT / "reports" / "ap30_linear_detrending_all_ssw"
WINDOW_H = 6.0
LAG_H = 6.0
EVENTS = {
    "2018_NH": dict(display="2017/2018 NH SSW", start="2018-01-20", end="2018-03-20 23:59:59", central="2018-02-12", ssw_end="2018-02-28 23:59:59", input=ROOT/"standardizeddata/karman_pretrained/2018", pattern="swarm_*_karman.parquet", temp="cosmic"),
    "2018_2019_NH": dict(display="2018/2019 NH major SSW", start="2018-12-10", end="2019-02-10 23:59:59", central="2019-01-02", ssw_end="2019-01-20 23:59:59", input=ROOT/"standardizeddata/2018_2019_NH_SSW", pattern="swarm_*_msis_normalized.parquet", temp="era5_2019nh"),
    "2019_SH": dict(display="2019 SH SSW", start="2019-08-20", end="2019-10-05 23:59:59", central="2019-09-19", ssw_end="2019-10-05 23:59:59", input=ROOT/"standardizeddata/karman_pretrained/2019", pattern="swarm_*_karman.parquet", temp="era5_sh"),
    "2021_NH": dict(display="2020/2021 NH SSW", start="2020-12-25", end="2021-02-05 23:59:59", central="2021-01-04", ssw_end="2021-01-16 23:59:59", input=ROOT/"standardizeddata/karman_pretrained/2021", pattern="swarm_*_karman.parquet", temp="era5_2021nh"),
}


def temperature(kind: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if kind == "cosmic":
        spec = dict(temp_type="cosmic", temp_file=ROOT/"cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv", date_start=start, date_end=end)
    elif kind == "era5_2019nh":
        spec = dict(temp_type="era5", temp_dir=ROOT/"data/SSW2018_2019_NH/ERA5", lat_range=(60.,90.), date_start=start, date_end=end)
    elif kind == "era5_sh":
        spec = dict(temp_type="era5", temp_dir=ROOT/"data/SSW2019/ERA5", lat_range=(-90.,-60.), date_start=start, date_end=end)
    else:
        spec = dict(temp_type="era5", temp_dir=ROOT/"data/SSW2021/ERA5", lat_range=(60.,90.), date_start=start, date_end=end)
    return load_temp(spec)


def satellite_name(path: Path) -> str:
    if path.name.startswith("grace_fo"):
        return "GRACE-FO"
    return f"SWARM-{path.name.split('_')[1].upper()}"


def main() -> None:
    fdir = OUT / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for event, cfg in EVENTS.items():
        start = pd.Timestamp(cfg["start"], tz="UTC"); end = pd.Timestamp(cfg["end"], tz="UTC")
        central = pd.Timestamp(cfg["central"], tz="UTC"); ssw_end = pd.Timestamp(cfg["ssw_end"], tz="UTC")
        temp = temperature(cfg["temp"], start, end)
        paths = sorted(cfg["input"].glob(cfg["pattern"]))
        if event == "2021_NH":
            paths += [cfg["input"] / "grace_fo_2021_karman.parquet"]
        for path in paths:
            sat = satellite_name(path)
            raw = pd.read_parquet(path)
            raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
            use = (raw["datetime"].between(start, end) & np.isfinite(raw["density_ratio_msis"])
                   & np.isfinite(raw["density"]) & np.isfinite(raw["rho_model_real"])
                   & (raw["density"] > 0) & (raw["rho_model_real"] > 0)
                   & (raw["density_ratio_msis"] >= 0))
            raw = raw.loc[use].copy()
            raw["block"] = raw["datetime"].dt.floor("90min")
            blocks = raw.groupby("block", as_index=False).agg(datetime=("datetime","median"), density_ratio_msis=("density_ratio_msis","median"))
            blocks["driver"] = attach_driver(blocks, event, "ap30", WINDOW_H, LAG_H)
            fit = blocks.dropna(subset=["driver","density_ratio_msis"])
            slope, intercept = np.polyfit(fit.driver, fit.density_ratio_msis, 1)
            corr = float(np.corrcoef(fit.driver, fit.density_ratio_msis)[0,1])
            raw["driver"] = attach_driver(raw, event, "ap30", WINDOW_H, LAG_H)
            raw["residual"] = raw.density_ratio_msis - (intercept + slope*raw.driver)
            raw["date"] = raw.datetime.dt.normalize()
            daily = raw.groupby("date").agg(residual=("residual","median"), driver=("driver","mean"))
            pre = daily.loc[daily.index < central, "residual"]
            during = daily.loc[daily.index.to_series().between(central, ssw_end), "residual"]
            rows.append(dict(event=event,event_display=cfg["display"],satellite=sat,index="ap30",window_h=WINDOW_H,lag_h=LAG_H,slope=slope,intercept=intercept,correlation=corr,pre_days=len(pre),ssw_days=len(during),residual_pre_median=pre.median(),residual_ssw_median=during.median(),ssw_minus_pre_residual_pp=100*(during.median()-pre.median())))

            edges = pd.date_range(start.normalize(), end.normalize()+pd.Timedelta(days=1), freq="D")
            lat_edges = np.arange(-60,63,3); z = compute_2d_grid(raw,"residual",edges,lat_edges)
            fig = plt.figure(figsize=(14,8),layout="constrained")
            gs = fig.add_gridspec(2,2,width_ratios=[1,.045],height_ratios=[1,1.55])
            ax=fig.add_subplot(gs[0,0]); spacer=fig.add_subplot(gs[0,1]); spacer.axis("off")
            ax2=fig.add_subplot(gs[1,0],sharex=ax); cax=fig.add_subplot(gs[1,1])
            tax=ax.twinx(); gax=ax.twinx(); gax.spines["right"].set_position(("axes",1.08))
            l1=ax.plot(daily.index,daily.residual,"o-",lw=2,ms=3.5,color="#1f77b4",label="ratio residual after ap30")
            l2=tax.plot(temp.index,temp.values,"^-",lw=2,ms=3.5,color="#dc143c",label="T(10 hPa) polar-cap mean")
            l3=gax.plot(daily.index,daily.driver,"s-",lw=1.4,ms=3.5,color="#e67e22",label=f"30-min ap30, {WINDOW_H:g} h causal trailing, lag {LAG_H:g} h")
            ax.axhline(0,color=".4",ls=":"); ax.axvline(central,color="red",ls="--")
            ax.set_ylabel("Density-ratio residual",color="#1f77b4",fontweight="bold"); tax.set_ylabel("T(10 hPa) [K]",color="#dc143c",fontweight="bold"); gax.set_ylabel("ap30",color="#e67e22",fontweight="bold")
            ax.tick_params(axis="y",labelcolor="#1f77b4"); tax.tick_params(axis="y",labelcolor="#dc143c"); gax.tick_params(axis="y",labelcolor="#e67e22")
            ax.text(.012,.96,f"slope a={slope:.5f}  intercept b={intercept:.5f}  corr(R,G)={corr:.3f}",transform=ax.transAxes,va="top",fontsize=8,bbox=dict(boxstyle="round,pad=.25",fc="white",ec=".25",alpha=.88))
            lines=l1+l2+l3; ax.legend(lines,[x.get_label() for x in lines],loc="upper right",fontsize=8); ax.grid(ls=":",alpha=.35); plt.setp(ax.get_xticklabels(),visible=False)
            im=ax2.pcolormesh(mdates.date2num(edges),lat_edges,z,cmap="RdBu_r",vmin=-.2,vmax=.2,shading="flat")
            ax2.axvline(central,color="red",ls="--"); ax2.set_ylim(-60,60); ax2.set_xlim(start,end)
            ax2.set_ylabel("Geographic latitude [deg]",fontweight="bold"); ax2.set_xlabel("Date (UTC)",fontweight="bold"); ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); ax2.grid(ls=":",alpha=.25)
            fig.colorbar(im,cax=cax,label="density-ratio residual")
            fig.suptitle(f"{cfg['display']} {sat} — standardized MSIS ratio linearly detrended with ap30\n30-minute ap30; {WINDOW_H:g} h causal trailing window; lag {LAG_H:g} h",fontsize=12.5,fontweight="bold")
            tag = cfg["display"].replace("/","-").replace(" ","_").replace("major_","")
            fig.savefig(fdir/f"{tag}_{sat}_ap30_linear_detrending.png",dpi=190); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/"ap30_linear_detrending_metrics.csv",index=False)
    print(f"Generated {len(rows)} true ap30 figures in {fdir}")


if __name__ == "__main__":
    main()
