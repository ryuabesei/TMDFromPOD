"""Regenerate poster-style SSW plots with Ap, ap, and ap60 corrections."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_2d_detrend_only_3years import compute_2d_grid, load_temp
from run_high_cadence_geomagnetic_analysis import ROOT, OUT, INPUT, attach_driver

COMPARISON_OUT = ROOT / "reports" / "daily_Ap_ap_ap60_linear_detrending"

METHODS={"Ap":(24.,0.),"ap":(24.,0.),"ap60":(24.,1.)}
FILE_TAG={"Ap":"daily_Ap","ap":"ap_3hour","ap60":"ap60_1hour"}
DISPLAY_EVENT={"2018_NH":"2017/2018 NH SSW","2019_SH":"2019 SH SSW","2021_NH":"2020/2021 NH SSW"}
EVENTS={
 "2018_NH":dict(year="2018",start="2018-01-20",end="2018-03-20 23:59:59",onset="2018-02-12",ssw_end="2018-02-28 23:59:59",temp="cosmic"),
 "2019_SH":dict(year="2019",start="2019-08-20",end="2019-10-05 23:59:59",onset="2019-09-04",ssw_end="2019-09-19 23:59:59",peak="2019-09-19",temp="era5"),
 "2021_NH":dict(year="2021",start="2020-12-25",end="2021-02-05 23:59:59",onset="2021-01-04",ssw_end="2021-01-16 23:59:59",temp="era5"),
}


def temp_series(event,cfg,start,end):
    if cfg["temp"]=="cosmic": spec=dict(temp_type="cosmic",temp_file=ROOT/"cosmic_T10hPa_daily_2018_DOY020_080_lat60_90N.csv",date_start=start,date_end=end)
    elif event=="2019_SH": spec=dict(temp_type="era5",temp_dir=ROOT/"data/SSW2019/ERA5",lat_range=(-90.,-60.),date_start=start,date_end=end)
    else: spec=dict(temp_type="era5",temp_dir=ROOT/"data/SSW2021/ERA5",lat_range=(60.,90.),date_start=start,date_end=end)
    return load_temp(spec)


def main():
    fdir=COMPARISON_OUT/"figures"; fdir.mkdir(parents=True,exist_ok=True)
    samples=pd.read_parquet(OUT/"merged_samples/swarm_90min_mlat_samples.parquet")
    rows=[]
    for event,cfg in EVENTS.items():
      start=pd.Timestamp(cfg["start"],tz="UTC"); end=pd.Timestamp(cfg["end"],tz="UTC")
      onset=pd.Timestamp(cfg["onset"],tz="UTC"); ssw_end=pd.Timestamp(cfg["ssw_end"],tz="UTC")
      peak=pd.Timestamp(cfg.get("peak",cfg["onset"]),tz="UTC"); temp=temp_series(event,cfg,start,end)
      for path in sorted((INPUT/cfg["year"]).glob("swarm_*_karman.parquet")):
       sat=path.stem.split("_")[1].upper(); raw=pd.read_parquet(path)
       raw["datetime"]=pd.to_datetime(raw.datetime,utc=True); raw=raw[raw.datetime.between(start,end)&raw.density_ratio_msis.notna()].copy()
       base=samples[(samples.event==event)&(samples.satellite==sat)&(samples.mlat_band=="all")].copy()
       for index,(window,lag) in METHODS.items():
        base["driver"]=attach_driver(base,event,index,window,lag); fit=base.dropna(subset=["driver","density_ratio_msis"])
        slope,intercept=np.polyfit(fit.driver,fit.density_ratio_msis,1)
        corr=float(np.corrcoef(fit.driver,fit.density_ratio_msis)[0,1])
        raw["driver"]=attach_driver(raw,event,index,window,lag)
        raw["residual"]=raw.density_ratio_msis-(intercept+slope*raw.driver)
        raw["date"]=raw.datetime.dt.normalize(); daily=raw.groupby("date").agg(residual=("residual","median"),driver=("driver","mean"))
        pre=daily[daily.index<onset].residual; during=daily[(daily.index>=onset)&(daily.index<=ssw_end)].residual
        rows.append(dict(event=event,satellite=sat,index=index,window_h=window,lag_h=lag,slope=slope,intercept=intercept,corr=corr,
                         pre_days=len(pre),ssw_days=len(during),residual_pre_median=pre.median(),residual_ssw_median=during.median(),
                         ssw_minus_pre_residual_pp=100*(during.median()-pre.median())))
        edges=pd.date_range(start.normalize(),end.normalize()+pd.Timedelta(days=1),freq="D"); lat_edges=np.arange(-60,63,3)
        z=compute_2d_grid(raw,"residual",edges,lat_edges)
        fig=plt.figure(figsize=(14,8),layout="constrained"); gs=fig.add_gridspec(2,2,width_ratios=[1,.045],height_ratios=[1,1.55])
        ax=fig.add_subplot(gs[0,0]); sp=fig.add_subplot(gs[0,1]);sp.axis("off"); ax2=fig.add_subplot(gs[1,0],sharex=ax);cax=fig.add_subplot(gs[1,1])
        at=ax.twinx(); ag=ax.twinx();ag.spines["right"].set_position(("axes",1.08))
        l1=ax.plot(daily.index,daily.residual,"o-",lw=2,ms=3.5,color="#1f77b4",label=f"ratio residual after {index}")
        l2=at.plot(temp.index,temp.values,"^-",lw=2,ms=3.5,color="#dc143c",label="T(10 hPa) polar-cap mean") if len(temp) else []
        if index == "Ap":
            driver_label="daily Ap (same UTC day; legacy baseline)"
            method_subtitle="same-UTC-day daily Ap; legacy linear detrending"
        else:
            native={"ap":"3-hour ap","ap60":"1-hour ap60"}[index]
            driver_label=f"{native}, {window:g} h causal trailing, lag {lag:g} h"
            method_subtitle=f"{native}; {window:g} h causal trailing window; lag {lag:g} h"
        l3=ag.plot(daily.index,daily.driver,"s-",lw=1.4,ms=3.5,color="#e67e22",label=driver_label)
        ax.axhline(0,color=".4",ls=":");ax.axvline(peak,color="red",ls="--")
        ax.set_ylabel("Density-ratio residual",color="#1f77b4",fontweight="bold");at.set_ylabel("T(10 hPa) [K]",color="#dc143c",fontweight="bold");ag.set_ylabel(index,color="#e67e22",fontweight="bold")
        ax.tick_params(axis="y",labelcolor="#1f77b4");at.tick_params(axis="y",labelcolor="#dc143c");ag.tick_params(axis="y",labelcolor="#e67e22")
        ax.text(.012,.96,f"slope a={slope:.5f}  intercept b={intercept:.5f}  corr(R,G)={corr:.3f}",
                transform=ax.transAxes,va="top",fontsize=8,
                bbox=dict(boxstyle="round,pad=.25",fc="white",ec=".25",alpha=.88))
        lines=l1+l2+l3;ax.legend(lines,[x.get_label() for x in lines],loc="upper right",fontsize=8);ax.grid(ls=":",alpha=.35);plt.setp(ax.get_xticklabels(),visible=False)
        im=ax2.pcolormesh(mdates.date2num(edges),lat_edges,z,cmap="RdBu_r",vmin=-.2,vmax=.2,shading="flat");ax2.axvline(peak,color="red",ls="--")
        ax2.set_ylim(-60,60);ax2.set_xlim(start,end);ax2.set_ylabel("Geographic latitude [deg]",fontweight="bold");ax2.set_xlabel("Date (UTC)",fontweight="bold");ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"));ax2.grid(ls=":",alpha=.25)
        fig.colorbar(im,cax=cax,label="density-ratio residual")
        fig.suptitle(f"{DISPLAY_EVENT[event]} SWARM-{sat} — standardized MSIS ratio linearly detrended with {index}\n{method_subtitle}",fontsize=14,fontweight="bold")
        fig.savefig(fdir/f"SSW_{event}_SWARM_{sat}_{FILE_TAG[index]}.png",dpi=190);plt.close(fig)
    pd.DataFrame(rows).to_csv(COMPARISON_OUT/"linear_detrending_comparison.csv",index=False)

if __name__=="__main__":main()
