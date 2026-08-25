"""Compare Ap/ap/ap60/ap30 against standardized Swarm MSIS density ratios.

The current same-UTC-day Ap assignment is preserved as the explicit baseline.
Higher-cadence predictors use completed GFZ intervals and causal trailing means.
Scientific unit: fixed 90-minute time block x AACGM latitude band median.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import aacgmv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GFZ = ROOT / "data/geomagnetic_gfz_high_cadence"
OUT = ROOT / "reports/high_cadence_geomagnetic_analysis"
INPUT = ROOT / "standardizeddata/karman_pretrained"
EVENTS = {"2018_NH": "2018", "2019_SH": "2019", "2021_NH": "2021"}
CADENCE_H = {"Ap": 24., "ap": 3., "ap60": 1., "ap30": .5}
WINDOWS = {"Ap": [24.], "ap": [3.,6.,12.,24.], "ap60": [1.,3.,6.,12.,24.], "ap30": [.5,1.,3.,6.]}
LAGS = {"Ap": [0.,12.,24.], "ap": [0.,3.,6.,9.,12.,18.,24.],
        "ap60": [0.,1.,2.,3.,4.,5.,6.,9.,12.], "ap30": [0.,.5,1.,1.5,2.,3.,4.,6.]}
MLAT_BANDS = ["all", "lt30", "30to60", "ge60"]
CENTRAL = {"2018_NH": pd.Timestamp("2018-02-12"), "2019_SH": pd.Timestamp("2019-09-19"),
           "2021_NH": pd.Timestamp("2021-01-04")}


def load_gfz(event: str, index: str) -> pd.Series:
    data = json.loads((GFZ / f"{event}_{index}_gfz.json").read_text())
    value_key = index if index in data else index.lower()
    s = pd.Series(data[value_key], index=pd.to_datetime(data["datetime"], utc=True), dtype=float).sort_index()
    s.index = s.index + pd.to_timedelta(CADENCE_H[index], unit="h")  # completed interval availability
    return s.replace(-1, np.nan)


def add_aacgm_lat(df: pd.DataFrame, event: str) -> np.ndarray:
    # Main-field evolution over a 1–2 month event is negligible for these broad bands.
    date = CENTRAL[event].to_pydatetime()
    mlat, _, _ = aacgmv2.convert_latlon_arr(df.lat.to_numpy(), df.lon.to_numpy(),
                                            df.alt_km.to_numpy(), date, method_code="G2A")
    return np.asarray(mlat)


def aggregate(path: Path, event: str, sat: str) -> tuple[pd.DataFrame, dict]:
    cols = ["datetime","lat","lon","alt_km","lst_h","density","rho_model_real",
            "density_ratio_msis","validity_flag"]
    d = pd.read_parquet(path, columns=cols)
    d["datetime"] = pd.to_datetime(d.datetime, utc=True)
    finite = np.isfinite(d.density_ratio_msis) & np.isfinite(d.density) & np.isfinite(d.rho_model_real)
    physical = (d.density > 0) & (d.rho_model_real > 0)
    valid_flag = d.validity_flag.fillna(1).eq(0)
    use = finite & physical & valid_flag
    q = d[use].copy()
    q["mlat"] = add_aacgm_lat(q, event)
    q["mlat_band"] = pd.cut(q.mlat.abs(), [-.001,30,60,90.001], labels=["lt30","30to60","ge60"])
    q["block"] = q.datetime.dt.floor("90min")
    agg = q.groupby(["block","mlat_band"], observed=True).agg(
        datetime=("datetime","median"), density_ratio_msis=("density_ratio_msis","median"),
        lat=("lat","median"), mlat=("mlat","median"), lst_h=("lst_h","median"), n_obs=("density_ratio_msis","size")).reset_index()
    overall = q.groupby("block").agg(datetime=("datetime","median"), density_ratio_msis=("density_ratio_msis","median"),
        lat=("lat","median"), mlat=("mlat","median"), lst_h=("lst_h","median"), n_obs=("density_ratio_msis","size")).reset_index()
    overall["mlat_band"] = "all"
    agg = pd.concat([overall, agg], ignore_index=True)
    agg.insert(0,"satellite",sat); agg.insert(0,"event",event)
    daily_counts = q.assign(day=q.datetime.dt.normalize()).groupby("day").size()
    ratio = q.density_ratio_msis
    jumps = overall.density_ratio_msis.diff().abs().dropna()
    jump_mad = float(np.median(np.abs(jumps-jumps.median())))
    jump_threshold = max(.10, float(jumps.median()+5*1.4826*jump_mad))
    qc = dict(event=event,satellite=sat,n_input=len(d),n_used=len(q),n_flagged=int((~valid_flag).sum()),
              n_nonfinite=int((~finite).sum()),n_nonpositive_density=int((d.density<=0).sum()),
              n_nonpositive_model=int((d.rho_model_real<=0).sum()),days=int(len(daily_counts)),
              missing_days=int(len(pd.date_range(daily_counts.index.min(),daily_counts.index.max(),freq="D").difference(daily_counts.index))),
              min_obs_per_day=int(daily_counts.min()),median_obs_per_day=float(daily_counts.median()),
              ratio_p001=float(ratio.quantile(.001)),ratio_p999=float(ratio.quantile(.999)),
              n_ratio_outside_0p1_3=int((~ratio.between(.1,3)).sum()),
              orbit_block_jump_threshold=jump_threshold,n_suspicious_orbit_block_jumps=int((jumps>jump_threshold).sum()),
              max_orbit_block_ratio_jump=float(jumps.max()))
    return agg, qc


def attach_driver(samples: pd.DataFrame, event: str, index: str, window: float, lag: float) -> np.ndarray:
    if index == "Ap":
        # Exact legacy convention: mean of the eight ap values for the same UTC day.
        # GFZ's JSON service normalizes `Ap` to the `ap` response, so calculate the
        # documented daily arithmetic mean explicitly from definitive 3-hour ap.
        raw = json.loads((GFZ / f"{event}_ap_gfz.json").read_text())
        three_hour = pd.Series(raw["ap"],index=pd.to_datetime(raw["datetime"],utc=True))
        byday = three_hour.resample("D").mean()
        key = (samples.datetime - pd.to_timedelta(lag,unit="h")).dt.normalize()
        return byday.reindex(key).to_numpy()
    s = load_gfz(event,index)
    roll = s.rolling(pd.Timedelta(hours=window), closed="right", min_periods=max(1,math.ceil(window/CADENCE_H[index]))).mean()
    left = pd.DataFrame({"cutoff":samples.datetime-pd.to_timedelta(lag,unit="h"),"order":np.arange(len(samples))}).sort_values("cutoff")
    right = pd.DataFrame({"available":roll.index,"driver":roll.values}).dropna().sort_values("available")
    merged = pd.merge_asof(left,right,left_on="cutoff",right_on="available",direction="backward")
    return merged.sort_values("order").driver.to_numpy()


def blocked_metrics(q: pd.DataFrame) -> dict:
    q=q.dropna(subset=["density_ratio_msis","driver"]).sort_values("datetime").copy()
    x=q.driver.to_numpy(); y=q.density_ratio_msis.to_numpy()
    if len(q)<20 or np.std(x)==0: return {}
    pear=stats.pearsonr(x,y); spear=stats.spearmanr(x,y); slope,intercept=np.polyfit(x,y,1)
    pred=np.full(len(q),np.nan); days=q.datetime.dt.normalize(); unique=days.drop_duplicates().to_numpy()
    folds=np.array_split(unique,min(4,len(unique)))
    for valdays in folds:
        test=days.isin(valdays).to_numpy(); train=~test
        if train.sum()<5 or test.sum()==0 or np.std(x[train])==0: continue
        b,a=np.polyfit(x[train],y[train],1); pred[test]=a+b*x[test]
    keep=np.isfinite(pred); residual=y[keep]-pred[keep]
    return dict(n_units=len(q),n_days=days.nunique(),pearson_r=pear.statistic,pearson_p_naive=pear.pvalue,
                spearman_r=spear.statistic,spearman_p_naive=spear.pvalue,slope=slope,intercept=intercept,r2=pear.statistic**2,
                cv_rmse=float(np.sqrt(np.mean(residual**2))),cv_mae=float(np.mean(np.abs(residual))),
                cv_residual_variance=float(np.var(residual,ddof=1)),
                cv_residual_driver_r=float(np.corrcoef(residual,x[keep])[0,1]),n_cv=len(residual))


def heatmaps(perf: pd.DataFrame) -> None:
    fdir=OUT/"figures/lag_window"; fdir.mkdir(parents=True,exist_ok=True)
    for band in MLAT_BANDS:
      for index in WINDOWS:
        q=perf[(perf.mlat_band==band)&(perf.geomagnetic_index==index)]
        grid=q.groupby(["window_h","lag_h"]).cv_rmse.mean().unstack()
        fig,ax=plt.subplots(figsize=(max(6,.7*len(grid.columns)),max(3.8,.65*len(grid))),layout="constrained")
        im=ax.imshow(grid,aspect="auto",cmap="viridis_r")
        ax.set_xticks(range(len(grid.columns)),grid.columns); ax.set_yticks(range(len(grid.index)),grid.index)
        ax.set_xlabel("Lag [h]"); ax.set_ylabel("Causal trailing window [h]")
        ax.set_title(f"{index}: mean blocked-CV RMSE across events/satellites — MLAT {band}")
        fig.colorbar(im,ax=ax,label="CV RMSE of density_ratio_msis")
        fig.savefig(fdir/f"lag_window_heatmap_{index}_mlat_{band}.png",dpi=180); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True); merged_dir=OUT/"merged_samples"; merged_dir.mkdir(exist_ok=True)
    config={"windows_h":WINDOWS,"lags_h":LAGS,"unit":"90-minute UTC block x AACGM latitude band median",
            "mlat_bands":MLAT_BANDS,"response":"standardized density_ratio_msis","qc":"QC0 only",
            "higher_cadence_timing":"GFZ interval-start shifted to interval-end; causal trailing averages",
            "Ap_baseline":"legacy same-UTC-day assignment; non-causal within the day",
            "validation":"4-fold contiguous UTC-day blocked CV"}
    (OUT/"lag_window_config.json").write_text(json.dumps(config,indent=2)+"\n")
    all_samples=[]; qc=[]
    for event,year in EVENTS.items():
      for path in sorted((INPUT/year).glob("swarm_*_karman.parquet")):
        sat=path.stem.split("_")[1].upper(); a,b=aggregate(path,event,sat); all_samples.append(a); qc.append(b)
    samples=pd.concat(all_samples,ignore_index=True); samples.to_parquet(merged_dir/"swarm_90min_mlat_samples.parquet",index=False)
    pd.DataFrame(qc).to_csv(OUT/"swarm_qc_2018_2019.csv",index=False)
    rows=[]
    for (event,sat,band),base in samples.groupby(["event","satellite","mlat_band"]):
      for index in WINDOWS:
       for window in WINDOWS[index]:
        for lag in LAGS[index]:
          q=base.copy(); q["driver"]=attach_driver(q,event,index,window,lag)
          m=blocked_metrics(q)
          if m: rows.append({"event":event,"satellite":sat,"mlat_band":band,"geomagnetic_index":index,
                             "window_h":window,"lag_h":lag,"is_legacy_Ap_baseline":index=="Ap" and lag==0,**m})
    perf=pd.DataFrame(rows); perf.to_csv(OUT/"lag_window_performance.csv",index=False)
    keys=["geomagnetic_index","window_h","lag_h"]
    primary=perf[perf.mlat_band=="all"]
    rank=primary.groupby(keys).agg(mean_cv_rmse=("cv_rmse","mean"),median_cv_rmse=("cv_rmse","median"),
          mean_abs_residual_driver_r=("cv_residual_driver_r",lambda x:np.mean(np.abs(x))),
          median_pearson_r=("pearson_r","median"),cases=("cv_rmse","size")).reset_index()
    rank["rmse_rank"]=rank.mean_cv_rmse.rank(method="min")
    rank=rank.sort_values(["mean_cv_rmse","mean_abs_residual_driver_r"])
    rank.to_csv(OUT/"geomagnetic_index_comparison.csv",index=False)
    best=rank.sort_values("mean_cv_rmse").groupby("geomagnetic_index",as_index=False).first().sort_values("mean_cv_rmse")
    best.to_csv(OUT/"best_geomagnetic_models.csv",index=False)
    bylat=perf.groupby(["mlat_band",*keys]).agg(mean_cv_rmse=("cv_rmse","mean"),median_cv_rmse=("cv_rmse","median"),
          mean_abs_residual_driver_r=("cv_residual_driver_r",lambda x:np.mean(np.abs(x))),cases=("cv_rmse","size")).reset_index()
    bylat.sort_values(["mlat_band","mean_cv_rmse"]).to_csv(OUT/"geomagnetic_index_comparison_by_mlat.csv",index=False)
    loo=[]
    for held in sorted(primary.event.unique()):
        train=primary[primary.event!=held]
        test=primary[primary.event==held]
        ranked=train.groupby(keys).cv_rmse.mean().sort_values()
        chosen=ranked.index[0]
        mask=np.logical_and.reduce([test[k].eq(v) for k,v in zip(keys,chosen)])
        base_mask=(test.geomagnetic_index=="Ap")&(test.window_h==24)&(test.lag_h==0)
        loo.append({"held_out_event":held,"selected_index":chosen[0],"selected_window_h":chosen[1],"selected_lag_h":chosen[2],
                    "training_mean_cv_rmse":ranked.iloc[0],"held_out_mean_cv_rmse":test[mask].cv_rmse.mean(),
                    "held_out_legacy_Ap_rmse":test[base_mask].cv_rmse.mean(),
                    "held_out_improvement_pct":100*(test[base_mask].cv_rmse.mean()-test[mask].cv_rmse.mean())/test[base_mask].cv_rmse.mean()})
    pd.DataFrame(loo).to_csv(OUT/"leave_one_event_out_selection.csv",index=False)
    heatmaps(perf)
    print(best.to_string(index=False))


if __name__=="__main__": main()
