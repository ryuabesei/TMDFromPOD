# Density-ratio residual figures

This directory is the canonical, readable collection of latitude–time figures
for the linearly geomagnetic-detrended density ratio. Original figures remain
in their historical locations and are not overwritten.

## Naming convention

```text
{event}__{platform}__{reference}-density-ratio__{index}__linear-detrended__latitude-time.png
```

Example:

```text
2017-2018_NH_SSW__SWARM-A__standardized-MSIS-density-ratio__daily-Ap__linear-detrended__latitude-time.png
```

The double underscore separates metadata fields. The filename always states:

1. SSW event/season and hemisphere;
2. observing platform;
3. reference-density product;
4. geomagnetic index and cadence;
5. processing method;
6. plot geometry.

## Directory structure

```text
standardized_MSIS/
  event_window_daily_Ap/
  four_index_comparison/
    daily_Ap/
    ap_3hour/
    ap60_1hour/
    ap30_30min/
legacy_MSIS/
  daily_Ap/
```

Each index directory is divided by event. Use `standardized_MSIS` for current
scientific interpretation. `legacy_MSIS` is retained only for traceability and
must not be mixed with standardized results.

`event_window_daily_Ap` contains the dedicated standardized daily-Ap event
figures. `four_index_comparison` contains the directly comparable daily Ap,
3-hour ap, ap60, and ap30 figure set. They are kept separate because their
source pipelines and plotting windows are not assumed to be interchangeable.

`figure_manifest.csv` maps every canonical copy to its original source file.
The collection can be rebuilt with:

```bash
python src/organize_density_ratio_residual_figures.py
```
