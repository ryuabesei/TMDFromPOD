# Linear detrending: all SSW events and geomagnetic indices

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
