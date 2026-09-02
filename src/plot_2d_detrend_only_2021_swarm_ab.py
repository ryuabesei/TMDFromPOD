"""Create the existing 2021 Ap-detrended 1D/2D figure for Swarm-A and B.

Only the satellite input and output name differ from the established Swarm-C
plot.  The date range, reference periods, ERA5 temperature, Ap regression,
latitude grid, aggregation, colour scale, and event marker are unchanged.
"""

from copy import deepcopy
from pathlib import Path

from plot_2d_detrend_only_3years import EVENTS, plot_event


STANDARDIZED = {
    "A": Path("standardizeddata/2021_abc_extension/2021/swarm_a_2021_msis_normalized.parquet"),
    "B": Path("standardizeddata/2021_abc_extension/2021/swarm_b_2021_msis_normalized.parquet"),
}


def main() -> None:
    template = next(event for event in EVENTS if event["year"] == 2021)
    for satellite, parquet in STANDARDIZED.items():
        event = deepcopy(template)
        event["label"] = f"2020/2021 NH SSW (SWARM-{satellite})"
        event["parquet"] = parquet
        event["output_name"] = f"2D_detrend_only_temp_ap_2021_SWARM-{satellite}.png"
        plot_event(event)


if __name__ == "__main__":
    main()
