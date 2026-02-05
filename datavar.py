from cdflib import CDF

path = "data/SWARM_C/SW_OPER_DNSCPOD_2__20180201T000000_20180201T235930_0301.cdf"
cdf = CDF(path)

for v in ["time","density","density_orbitmean","validity_flag","altitude","latitude","longitude","local_solar_time"]:
    at = cdf.varattsget(v)
    print("====", v, "====")
    for k in ["UNITS", "VARIABLE DESCRIPTION", "DESCRIPTION", "FILLVAL", "FORMAT"]:
        if k in at:
            print(f"{k}: {at[k]}")
