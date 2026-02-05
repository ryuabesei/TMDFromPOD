from cdflib import CDF

path = "SW_OPER_DNSCPOD_2__20180201T000000_20180201T235930_0301.cdf"
cdf = CDF(path)

info = cdf.cdf_info()
print("zVariables:", info["zVariables"])
print("rVariables:", info["rVariables"])
print("Attributes:", info["Attributes"])
