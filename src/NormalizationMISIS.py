import pandas as pd

df_geo = pd.read_csv("data/Kpindex/SW-All.csv", parse_dates=["DATE"])
df_geo[["DATE", "F10.7_ADJ", "AP_AVG"]].head()
