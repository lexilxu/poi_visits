import os
import sys
import ast
import numpy as np
import pandas as pd
import torch

# Expand to daily visits
df = pd.read_csv("Seattle_all.csv")
df = df[["location_name", "placekey", "visits_by_day", "date_range_start"]]

df["date_range_start"] = pd.to_datetime(
    df["date_range_start"].astype(str).str.split("T").str[0],
    errors="coerce"
).dt.date
df["visits_by_day"] = df["visits_by_day"].apply(ast.literal_eval)
df_expanded = df.explode("visits_by_day").copy()

df_expanded["day_index"] = df_expanded.groupby(
    ["location_name", "placekey", "date_range_start"]
).cumcount()

df_expanded["date"] = pd.to_datetime(
    df_expanded["date_range_start"]
) + pd.to_timedelta(df_expanded["day_index"], unit="D")

df_expanded.rename(columns={"visits_by_day": "visits"}, inplace=True)

df_expanded = df_expanded[["location_name", "placekey", "date", "visits"]]
df = df_expanded.sort_values(
    ["placekey", "date"]
)

df_pivot = df.pivot(index="date", columns="placekey", values="visits")
df_pivot = df_pivot.sort_index()
df_pivot = df_pivot.fillna(0)
data = df_pivot.values.astype(float) 
# Multivariate time series
np.savetxt('mTS_only.txt', data, delimiter=',')


# Build location table
poi = pd.read_csv("Seattle_all_poi.csv")
# IMPORTANT: align location order
visit_placekeys = df["placekey"].dropna().drop_duplicates()
poi["placekey"] = poi["placekey"].astype(str)
location_table = poi[poi["placekey"].isin(visit_placekeys)].copy()

N = df_pivot.shape[1]
print(N)

location_table.to_csv("location_table.csv", index=False)




