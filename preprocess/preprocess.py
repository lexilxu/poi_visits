# Data preprocessing

import pandas as pd
from pathlib import Path
#import ast
#import googlemaps
#import time
#from geopy.geocoders import Nominatim
#from geopy.extra.rate_limiter import RateLimiter
#from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
#import ssl
#import certifi

# Downloading and concatenating datasets
folders_visits = [
    Path("./Data/MonthlyPattern/2020-11"),
    Path("./Data/MonthlyPattern/2020-12"),
    Path("./Data/MonthlyPattern/2021-01"),
]

folders_poi = [
    Path("./Data/POIs/2020-11"),
    Path("./Data/POIs/2020-12"),
    Path("./Data/POIs/2021-01"),
]

sea_frames = []
poi_frames = []
processed_dir = Path("./Data/processed")

for folder in folders_visits:
    for fp in folder.glob("*.csv.gz"):
        print(f"Processing {fp.name}")

        pieces = []
        for chunk in pd.read_csv(fp, compression="gzip", chunksize=200_000, low_memory=False):
            pieces.append(chunk)

        df = pd.concat(pieces, ignore_index=True)
        sea = df[df["city"] == "Seattle"]
        if not sea.empty:
                sea_frames.append(sea)
        out_path = processed_dir / fp.name.replace(".csv.gz", ".parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved {out_path}")
    


for folder in folders_poi:
    for fp in folder.glob("*.csv.gz"):
        print(f"Processing {fp.name}")

        pieces = []
        for chunk in pd.read_csv(fp, compression="gzip", chunksize=200_000, low_memory=False):
            pieces.append(chunk)
            print(chunk.head())

        df = pd.concat(pieces, ignore_index=True)
        sea = df[df["city"] == "Seattle"]
        if not sea.empty:
                poi_frames.append(sea)
        out_path = processed_dir / fp.name.replace(".csv.gz", ".parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved {out_path}")
    
            
sea_all = pd.concat(sea_frames, ignore_index=True)
sea_all = sea_all.sort_values("location_name")
sea_all = sea_all.drop_duplicates()
sea_all.to_csv("Seattle_all.csv", index=False)


sea_all = pd.concat(poi_frames, ignore_index=True)
sea_all = sea_all.sort_values("location_name")
sea_all = sea_all.drop_duplicates()
sea_all.to_csv("Seattle_all_poi.csv", index=False)

## Using google maps to find the coordinates (might not be needed anymore)
#
#ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
#
## read the CSV
#df = pd.read_csv("Seattle_all.csv")
#
#df["full_address"] = (
#    df["location_name"].astype(str) + ", " +
#    df["street_address"].astype(str) + ", " +
#    df["city"].astype(str) + ", " +
#    df["region"].astype(str) + " " +
#    df["postal_code"].astype(str)
#)
#
## Preprocessing
#df = df[["location_name", "full_address", "visits_by_day", "date_range_start"]]
#
#df["date_range_start"] = pd.to_datetime(df["date_range_start"]).dt.date
#df["visits_by_day"] = df["visits_by_day"].apply(ast.literal_eval)
#df_expanded = df.explode("visits_by_day").copy()
#
#df_expanded["day_index"] = df_expanded.groupby(
#    ["location_name", "full_address", "date_range_start"]
#).cumcount()
#
#df_expanded["date"] = pd.to_datetime(
#    df_expanded["date_range_start"]
#) + pd.to_timedelta(df_expanded["day_index"], unit="D")
#
#df_expanded.rename(columns={"visits_by_day": "visits"}, inplace=True)
#
#df_expanded = df_expanded[["location_name", "full_address", "date", "visits"]]
#df_expanded = df_expanded.sort_values(
#    ["location_name", "date"]
#)
#
#unique_addresses = df_expanded["full_address"].drop_duplicates()
#
## Initialize Google Maps client
#gmaps = googlemaps.Client(key="AIzaSyBcPZcvu80kRlCI67HH0gMPThUWpeotDik")
#
#def get_coordinates(address):
#    try:
#        result = gmaps.geocode(address)
#        if result:
#            location = result[0]["geometry"]["location"]
#            return location["lat"], location["lng"]
#        return None, None
#    except Exception as e:
#        print("Error:", e)
#        return None, None
#
## Apply geocoding
#coords = unique_addresses.apply(
#    lambda x: pd.Series(get_coordinates(x))
#)
#
#coords.columns = ["latitude", "longitude"]
#
#geo_df = pd.concat(
#    [unique_addresses.reset_index(drop=True), coords],
#    axis=1
#)
#
#df = df.merge(geo_df, on="full_address", how="left")
#
## Save to the processed.csv file
#df.to_csv("processed.csv", index=False)








