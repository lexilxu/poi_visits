# src/prepare_data.py

import os
import ast
import numpy as np
import pandas as pd

def prepare_poi_visit_matrix(
    input_csv,
    output_dir,
    date_col="date_range_start",
    visits_col="visits_by_day",
    id_col="placekey",
    name_col="location_name",
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    df = df[[id_col, name_col, date_col, visits_col]].copy()

    # Keep only date part, removing time and timezone
    df[date_col] = pd.to_datetime(df[date_col].astype(str).str.split("T").str[0],
        errors="coerce").dt.date
    
    # Parse visits_by_day
    df[visits_col] = df[visits_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Expand daily visits
    df_expanded = df.explode(visits_col).copy()

    df_expanded["day_offset"] = df_expanded.groupby(
        [id_col, date_col]
    ).cumcount()

    df_expanded["date"] = pd.to_datetime(df_expanded[date_col]) + pd.to_timedelta(
        df_expanded["day_offset"], unit="D"
    )

    df_expanded["visits"] = pd.to_numeric(df_expanded[visits_col], errors="coerce")

    # Aggregate in case duplicate POI-date rows exist
    daily = (
        df_expanded.groupby([id_col, name_col, "date"], as_index=False)["visits"]
        .sum()
    )

    # Pivot to date x POI matrix
    visit_mat = daily.pivot(index="date", columns=id_col, values="visits")

    # Sort dates and POIs
    visit_mat = visit_mat.sort_index()
    visit_mat = visit_mat.sort_index(axis=1)

    # Drop POIs with any NaN
    visit_mat = visit_mat.dropna(axis=1, how="any")

    dates = visit_mat.index.to_frame(index=False)
    dates.columns = ["date"]

    poi_ids = visit_mat.columns.to_list()

    poi_meta = (
        daily[[id_col, name_col]]
        .drop_duplicates()
        .set_index(id_col)
        .loc[poi_ids]
        .reset_index()
    )

    visits = visit_mat.to_numpy(dtype=np.float32)

    np.save(os.path.join(output_dir, "visits.npy"), visits)
    dates.to_csv(os.path.join(output_dir, "dates.csv"), index=False)
    poi_meta.to_csv(os.path.join(output_dir, "poi_meta.csv"), index=False)

    print("Saved visits.npy with shape:", visits.shape)
    print("Number of dates:", visits.shape[0])
    print("Number of POIs:", visits.shape[1])


if __name__ == "__main__":
    prepare_poi_visit_matrix(
        input_csv="data/raw/Seattle_all.csv",
        output_dir="data/processed",
    )