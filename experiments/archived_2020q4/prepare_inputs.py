"""Verify and package the approved October--December panel for the formal run."""
import ast
from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
PREP = PROJECT / "dataset_preparation"
REPO = PROJECT / "poi_visits"
COMMIT = "3e5e14f82949b3e0d90e7fbb46d64d7ef48f23ba"


def category_mapping():
    tree = ast.parse((REPO / "attri_exp/exps/exp1_distribute.py").read_text(encoding="utf-8"))
    return next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CATEGORY_MAP" for t in node.targets)
    )


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    names = ("approved_dates.csv", "identified_9884_pois.csv",
             "approved_period_visits_with_missing.npy", "approved_period_source.csv.gz")
    dates_file, ids_file, visits_file, months_file = [PREP / name for name in names]
    days = pd.read_csv(dates_file)["date"]
    ids = pd.read_csv(ids_file)["placekey"]
    visits = np.load(visits_file, allow_pickle=False)
    assert days.tolist() == pd.date_range("2020-10-01", "2020-12-31").strftime("%Y-%m-%d").tolist()
    assert len(ids) == ids.nunique() == 9884 and ids.is_monotonic_increasing
    assert visits.shape == (92, 9884) and visits.dtype == np.float32
    observed = np.isfinite(visits)
    assert observed.sum() == 905918 and (~observed).sum() == 3410
    assert not (~observed)[31:].any() and (~observed).any(axis=0).sum() == 110

    # Reconstruct the approved cohort independently from the monthly source rows.
    months = pd.read_csv(months_file, usecols=["placekey", "date_range_start", "visits_by_day"])
    months = months.loc[months.placekey.isin(ids)].copy()
    months["month"] = pd.to_datetime(months.date_range_start.str[:10])
    assert not months.duplicated(["placekey", "month"]).any()
    months["values"] = months.visits_by_day.map(ast.literal_eval)
    assert months["values"].str.len().eq(months.month.dt.days_in_month).all()
    long = months[["placekey", "month", "values"]].explode("values")
    long["date"] = long.month + pd.to_timedelta(long.groupby(["placekey", "month"]).cumcount(), unit="D")
    source = long.pivot(index="date", columns="placekey", values="values").reindex(
        index=pd.to_datetime(days), columns=ids
    )
    np.testing.assert_array_equal(source.to_numpy(dtype=np.float32), visits)

    # The formal original-with-mask cohort and the descriptive figures use this
    # same static attribute snapshot and nine-group mapping from the pinned repo.
    fields = ["placekey", "location_name", "latitude", "longitude", "top_category",
              "sub_category", "naics_code", "brands", "safegraph_brand_ids",
              "parent_placekey", "open_hours"]
    attr_file = REPO / "Seattle_all_poi.csv"
    attrs = (pd.read_csv(attr_file, usecols=fields, low_memory=False)
             .drop_duplicates("placekey").set_index("placekey").loc[ids].reset_index())
    attrs["category9"] = attrs.top_category.map(category_mapping())
    assert attrs.placekey.tolist() == ids.tolist()
    assert attrs[["latitude", "longitude", "top_category", "naics_code", "category9"]].notna().all().all()
    meta = {
        "dataset": "SafeGraph Seattle approved original-with-mask cohort",
        "date_start": days.iloc[0], "date_end": days.iloc[-1], "days": len(days),
        "nodes": len(ids), "observed_poi_days": int(observed.sum()),
        "missing_poi_days": int((~observed).sum()), "missing_october_pois": 110,
        "source_sha256": {p.name: digest(p) for p in [visits_file, dates_file, ids_file, months_file, attr_file]},
        "repository_commit": COMMIT,
    }
    buf = io.BytesIO()
    np.save(buf, visits, allow_pickle=False)
    bundle = HERE / "approved_panel.zip"
    entries = {
        "visits.npy": buf.getvalue(),
        "dates.csv": days.to_frame().to_csv(index=False).encode(),
        "poi_attributes.csv": attrs.to_csv(index=False).encode(),
        "dataset_manifest.json": json.dumps(meta, indent=2).encode(),
    }
    with ZipFile(bundle, "w") as out:
        for name, content in entries.items():
            out.writestr(ZipInfo(name), content, compress_type=ZIP_DEFLATED, compresslevel=6)
    print(json.dumps({**meta, "bundle_sha256": digest(bundle), "bundle_bytes": bundle.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
