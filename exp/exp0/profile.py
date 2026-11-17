"""Check the approved experiment period and the identified industry cohort."""
import ast
from exp.exp0.categories import CATEGORY_MAP
from exp.common.paths import DATA_ROOT, RESULTS_ROOT
import json
from pathlib import Path

import numpy as np
import pandas as pd

from argparse import ArgumentParser
from exp.common.paths import DATA_ROOT, RESULTS_ROOT, REPO_ROOT


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=DATA_ROOT/'raw')
    parser.add_argument('--attributes', type=Path, default=DATA_ROOT/'Seattle_all_poi.csv')
    parser.add_argument('--out', type=Path, default=DATA_ROOT/'prepared')
    args = parser.parse_args()

    ROOT = args.raw_root
    OUT = args.out

    raw = pd.read_csv(ROOT / "paper_period_audit/Seattle_2020_09_to_12.csv", low_memory=False)
    raw = raw.loc[raw["date_range_start"].str[:7].isin(["2020-10", "2020-11", "2020-12"])].copy()
    monthly = raw[["placekey", "location_name", "date_range_start", "visits_by_day"]].copy()
    monthly["month"] = pd.to_datetime(monthly["date_range_start"].str.split("T").str[0])
    monthly["visits_by_day"] = monthly["visits_by_day"].apply(ast.literal_eval)
    monthly["num_daily_values"] = monthly["visits_by_day"].str.len()
    expanded = monthly.explode("visits_by_day").copy()
    expanded["day_offset"] = expanded.groupby(["placekey", "month"]).cumcount()
    expanded["date"] = expanded["month"] + pd.to_timedelta(expanded["day_offset"], unit="D")
    expanded["visits"] = pd.to_numeric(expanded["visits_by_day"], errors="raise")
    daily = expanded.groupby(["placekey", "location_name", "date"], as_index=False)["visits"].sum()
    matrix = daily.pivot(index="date", columns="placekey", values="visits").sort_index().sort_index(axis=1)
    matrix = matrix.loc["2020-10-01":"2020-12-31"]
    complete = matrix.dropna(axis=1, how="any")

    old_attrs = pd.read_csv(args.attributes, low_memory=False).drop_duplicates("placekey").set_index("placekey")
    main_meta = pd.read_csv(ROOT / "Seattle/processed/poi_meta.csv")
    aligned_main = old_attrs.reindex(main_meta["placekey"])
    known = aligned_main["top_category"].notna() & aligned_main["naics_code"].notna()
    identified_ids = aligned_main.index[known]
    identified_matrix = matrix.reindex(columns=identified_ids)
    missing_ids = identified_matrix.columns[identified_matrix.isna().any()]

    category_map = CATEGORY_MAP
    complete_attrs = old_attrs.reindex(complete.columns)
    mapped = complete_attrs["top_category"].map(category_map)
    report = {
        "approved_dates": {"start": "2020-10-01", "end": "2020-12-31", "days": len(matrix)},
        "monthly_rows": len(raw),
        "monthly_columns": list(raw.columns),
        "attribute_columns": list(old_attrs.columns),
        "duplicate_poi_month_records": int(monthly.duplicated(["placekey", "month"]).sum()),
        "duplicate_poi_day_names": int(daily.duplicated(["placekey", "date"]).sum()),
        "monthly_day_lengths": monthly.groupby("month")["num_daily_values"].agg(["min", "max"]).astype(int).to_dict(),
        "complete_observation_pois": complete.shape[1],
        "complete_pois_with_industry_labels": int((complete_attrs["top_category"].notna() & complete_attrs["naics_code"].notna()).sum()),
        "complete_pois_with_nine_group_mapping": int(mapped.notna().sum()),
        "identified_main_industry_cohort": len(identified_ids),
        "identified_cohort_complete_in_approved_period": int(identified_matrix.notna().all().sum()),
        "identified_cohort_missing_observations_pois": len(missing_ids),
        "identified_cohort_missing_poi_days": int(identified_matrix.isna().sum().sum()),
        "identified_cohort_missing_by_date": {d.strftime("%Y-%m-%d"): int(v) for d, v in identified_matrix.isna().sum(axis=1).items() if v},
        "identified_cohort_missing_placekeys": missing_ids.tolist(),
        "candidate_industry_category_counts": mapped.value_counts().to_dict(),
    }
    report["monthly_day_lengths"] = {k: {pd.Timestamp(m).strftime("%Y-%m-%d"): v for m, v in values.items()} for k, values in report["monthly_day_lengths"].items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cohort_profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"placekey": identified_ids}).to_csv(OUT / "identified_9884_pois.csv", index=False)
    np.save(OUT / "approved_period_visits_with_missing.npy", identified_matrix.to_numpy(dtype=np.float32))
    pd.DataFrame({"date": matrix.index.strftime("%Y-%m-%d")}).to_csv(OUT / "approved_dates.csv", index=False)
    raw.loc[raw['placekey'].isin(identified_ids)].to_csv(
        OUT / 'approved_period_source.csv.gz', index=False,
        compression={'method': 'gzip', 'mtime': 0})
    print(json.dumps({k: v for k, v in report.items() if k not in {"monthly_columns", "attribute_columns", "identified_cohort_missing_placekeys", "identified_cohort_missing_by_date", "candidate_industry_category_counts"}}, ensure_ascii=False, indent=2), flush=True)
    print("monthly_columns", list(raw.columns), flush=True)
    print("attribute_columns", list(old_attrs.columns), flush=True)


if __name__ == '__main__':
    main()
