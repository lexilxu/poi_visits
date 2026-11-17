"""Keep one official 2020 static record per candidate POI with source provenance."""
import concurrent.futures
import json
from pathlib import Path

import pandas as pd

from argparse import ArgumentParser
from exp.common.paths import DATA_ROOT, RESULTS_ROOT, REPO_ROOT


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=DATA_ROOT/'raw')
    parser.add_argument('--out', type=Path, default=DATA_ROOT/'prepared')
    args = parser.parse_args()

    ROOT = args.raw_root
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    identified = set(pd.read_csv(OUT / "identified_9884_pois.csv")["placekey"])
    raw = pd.read_csv(ROOT / "paper_period_audit/Seattle_2020_09_to_12.csv", usecols=["placekey", "date_range_start"])
    period = raw.loc[raw["date_range_start"].str[:7].isin(["2020-10", "2020-11", "2020-12"])]
    counts = period.groupby("placekey")["date_range_start"].nunique()
    ids = identified | set(counts.index[counts.eq(3)])
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    items = [f for f in manifest["files"] if f["kind"] == "POIs" and f["month"] in ["2020-11", "2020-12"]]

    def extract(item):
        frames = []
        for chunk in pd.read_csv(ROOT / item["relative_path"], compression="gzip", chunksize=150_000, low_memory=False):
            selected = chunk.loc[chunk["placekey"].isin(ids)].copy()
            if len(selected):
                selected["attribute_source_month"] = item["month"]
                selected["attribute_source_file"] = item["relative_path"]
                frames.append(selected)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        print(json.dumps({"source": item["relative_path"], "selected_rows": len(result)}, ensure_ascii=False), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        frames = list(executor.map(extract, items))
    all_attrs = pd.concat(frames, ignore_index=True)
    all_attrs = all_attrs.sort_values(["attribute_source_month", "attribute_source_file", "placekey"], kind="stable")
    selected = all_attrs.drop_duplicates("placekey", keep="first")
    selected.to_csv(OUT / "candidate_attributes_2020.csv", index=False)
    selected_index = selected.set_index("placekey").reindex(sorted(identified))
    report = {
        "candidate_pois": len(ids),
        "selected_attribute_rows": len(selected),
        "selected_source_month_counts": selected["attribute_source_month"].value_counts().to_dict(),
        "identified_9884_missing_attribute_records": int(selected_index["attribute_source_month"].isna().sum()),
        "identified_9884_missing_top_category": int(selected_index["top_category"].isna().sum()),
        "identified_9884_missing_naics": int(selected_index["naics_code"].isna().sum()),
        "selection_rule": "First available official Core POI snapshot in 2020, by month and source file; original placekeys retained",
    }
    (OUT / "attribute_extraction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
