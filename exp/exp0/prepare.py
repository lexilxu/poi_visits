"""Build the final experiment inputs using the explicitly selected cohort policy."""
import argparse
import ast
from exp.exp0.categories import CATEGORY_MAP
from exp.common.paths import DATA_ROOT, RESULTS_ROOT
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = DATA_ROOT / 'raw'
WORK = DATA_ROOT / 'prepared'
REPO = DATA_ROOT
OUTPUT = DATA_ROOT / 'formal'
COMMIT = "3e5e14f82949b3e0d90e7fbb46d64d7ef48f23ba"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def category_mapping():
    return CATEGORY_MAP


def main():
    global ROOT, WORK, REPO, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["original-with-mask", "complete-original", "complete-reselect"], required=True)
    parser.add_argument('--raw-root', type=Path, default=ROOT)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--attributes', type=Path, default=REPO/'Seattle_all_poi.csv')
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    ROOT, WORK, OUTPUT = args.raw_root, args.work, args.out
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError("The staging directory already contains a dataset")

    raw = pd.read_csv(ROOT / "paper_period_audit/Seattle_2020_09_to_12.csv", low_memory=False)
    raw = raw.loc[raw["date_range_start"].str[:7].isin(["2020-10", "2020-11", "2020-12"])].copy()
    if raw.duplicated(["placekey", "date_range_start"]).any():
        raise ValueError("A source POI-month has multiple rows")
    frame = raw[["placekey", "location_name", "date_range_start", "visits_by_day"]].copy()
    frame["month"] = pd.to_datetime(frame["date_range_start"].str.split("T").str[0])
    frame["visits_by_day"] = frame["visits_by_day"].apply(ast.literal_eval)
    if not frame["visits_by_day"].str.len().eq(frame["month"].dt.days_in_month).all():
        raise ValueError("A monthly visits list differs from the calendar month length")
    expanded = frame.explode("visits_by_day").copy()
    expanded["date"] = expanded["month"] + pd.to_timedelta(expanded.groupby(["placekey", "month"]).cumcount(), unit="D")
    expanded["visits"] = pd.to_numeric(expanded["visits_by_day"], errors="raise")
    if expanded.duplicated(["placekey", "date"]).any():
        raise ValueError("A source POI-day has multiple rows")
    matrix = expanded.pivot(index="date", columns="placekey", values="visits").sort_index().sort_index(axis=1)
    matrix = matrix.loc["2020-10-01":"2020-12-31"]
    if not matrix.index.equals(pd.date_range("2020-10-01", "2020-12-31")):
        raise ValueError("The experiment requires all 92 calendar dates")
    complete_ids = set(matrix.columns[matrix.notna().all()])
    original_ids = pd.read_csv(WORK / "identified_9884_pois.csv")["placekey"].tolist()
    if len(original_ids) != 9884 or len(set(original_ids)) != 9884:
        raise ValueError("The identified industry cohort requires 9884 distinct POIs")
    old_attrs = pd.read_csv(args.attributes, low_memory=False).drop_duplicates("placekey").set_index("placekey")
    mapping = category_mapping()
    original_groups = old_attrs.reindex(original_ids)["top_category"].map(mapping)
    if original_groups.isna().any():
        raise ValueError("The identified original cohort requires defined category groups")

    selection = {"policy": args.mode, "original_identified_cohort_pois": len(original_ids)}
    if args.mode == "original-with-mask":
        ids = sorted(original_ids)
        attributes = old_attrs.reindex(ids).copy()
        groups = attributes["top_category"].map(mapping)
        selection["rule"] = "Original 9884 industry-labelled POIs; source gaps remain NaN and have a separate observation mask"
        selection["static_metadata_sources"] = "The main repository's official 2020-11, 2020-12, and 2021-01 Core POI snapshots"
    elif args.mode == "complete-original":
        ids = sorted(set(original_ids) & complete_ids)
        attributes = old_attrs.reindex(ids).copy()
        groups = attributes["top_category"].map(mapping)
        selection["rule"] = "Original identified cohort restricted to complete observed daily records for the approved period"
        selection["static_metadata_sources"] = "The main repository's official 2020-11, 2020-12, and 2021-01 Core POI snapshots"
    else:
        attrs_2020 = pd.read_csv(WORK / "candidate_attributes_2020.csv", low_memory=False).set_index("placekey")
        eligible = attrs_2020.loc[attrs_2020.index.isin(complete_ids)].copy()
        eligible["category9"] = eligible["top_category"].map(mapping)
        eligible = eligible.dropna(subset=["top_category", "naics_code", "category9", "latitude", "longitude"])
        rng = np.random.default_rng(42)
        picked = []
        targets = original_groups.value_counts().to_dict()
        primary_ids = set(original_ids)
        for group, target in sorted(targets.items()):
            pool = eligible.loc[eligible["category9"].eq(group)]
            primary = sorted(set(pool.index) & primary_ids)
            secondary = sorted(set(pool.index) - primary_ids)
            if len(primary) >= target:
                chosen = rng.choice(primary, size=target, replace=False).tolist() if len(primary) > target else primary
            else:
                needed = target - len(primary)
                if len(secondary) < needed:
                    raise ValueError(f"Insufficient complete candidates for {group}")
                chosen = primary + rng.choice(secondary, size=needed, replace=False).tolist()
            picked.extend(chosen)
        ids = sorted(picked)
        if len(ids) != 9884 or len(set(ids)) != 9884:
            raise ValueError("The reselected cohort requires 9884 unique POIs")
        attributes = eligible.reindex(ids).copy()
        groups = attributes["category9"]
        selection.update({"rule": "Complete 92-day observations and 2020 static attributes; retain eligible original POIs first, then sample within category groups",
                          "random_seed": 42, "category_targets": targets, "eligible_candidates": len(eligible),
                          "retained_original_pois": len(set(ids) & primary_ids),
                          "added_complete_pois": len(set(ids) - primary_ids),
                          "static_metadata_sources": "First available official 2020 Core POI snapshot by month and source file"})

    values = matrix.reindex(columns=ids).to_numpy(dtype=np.float32)
    observed = ~np.isnan(values)
    if (values[observed] < 0).any() or not np.equal(values[observed], np.floor(values[observed])).all():
        raise ValueError("Daily observed visits require nonnegative integer values")
    if args.mode != "original-with-mask" and not observed.all():
        raise ValueError("The selected complete cohort contains a missing observation")
    if not np.isfinite(values[observed]).all():
        raise ValueError("Observed source visits require finite values")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "visits.npy", values)
    if not observed.all():
        np.save(OUTPUT / "observed_mask.npy", observed)
    pd.DataFrame({"date": matrix.index.strftime("%Y-%m-%d")}).to_csv(OUTPUT / "dates.csv", index=False)
    pd.DataFrame({"placekey": ids, "location_name": attributes["location_name"].to_numpy()}).to_csv(OUTPUT / "poi_meta.csv", index=False)
    attribute_fields = ["location_name", "latitude", "longitude", "top_category", "sub_category", "naics_code", "brands", "safegraph_brand_ids", "parent_placekey", "open_hours"]
    if args.mode == "complete-reselect":
        attribute_fields += ["attribute_source_month"]
    attribute_table = attributes[attribute_fields].copy()
    attribute_table["category9"] = groups.to_numpy()
    attribute_table.reset_index().to_csv(OUTPUT / "poi_attributes.csv", index=False)
    selected_raw = raw.loc[raw["placekey"].isin(ids), ["placekey", "safegraph_place_id", "location_name", "city", "date_range_start", "date_range_end", "visits_by_day"]]
    selected_raw = selected_raw.sort_values(["placekey", "date_range_start"], kind="stable")
    selected_raw.to_csv(OUTPUT / "monthly_visits.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    reports = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in ["download_report.json", "paper_months_download_report.json"]]
    sources = [f for report in reports for f in report["files"]]
    selected_sources = []
    for source in sources:
        relevant = source["kind"] == "MonthlyPattern" and source["month"] in ["2020-10", "2020-11", "2020-12"]
        relevant |= source["kind"] == "POIs" and (args.mode != "complete-reselect" or source["month"] in ["2020-11", "2020-12"])
        if relevant:
            selected_sources.append({k: source[k] for k in ["relative_path", "kind", "month", "drive_file_id", "expected_bytes", "sha256", "gzip_crc_verified"]})
    info = {
        "dataset": "Seattle SafeGraph POI daily observed visits",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/lexilxu/poi_visits", "branch": "main", "commit": COMMIT,
        "date_start": "2020-10-01", "date_end": "2020-12-31", "days": 92, "pois": len(ids),
        "matrix_shape": list(values.shape), "dtype": str(values.dtype), "unit": "observed visits per POI per day",
        "axes": {"axis_0": "dates.csv row order", "axis_1": "poi_meta.csv and poi_attributes.csv row order"},
        "selection": selection, "monthly_source_records": len(selected_raw),
        "observed_poi_days": int(observed.sum()), "missing_poi_days": int((~observed).sum()),
        "pois_with_missing_observations": int((~observed).any(axis=0).sum()),
        "observed_zero_poi_days": int(((values == 0) & observed).sum()),
        "category_counts": dict(Counter(groups)),
        "missing_attributes": attribute_table.isna().sum().astype(int).to_dict(),
        "source_archives": selected_sources, "source_verification": "Original download size and full gzip CRC checked; SHA-256 recorded locally",
        "checks": {"all_92_calendar_days": True, "unique_poi_ids": True, "one_record_per_poi_month": True,
                   "one_value_per_observed_poi_day": True, "nonnegative_integer_observed_visits": True,
                   "complete_daily_observations": bool(observed.all()), "aligned_poi_order": True},
    }
    mode_description = {"original-with-mask": "原 9,884 个行业标签完整的 POI；缺失访问量保存为 NaN，observed_mask.npy 标记实际观测。",
                        "complete-original": "原行业标签完整的 POI 名单中，在确认日期内每天都有观测的 POI。",
                        "complete-reselect": "从确认日期内每天都有观测且具备 2020 年属性的 POI 中选定 9,884 个；优先保留原名单并维持九类 POI 的数量。"}[args.mode]
    (OUTPUT / "README.md").write_text(
        f"# Seattle 正式实验数据\n\n日期：2020-10-01 至 2020-12-31，共 92 天。POI 数：{len(ids):,}。\n\n"
        f"样本口径：{mode_description}\n\n"
        "visits.npy 的行对应 dates.csv，列对应 poi_meta.csv 和 poi_attributes.csv，单位为每天实际观测的访问次数。\n\n"
        "monthly_visits.csv.gz 保存入选 POI 的原始月度访问记录；poi_attributes.csv 保存坐标、行业、品牌、父 POI 和营业时间等实验属性。\n\n"
        f"静态属性来源：{selection['static_metadata_sources']}。\n\n"
        f"实际观测：{int(observed.sum()):,} 个 POI 日；缺失：{int((~observed).sum()):,} 个 POI 日。\n\n"
        "source_manifest.json 记录样本规则、来源分片、源文件校验值和最终输入文件校验值。\n\n"
        f"参考仓库 main 提交：{COMMIT}。研究日期与样本口径按本次确认生成。\n", encoding="utf-8")
    info["files"] = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(OUTPUT.iterdir()) if p.is_file()]
    (OUTPUT / "source_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: info[k] for k in ["date_start", "date_end", "days", "pois", "matrix_shape", "selection", "observed_poi_days", "missing_poi_days", "files"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
