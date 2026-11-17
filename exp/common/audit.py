"""Audit the paper's archived protocol without importing a training framework.

The bundle is a local, licensed input; only aggregate audit outputs belong in git.
Run from the repository root with --bundle and --out. No fitted result is changed.
"""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

SETTINGS = ((14, 7), (28, 7), (28, 14), (28, 28))
TRAIN_END, VAL_END = 57, 64
EXPECTED_BUNDLE_SHA256 = "12d6414990eeb5703b1179152b5993518dcfa68361e39e09ae621919950695ec"


def load_panel(bundle):
    digest = sha256(bundle.read_bytes()).hexdigest()
    if digest != EXPECTED_BUNDLE_SHA256:
        raise ValueError(f"Unrecognized approved bundle: {digest}")
    with ZipFile(bundle) as z:
        raw = np.load(BytesIO(z.read("visits.npy")), allow_pickle=False)
        dates = pd.read_csv(BytesIO(z.read("dates.csv"))).date.to_numpy()
        attrs = pd.read_csv(BytesIO(z.read("poi_attributes.csv")))
    assert raw.shape == (92, 9884)
    assert attrs.placekey.is_unique
    assert np.isfinite(raw).sum() == 905918
    assert np.array_equal(dates, pd.date_range("2020-10-01", "2020-12-31").strftime("%Y-%m-%d"))
    return raw, dates, attrs


def window_arrays(raw, t, h):
    origins = np.arange(t, len(raw) - h + 1)
    history = origins[:, None] + np.arange(-t, 0)
    target = origins[:, None] + np.arange(h)
    split = np.where(origins < TRAIN_END, "train", np.where(origins < VAL_END, "validation", "test"))
    mask = np.isfinite(raw[target])
    mask[split == "train"] &= (target[split == "train", :, None] < TRAIN_END)
    mask[split == "validation"] &= (target[split == "validation", :, None] < VAL_END)
    return origins, history, target, split, mask


def low_history(raw):
    count = np.isfinite(raw).sum(axis=0)
    mean = np.nansum(raw, axis=0) / count
    zero = (raw == 0).sum(axis=0) / count
    q = np.quantile(mean[mean > 0], .25)
    return (mean > 0) & (mean <= q) & (zero >= .7), float(q)


def canonical_partition(values):
    return pd.factorize(np.asarray(values), sort=False)[0]


def graph_audit(attrs):
    naics = pd.to_numeric(attrs.naics_code, errors="raise").astype("int64").astype(str)
    codes = {"NAICS-2": canonical_partition(naics.str[:2]),
             "NAICS-4": canonical_partition(naics.str[:4]),
             "Top category": canonical_partition(attrs.top_category)}
    rows = []
    for name, group in codes.items():
        degree = np.bincount(group)[group] - 1
        rows.append({"adjacency": name, "groups": int(group.max()+1),
                     "directed_edges_excluding_self": int(degree.sum()),
                     "mean_degree": float(degree.mean()), "median_degree": float(np.median(degree)),
                     "isolated_pois": int((degree == 0).sum()),
                     "partition_sha256": sha256(group.astype("<i8").tobytes()).hexdigest()})
    exact = np.array_equal(codes["NAICS-4"], codes["Top category"])
    return {"statistics": rows, "naics4_top_category_identical_partition": bool(exact),
            "interpretation": ("Both labels induce exactly the same clique adjacency on this cohort. "
                               "Identical model outputs under identical seeds and hyperparameters are expected."
                               if exact else "Partitions differ; inspect model predictions and table exports.")}


def run(bundle, out):
    raw, dates, attrs = load_panel(bundle)
    out.mkdir(parents=True, exist_ok=True)
    fixed, fixed_q = low_history(raw[:TRAIN_END])
    windows, summaries = [], []
    for t, h in SETTINGS:
        origins, hx, targets, splits, valid = window_arrays(raw, t, h)
        per_lead = {}
        target_sets = {}
        for name in ("train", "validation", "test"):
            rows = splits == name
            per_lead[name] = valid[rows].any(axis=2).sum(axis=0).astype(int).tolist()
            target_sets[name] = set(targets[rows][valid[rows].any(axis=2)].tolist())
        assert not (target_sets["train"] & target_sets["validation"])
        assert not (target_sets["train"] & target_sets["test"])
        assert not (target_sets["validation"] & target_sets["test"])
        assert np.all(hx[:, -1] < targets[:, 0])
        assert hx[splits == "train"].max() < TRAIN_END
        legacy, q = low_history(raw[hx[splits == "train"]].reshape(-1, raw.shape[1]))
        for b, origin in enumerate(origins):
            scored = targets[b][valid[b].any(axis=1)]
            windows.append({"T": t, "H": h, "split": splits[b], "origin_index": int(origin),
                            "input_start": dates[hx[b, 0]], "input_end": dates[hx[b, -1]],
                            "requested_target_start": dates[targets[b, 0]],
                            "requested_target_end": dates[targets[b, -1]],
                            "scored_target_start": dates[scored[0]], "scored_target_end": dates[scored[-1]],
                            "scored_leads": int(len(scored)), "observed_scored_events": int(valid[b].sum()),
                            "normalization_last_date": dates[hx[splits == "train"].max()],
                            "visit_graph_last_date": dates[TRAIN_END-1]})
        summaries.append({"T": t, "H": h,
                          "origins": {name: int((splits == name).sum()) for name in per_lead},
                          "observed_events": {name: int(valid[splits == name].sum()) for name in per_lead},
                          "scored_origins_by_lead": per_lead,
                          "unvalidated_leads": [i+1 for i, n in enumerate(per_lead["validation"]) if n == 0],
                          "target_dates_disjoint": True,
                          "legacy_low_history_pois": int(legacy.sum()), "legacy_cutoff": q,
                          "fixed_calendar_low_history_pois": int(fixed.sum()), "fixed_calendar_cutoff": fixed_q,
                          "cohort_membership_changes_vs_calendar": int(np.count_nonzero(legacy != fixed))})
    coords = attrs[["latitude", "longitude"]]
    parent_counts = attrs.parent_placekey.value_counts()
    parent_encoding = {
        "observed_pois": int(attrs.parent_placekey.notna().sum()),
        "distinct_recorded_labels": int(len(parent_counts)),
        "frequency_threshold": 20,
        "retained_labels": int((parent_counts > 20).sum()),
        "pooled_labels": int((parent_counts <= 20).sum()),
        "pooled_pois": int(parent_counts[parent_counts <= 20].sum()),
        "missing_pois": int(attrs.parent_placekey.isna().sum()),
    }
    report = {"bundle_sha256": EXPECTED_BUNDLE_SHA256, "panel_shape": list(raw.shape),
              "missing_poi_days": int(np.isnan(raw).sum()), "windows": summaries,
              "graphs": graph_audit(attrs),
              "parent_encoding": parent_encoding,
              "raw_coordinate_means": coords.mean().to_dict(),
              "raw_coordinate_standard_deviations": coords.std(ddof=0).to_dict(),
              "limitations": ["Disjoint scored labels do not validate leads absent from the validation period.",
                              "Overlapping forecast events are not independent observations.",
                              "This audit establishes data and code properties, not forecasting performance."]}
    (out / "protocol_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    pd.DataFrame(windows).to_csv(out / "window_manifest.csv", index=False)
    pd.DataFrame(report["graphs"]["statistics"]).to_csv(out / "graph_statistics.csv", index=False)
    print(json.dumps({"target_dates_disjoint": True,
                      "identical_naics4_top_category_graph": report["graphs"]["naics4_top_category_identical_partition"],
                      "fixed_calendar_low_history_pois": int(fixed.sum()),
                      "unvalidated_leads": {f'T{x["T"]}_H{x["H"]}': x["unvalidated_leads"] for x in summaries}}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.bundle, args.out)
