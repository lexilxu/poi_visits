"""Shared attribution-regime assignment and representative-POI export."""
import numpy as np
import pandas as pd

ATTR_GROUPS = ("History", "Location", "Identity", "Parent", "Others")


def export_attribution_regimes(attrs, shares, mae, out):
    # Compare each feature group with its own distribution across POIs.
    centered = shares - shares.mean(axis=0)
    spread = shares.std(axis=0)
    scores = np.divide(centered, spread, out=np.zeros_like(centered), where=spread > 0)
    labels = np.array([f"{group}-driven" for group in ATTR_GROUPS])
    winner = scores.argmax(axis=1)
    rows = []
    for k, label in enumerate(labels):
        members = np.flatnonzero(winner == k)
        centroid = shares[members].mean(0) if members.size else np.full(len(labels), np.nan)
        distance = np.linalg.norm(shares[members] - centroid, axis=1)
        reps = members[np.argsort(distance, kind="stable")[:3]]
        rows.append({"display_label": label, "n_pois": len(members),
                     "mae": float(mae[members].mean()) if members.size else None,
                     "dominant_source": ATTR_GROUPS[k],
                     "representative_pois": "; ".join(attrs.location_name.iloc[reps].astype(str)),
                     **dict(zip(ATTR_GROUPS, centroid))})
    table = pd.DataFrame(rows)
    table.to_csv(out/"04_attribution_clusters.csv", index=False)
    table[["display_label", *ATTR_GROUPS]].to_csv(out/"04_attribution_fingerprint.csv", index=False)
    assignments = pd.DataFrame({"placekey": attrs.placekey, "cluster": labels[winner], "mae": mae,
                                **dict(zip(ATTR_GROUPS, shares.T)),
                                **{f"{group}_z": score for group, score in zip(ATTR_GROUPS, scores.T)}})
    assignments.to_csv(out/"attribution_cluster_assignments.csv", index=False)
    category = pd.crosstab(attrs.category9, assignments.cluster).reindex(columns=labels, fill_value=0)
    category.insert(0, "n_pois", category.sum(axis=1))
    category[labels] = category[labels].div(category.n_pois, axis=0)
    category["mae"] = assignments.groupby(attrs.category9).mae.mean()
    category.to_csv(out/"category_cluster_distribution.csv")
