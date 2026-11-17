"""Training-only density cohorts and provenance-checked alternative metadata."""
from hashlib import sha256
from pathlib import Path
import re
import numpy as np
import pandas as pd


def density_indices(raw,name,train_end=50):
    match=re.fullmatch(r'(Dense|Recent-dense|Persist-dense)(10|25|50|75)',name)
    if match is None:raise ValueError(name)
    kind,excluded=match.groups();values=raw[:train_end];observed=np.isfinite(values)
    ratio=(values>0).sum(0)/observed.sum(0)
    if kind=='Recent-dense':
        values=values[-14:];score=(values>0).sum(0)/np.isfinite(values).sum(0)
    elif kind=='Persist-dense':
        paired=np.isfinite(values[:-1]) & np.isfinite(values[1:])
        denominator=((values[:-1]>0)&paired).sum(0)
        numerator=((values[:-1]>0)&(values[1:]>0)&paired).sum(0)
        score=ratio*np.divide(numerator,denominator,out=np.zeros(len(ratio)),where=denominator>0)
    else:score=ratio
    keep=int(np.ceil((100-int(excluded))*raw.shape[1]/100))
    # Stable ranking gives an exact cohort size even at tied nonzero ratios.
    indices=np.sort(np.argsort(-score,kind='stable')[:keep])
    return indices,dict(name=name,POIs=keep,excluded_percent=int(excluded),
                        training_nonzero_ratio=float(ratio[indices].mean()),
                        selection_sha256=sha256(indices.astype('<i4').tobytes()).hexdigest(),
                        tie_rule='Original POI order, stable descending score rank')


def replace_attributes(attrs,path):
    replacement=pd.read_csv(path)
    np.testing.assert_array_equal(replacement.placekey,attrs.placekey)
    required=['latitude','longitude','naics_code','top_category','sub_category','brands','parent_placekey','open_hours']
    if not set(required)<=set(replacement):raise ValueError('Incomplete static metadata schema')
    return replacement,sha256(Path(path).read_bytes()).hexdigest()
