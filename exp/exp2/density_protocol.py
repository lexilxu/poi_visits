"""Original density rankings with fixed 57-day observed-label denominators."""
from hashlib import sha256
import re
import numpy as np


def selected_indices(raw,label):
    match=re.fullmatch(r'(Dense|Recent-dense|Persist-dense)(10|25|50|75)',label)
    if match is None:raise ValueError(label)
    group,percentage=match.group(1),int(match.group(2))
    train=raw[:57];observed=np.isfinite(train)
    ratio=(train>0).sum(0)/observed.sum(0)
    recent=(train[-14:]>0).sum(0)/observed[-14:].sum(0)
    eligible=(train[:-1]>0)&observed[1:]
    persistence=(eligible&(train[1:]>0)).sum(0)/np.maximum(eligible.sum(0),1)
    rank={'Dense':ratio,'Recent-dense':recent,'Persist-dense':ratio*persistence}[group]
    count=int(np.ceil((100-percentage)*raw.shape[1]/100))
    ids=np.sort(np.argsort(-rank,kind='stable')[:count])
    return ids,dict(label=label,n_pois=len(ids),nonzero_ratio=float(ratio[ids].mean()),
                    excluded_percent=percentage,train_end=57,
                    indices_sha256=sha256(ids.astype('<i8').tobytes()).hexdigest())
