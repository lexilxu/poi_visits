"""Fully observed forecast-horizon boundaries for the planned neural reruns."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Protocol:
    T: int = 14
    H: int = 7
    train_end: int = 50
    validation_end: int = 64

    def windows(self, raw):
        origins = np.arange(self.T,len(raw)-self.H+1)
        target = origins[:,None]+np.arange(self.H)
        history = origins[:,None]+np.arange(-self.T,0)
        end = target[:,-1]
        train = end < self.train_end
        val = (origins >= self.train_end) & (end < self.validation_end)
        test = origins >= self.validation_end
        keep = train | val | test
        split = np.where(train,'train',np.where(val,'validation','test'))[keep]
        origins,history,target = origins[keep],history[keep],target[keep]
        mask = np.isfinite(raw[target])
        for name in ('train','validation','test'):
            rows = split == name
            if not rows.any() or not mask[rows].any((0,2)).all():
                raise ValueError(f'{name}: missing forecast origins or entire forecast leads')
        return origins,history,target,split,mask


def feature_mask(variant):
    # Encoded order: four numeric inputs, then category, NAICS2, NAICS4,
    # brand, parent, subcategory, each with eight embedding dimensions.
    groups = {'location':[0,1], 'identity':list(range(4,36))+list(range(44,52)),
              'parent':list(range(36,44)), 'others':[2,3],
              'parent_identity':list(range(37,44)), 'parent_presence':[36]}
    semantic = {'category':list(range(4,12)), 'naics2':list(range(12,20)),
                'naics4':list(range(20,28)), 'subcategory':list(range(44,52))}
    mask = np.ones(52,dtype=np.float32)
    if variant == 'full':
        return mask
    if variant == 'history_only':
        return mask*0
    if variant.startswith('without_') and variant[8:] in groups:
        mask[groups[variant[8:]]] = 0
    elif variant.startswith('only_') and variant[5:] in semantic:
        for name, indices in semantic.items():
            if name != variant[5:]:
                mask[indices] = 0
    elif variant.startswith('without_') and variant[8:] in semantic:
        mask[semantic[variant[8:]]] = 0
    else:
        raise ValueError(f'Unknown feature variant {variant}')
    return mask
