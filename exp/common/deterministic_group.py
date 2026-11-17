"""Diagnostic deterministic summation for the fitted clique graph operator."""
import numpy as np
import torch


class DeterministicGroupGraph:
    def __init__(self, original):
        codes=original.codes.detach().cpu().numpy()
        self.codes=original.codes
        self.order=torch.as_tensor(np.argsort(codes,kind='stable'),device=self.codes.device)
        self.lengths=torch.bincount(self.codes)

    def __call__(self,x):
        with torch.autocast(x.device.type,enabled=False):
            ordered=x[:,self.order].permute(1,0,2).contiguous().double()
            sums=torch.segment_reduce(ordered,reduce='sum',lengths=self.lengths,axis=0)
            means=(sums/self.lengths[:,None,None]).permute(1,0,2)[:,self.codes]
            return (x+means.to(x.dtype))*.5
