"""Reuse identical full-input interventions with the fixed eight-world arithmetic."""
import numpy as np
import torch
from exp.common.evolvegcn_io import world_forward


def world_key(peers,donors=None):
    peers=np.asarray(peers,dtype=np.int64)
    assert len(np.unique(peers))==len(peers)
    if not len(peers):return ('base',)
    if donors is None:return ('zero',*sorted(peers.tolist()))
    donors=np.asarray(donors,dtype=np.int64);assert len(peers)==len(donors)
    return ('replace',*sorted(zip(peers.tolist(),donors.tolist())))


def targeted_worlds_reused(loaded,history,targets,peer_sets,donor_sets=None,batch_size=8,return_baseline=False):
    if batch_size!=8:raise ValueError('Use the verified eight-world physical batch')
    targets=np.asarray(targets,dtype=np.int64);groups={}
    for i,target in enumerate(targets):
        peers=np.asarray(peer_sets[i],dtype=np.int64)
        if target in peers:raise ValueError('Neighbor intervention includes its target')
        key=world_key(peers,None if donor_sets is None else donor_sets[i])
        groups.setdefault(key,[]).append(i)
    base=torch.as_tensor((history-loaded['mean'])/loaded['std'],device='cuda',dtype=torch.float32)
    with torch.no_grad():
        unperturbed=world_forward(loaded,base[None].expand(8,-1,-1).clone())
        for i in range(1,8):np.testing.assert_array_equal(unperturbed[i],unperturbed[0])
        paired=unperturbed[0][:,targets].copy();after=np.empty_like(paired)
        if ('base',) in groups:
            rows=groups.pop(('base',));after[:,rows]=paired[:,rows]
        keys=list(groups)
        for start in range(0,len(keys),8):
            selected=keys[start:start+8];worlds=base[None].expand(len(selected),-1,-1).clone()
            for row,key in enumerate(selected):
                if key[0]=='zero':worlds[row,:,np.asarray(key[1:],dtype=np.int64)]=(0-loaded['mean'])/loaded['std']
                else:
                    peers,donors=map(np.asarray,zip(*key[1:]));worlds[row,:,peers]=base[:,donors]
            prediction=world_forward(loaded,worlds)
            for row,key in enumerate(selected):
                ids=groups[key];after[:,ids]=prediction[row][:,targets[ids]]
    return (after,paired) if return_baseline else after
