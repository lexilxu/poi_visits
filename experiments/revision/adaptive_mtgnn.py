"""Sparse evaluation of the pinned MTGNN learned top-k graph.

Candidate scores are computed in row blocks without retaining dense gradients.
The selected edge weights are recomputed with autograd. Top-k indices have no
derivative in the original dense implementation either. Inference uses a fixed
tie-noise seed, making validation and saved predictions repeatable.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SparseGraphConstructor(nn.Module):
    def __init__(self, original, block_size=256):
        super().__init__()
        self.original=original
        self.block_size=block_size

    def forward(self,idx,noise=None):
        gc=self.original
        if gc.static_feat is None:
            first,second=gc.emb1(idx),gc.emb2(idx)
        else:
            first=gc.static_feat[idx];second=first
        first=torch.tanh(gc.alpha*gc.lin1(first))
        second=torch.tanh(gc.alpha*gc.lin2(second))
        n=len(idx);k=min(gc.k,n)
        generator=None
        if not self.training:
            generator=torch.Generator(device=first.device).manual_seed(0)
        rows=[];columns=[]
        with torch.no_grad():
            for start in range(0,n,self.block_size):
                stop=min(n,start+self.block_size)
                score=F.relu(torch.tanh(gc.alpha*(first[start:stop]@second.T-second[start:stop]@first.T)))
                perturbation=(torch.rand(score.shape,device=score.device,dtype=score.dtype,generator=generator)
                              if noise is None else noise[start:stop])
                chosen=(score+.01*perturbation).topk(k,dim=1).indices
                rows.append(torch.arange(start,stop,device=idx.device)[:,None].expand_as(chosen).reshape(-1))
                columns.append(chosen.reshape(-1))
        row=torch.cat(rows);column=torch.cat(columns)
        score=(first[row]*second[column]).sum(-1)-(second[row]*first[column]).sum(-1)
        weights=F.relu(torch.tanh(gc.alpha*score))
        return torch.sparse_coo_tensor(torch.stack([row,column]),weights,(n,n)).coalesce()


def normalize_with_self(adjacency):
    adjacency=adjacency.coalesce()
    n=adjacency.shape[0]
    diagonal=torch.arange(n,device=adjacency.device)
    indices=torch.cat([adjacency.indices(),torch.stack([diagonal,diagonal])],dim=1)
    values=torch.cat([adjacency.values(),torch.ones(n,device=adjacency.device,dtype=adjacency.dtype)])
    combined=torch.sparse_coo_tensor(indices,values,(n,n)).coalesce()
    row=combined.indices()[0]
    degree=torch.zeros(n,device=values.device,dtype=values.dtype).scatter_add(0,row,combined.values())
    return torch.sparse_coo_tensor(combined.indices(),combined.values()/degree[row],(n,n)).coalesce()


class SparseMixProp(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.alpha=original.alpha;self.depth=original.gdep;self.mlp=original.mlp

    def forward(self,x,adjacency):
        transition=normalize_with_self(adjacency)
        b,c,n,steps=x.shape
        h=x;levels=[h]
        for _ in range(self.depth):
            matrix=h.permute(2,0,1,3).reshape(n,-1)
            propagated=torch.sparse.mm(transition,matrix)
            propagated=propagated.reshape(n,b,c,steps).permute(1,2,0,3).contiguous()
            h=self.alpha*x+(1-self.alpha)*propagated
            levels.append(h)
        return self.mlp(torch.cat(levels,dim=1))


def adapt(network):
    network.gc=SparseGraphConstructor(network.gc)
    for modules in (network.gconv1,network.gconv2):
        for index,original in enumerate(modules):
            modules[index]=SparseMixProp(original)
    return network
