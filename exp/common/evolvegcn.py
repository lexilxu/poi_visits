"""Batched EvolveGCN-H backbone, following the vendored AAAI 2020 equations.

Parameter names inside each layer match external/EvolveGCN/egcn_h.py for
numerical forward/backward comparison. The forecast head is a linear mapping
from the final node embeddings to H visits. No node-state GRU or skip branch.
"""
import math
import torch
from torch import nn


class MatrixGate(nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.W=nn.Parameter(torch.empty(rows,rows))
        self.U=nn.Parameter(torch.empty(rows,rows))
        self.bias=nn.Parameter(torch.zeros(rows,cols))
        self.activation=activation
        nn.init.uniform_(self.W,-1/math.sqrt(rows),1/math.sqrt(rows))
        nn.init.uniform_(self.U,-1/math.sqrt(rows),1/math.sqrt(rows))

    def forward(self,x,hidden):
        return self.activation(self.W@x+self.U@hidden+self.bias)


class TopK(nn.Module):
    def __init__(self,features,k):
        super().__init__()
        self.scorer=nn.Parameter(torch.empty(features,1))
        nn.init.uniform_(self.scorer,-1/math.sqrt(features),1/math.sqrt(features))
        self.k=k

    def forward(self,nodes):
        scores=(nodes@self.scorer)/self.scorer.norm()
        indices=scores.squeeze(-1).topk(min(self.k,nodes.shape[1]),dim=1).indices
        if indices.shape[1]<self.k:
            indices=torch.cat([indices,indices[:,-1:].expand(-1,self.k-indices.shape[1])],dim=1)
        selected=nodes.gather(1,indices[:,:,None].expand(-1,-1,nodes.shape[-1]))
        return (selected*torch.tanh(scores.gather(1,indices[:,:,None]))).transpose(1,2)


class WeightEvolution(nn.Module):
    def __init__(self,rows,cols):
        super().__init__()
        self.update=MatrixGate(rows,cols,torch.sigmoid)
        self.reset=MatrixGate(rows,cols,torch.sigmoid)
        self.htilda=MatrixGate(rows,cols,torch.tanh)
        self.choose_topk=TopK(rows,cols)

    def forward(self,weights,nodes):
        summary=self.choose_topk(nodes)
        update=self.update(summary,weights)
        reset=self.reset(summary,weights)
        candidate=self.htilda(summary,reset*weights)
        return (1-update)*weights+update*candidate


class EvolvingLayer(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.evolve_weights=WeightEvolution(in_features,out_features)
        self.activation=nn.RReLU()
        self.GCN_init_weights=nn.Parameter(torch.empty(in_features,out_features))
        nn.init.uniform_(self.GCN_init_weights,-1/math.sqrt(out_features),1/math.sqrt(out_features))

    def forward(self,sequence,graph,trace=False):
        weights=self.GCN_init_weights.expand(sequence.shape[0],-1,-1)
        output=[]; evolution=[]
        for nodes in sequence.unbind(1):
            weights=self.evolve_weights(weights,nodes)
            output.append(self.activation(graph(nodes@weights)))
            if trace:evolution.append(weights.detach().clone())
        result=torch.stack(output,dim=1)
        return (result,evolution) if trace else result


class EvolveGCNForecast(nn.Module):
    def __init__(self,horizon,graph,static=None,hidden=32):
        super().__init__()
        self.graph=graph
        self.register_buffer('static',None if static is None else torch.as_tensor(static,dtype=torch.float32))
        sizes=[] if static is None else [int(static[:,i].max())+1 for i in range(4,static.shape[1])]
        self.embeddings=nn.ModuleList(nn.Embedding(n,8) for n in sizes)
        width=1+(4+8*len(sizes) if sizes else 0)
        self.layers=nn.ModuleList([EvolvingLayer(width,hidden),EvolvingLayer(hidden,hidden)])
        self.decoder=nn.Linear(hidden,horizon)

    def encode_static(self):
        return torch.cat([self.static[:,:4]]+[
            layer(self.static[:,4+i].long()) for i,layer in enumerate(self.embeddings)],dim=1)

    def forward(self,own):
        sequence=own[...,None]
        if self.static is not None:
            features=self.encode_static()[None,None].expand(own.shape[0],own.shape[1],-1,-1)
            sequence=torch.cat([sequence,features],dim=-1)
        for layer in self.layers:sequence=layer(sequence,self.graph)
        return self.decoder(sequence[:,-1]).transpose(1,2)


class IdentityGraph:
    def __call__(self,x):return x


class GroupGraph:
    """Exact original effective graph: 0.5 I + 0.5 within-group average."""
    def __init__(self,codes,device):
        self.codes=torch.as_tensor(codes,dtype=torch.long,device=device)
        self.sizes=torch.bincount(self.codes).to(torch.float32)

    def __call__(self,x):
        sums=x.new_zeros((x.shape[0],len(self.sizes),x.shape[-1]))
        sums.index_add_(1,self.codes,x)
        means=(sums/self.sizes[None,:,None])[:,self.codes]
        return (x+means)*.5


class SparseGraph:
    def __init__(self,matrix,device):
        matrix=matrix.tocsr();matrix.sort_indices()
        self.kernel=torch.sparse_csr_tensor(matrix.indptr,matrix.indices,matrix.data,
                                            size=matrix.shape,device=device,dtype=torch.float32)

    def __call__(self,x):
        b,n,c=x.shape
        flat=x.permute(1,0,2).reshape(n,b*c)
        with torch.autocast(x.device.type,enabled=False):
            product=torch.sparse.mm(self.kernel,flat.to(self.kernel.dtype))
        return product.reshape(n,b,c).permute(1,0,2)
