"""Check sparse learned-graph values and gradients against the dense operators."""
from pathlib import Path
import copy
import sys
import unittest
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'external/MTGNN'))
from layer import graph_constructor,mixprop
from exp.common.adaptive_mtgnn import SparseGraphConstructor, SparseMixProp


class AdaptiveTests(unittest.TestCase):
    def test_dense_sparse_values_and_gradients(self):
        torch.manual_seed(18)
        dense=graph_constructor(19,4,6,torch.device('cpu')).double()
        sparse=SparseGraphConstructor(copy.deepcopy(dense),block_size=5)
        idx=torch.arange(19);noise=torch.rand(19,19,dtype=torch.float64)
        first=torch.tanh(dense.alpha*dense.lin1(dense.emb1(idx)))
        second=torch.tanh(dense.alpha*dense.lin2(dense.emb2(idx)))
        adjacency=F.relu(torch.tanh(dense.alpha*(first@second.T-second@first.T)))
        mask=torch.zeros_like(adjacency).scatter(1,(adjacency+.01*noise).topk(4,dim=1).indices,1.)
        adjacency=adjacency*mask
        sparse_adj=sparse(idx,noise)
        torch.testing.assert_close(adjacency,sparse_adj.to_dense(),atol=1e-12,rtol=1e-12)
        dense_conv=mixprop(2,3,2,.1,.05).double()
        sparse_conv=SparseMixProp(copy.deepcopy(dense_conv))
        x=torch.randn(2,2,19,3,dtype=torch.float64,requires_grad=True)
        z=x.detach().clone().requires_grad_(True)
        a=dense_conv(x,adjacency);b=sparse_conv(z,sparse_adj)
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        a.square().sum().backward();b.square().sum().backward()
        torch.testing.assert_close(x.grad,z.grad,atol=1e-11,rtol=1e-10)
        for p,q in zip(dense.parameters(),sparse.original.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-9)

    def test_evaluation_graph_is_repeatable(self):
        model=SparseGraphConstructor(graph_constructor(23,4,6,torch.device('cpu')),block_size=5).eval()
        a=model(torch.arange(23)).to_dense();torch.rand(100)
        b=model(torch.arange(23)).to_dense()
        torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
