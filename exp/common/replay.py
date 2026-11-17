"""High-precision evaluation of the frozen graph operators on AutoDL4."""
import numpy as np
import torch
from scipy import sparse


def sparse_operator(adjacency,device):
    matrix=(adjacency+sparse.eye(adjacency.shape[0],dtype=np.float32,format='csr')).tocsr()
    matrix=(sparse.diags(1/np.asarray(matrix.sum(1)).ravel())@matrix).tocsr()
    matrix.sort_indices()
    # Preserve the archived FP32 weights and accumulate their products in FP64.
    values=matrix.data.astype(np.float32).astype(np.float64)
    kernel=torch.sparse_csr_tensor(matrix.indptr,matrix.indices,values,
                                  size=matrix.shape,dtype=torch.float64,device=device)
    def multiply(x):
        b,c,n,steps=x.shape
        flat=x.to(torch.float64).permute(2,0,1,3).reshape(n,-1)
        result=torch.sparse.mm(kernel,flat).to(x.dtype)
        return result.reshape(n,b,c,steps).permute(1,2,0,3).contiguous()
    return multiply


def clique_operator(group,device):
    codes=torch.as_tensor(group,device=device)
    members=[torch.as_tensor(np.flatnonzero(group==i),device=device) for i in range(int(group.max())+1)]
    def multiply(x):
        values=x.to(torch.float64)
        means=torch.stack([values[:,:,ids,:].mean(2) for ids in members],dim=2)
        return ((values+means[:,:,codes,:])*.5).to(x.dtype)
    return multiply


def normalize_with_self(adjacency):
    a=adjacency.coalesce();n=a.shape[0]
    diagonal=torch.arange(n,device=a.device)
    indices=torch.cat([a.indices(),torch.stack([diagonal,diagonal])],dim=1)
    values=torch.cat([a.values(),torch.ones(n,device=a.device,dtype=a.dtype)])
    combined=torch.sparse_coo_tensor(indices,values,(n,n)).coalesce()
    row=combined.indices()[0]
    lengths=torch.bincount(row,minlength=n)
    degree=torch.segment_reduce(combined.values().double(),'sum',lengths=lengths)
    weights=(combined.values().double()/degree[row]).to(a.dtype).double()
    return torch.sparse_coo_tensor(combined.indices(),weights,(n,n)).coalesce().to_sparse_csr()


def adaptive_forward(self,x,adjacency):
    transition=normalize_with_self(adjacency)
    b,c,n,steps=x.shape
    h=x;levels=[h]
    for _ in range(self.depth):
        matrix=h.double().permute(2,0,1,3).reshape(n,-1)
        propagated=torch.sparse.mm(transition,matrix).to(x.dtype)
        propagated=propagated.reshape(n,b,c,steps).permute(1,2,0,3).contiguous()
        h=self.alpha*x+(1-self.alpha)*propagated
        levels.append(h)
    return self.mlp(torch.cat(levels,dim=1))


def validate(device):
    x=torch.arange(2*3*5*4,device=device,dtype=torch.float32).reshape(2,3,5,4)/17
    a=sparse.csr_matrix(np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]],dtype=np.float32))
    m=a+sparse.eye(5,dtype=np.float32,format='csr');m=sparse.diags(1/np.asarray(m.sum(1)).ravel())@m
    weights=torch.as_tensor(m.toarray().astype(np.float32),device=device).double()
    expected=torch.einsum('nm,bcmt->bcnt',weights,x.double()).float()
    operation=sparse_operator(a,device)
    actual=operation(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(actual,operation(x))
    group=np.array([0,1,0,1,1]);op=clique_operator(group,device)
    dense=torch.as_tensor((group[:,None]==group[None,:])/np.bincount(group)[group,None],device=device).double()
    reference=((x.double()+torch.einsum('nm,bcmt->bcnt',dense,x.double()))*.5).float()
    torch.testing.assert_close(op(x),reference,rtol=1e-7,atol=1e-7)
    assert torch.equal(op(x),op(x))


def install():
    import exp.exp1.train_mtgnn as train_mtgnn
    import exp.common.adaptive_mtgnn as adaptive_mtgnn
    train_mtgnn.sparse_operator=sparse_operator
    train_mtgnn.clique_operator=clique_operator
    adaptive_mtgnn.SparseMixProp.forward=adaptive_forward
