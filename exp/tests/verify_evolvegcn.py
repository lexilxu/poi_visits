"""Run the CUDA equivalence gate against the vendored author implementation."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,hashlib,io,json,sys
import numpy as np
import torch
from exp.common.evolvegcn import EvolveGCNForecast, IdentityGraph, GroupGraph, SparseGraph, EvolvingLayer
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'external/EvolveGCN'))
from egcn_h import GRCU


def main(out):
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    torch.manual_seed(20261115)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');cases=[]
    factory=ast.parse((ROOT/'external/EvolveGCN/run_exp.py').read_text())
    calls=[node for node in ast.walk(factory) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Attribute) and node.func.attr=='EGCN'
           and isinstance(node.func.value,ast.Name) and node.func.value.id=='egcn_h']
    assert len(calls)==2
    for call in calls:
        activation=next(kw.value for kw in call.keywords if kw.arg=='activation')
        assert ast.unparse(activation)=='torch.nn.RReLU()'
    for width in [1,5,32,53]:
        hidden=8;nodes=17;steps=4;batch=2
        actual=EvolvingLayer(width,hidden).to(device).double().eval()
        reference=GRCU(SimpleNamespace(in_feats=width,out_feats=hidden,activation=torch.nn.RReLU())).to(device).double().eval()
        reference.load_state_dict(actual.state_dict(),strict=True)
        codes=np.arange(nodes)%4
        graph=GroupGraph(codes,device)
        dense=torch.eye(nodes,device=device,dtype=torch.float64)*.5
        for i in range(nodes):
            peers=np.flatnonzero(codes==codes[i]);dense[i,peers]+=.5/len(peers)
        x=torch.randn(batch,steps,nodes,width,device=device,dtype=torch.float64,requires_grad=True)
        z=x.detach().clone().requires_grad_()
        output,trace=actual(x,graph,trace=True)
        expected=torch.stack([torch.stack(reference([dense]*steps,list(item.unbind(0)),
                                      [torch.zeros(nodes,1,device=device,dtype=torch.float64)]*steps)) for item in z])
        torch.testing.assert_close(output,expected,rtol=1e-9,atol=1e-10)
        probe=torch.randn_like(output)
        (output*probe).sum().backward();(expected*probe).sum().backward()
        torch.testing.assert_close(x.grad,z.grad,rtol=1e-8,atol=1e-9)
        grad_error=0.
        for (key,a),(refkey,b) in zip(actual.named_parameters(),reference.named_parameters()):
            assert key==refkey
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-8,atol=1e-9)
            grad_error=max(grad_error,float((a.grad-b.grad).abs().max()))
        change=float((trace[1]-trace[0]).abs().max())
        assert change>0
        cases.append(dict(features=width,max_forward_error=float((output-expected).abs().max()),
                          max_gradient_error=grad_error,weight_evolution_max_change=change))
    torch.manual_seed(42)
    # Training-mode stochastic activation also agrees for an identical random stream.
    actual=EvolvingLayer(5,8).to(device).double().train()
    reference=GRCU(SimpleNamespace(in_feats=5,out_feats=8,activation=torch.nn.RReLU())).to(device).double().train()
    reference.load_state_dict(actual.state_dict(),strict=True)
    x=torch.randn(1,4,17,5,device=device,dtype=torch.float64,requires_grad=True)
    z=x.detach().clone().requires_grad_()
    identity=torch.eye(17,device=device,dtype=torch.float64)
    state=torch.cuda.get_rng_state()
    output=actual(x,IdentityGraph())
    torch.cuda.set_rng_state(state)
    expected=torch.stack(reference([identity]*4,list(z[0].unbind(0)),
        [torch.zeros(17,1,device=device,dtype=torch.float64)]*4))[None]
    torch.testing.assert_close(output,expected,rtol=1e-9,atol=1e-10)
    output.square().sum().backward();expected.square().sum().backward()
    torch.testing.assert_close(x.grad,z.grad,rtol=1e-8,atol=1e-9)
    train_error=float((output-expected).abs().max())
    for (name,a),(other,b) in zip(actual.named_parameters(),reference.named_parameters()):
        assert name==other
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-8,atol=1e-9)
    net=EvolveGCNForecast(7,IdentityGraph()).to(device).eval()
    x=torch.randn(2,14,40,device=device)
    output=net(x);assert output.shape==(2,7,40) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    buffer=io.BytesIO();torch.save(net.state_dict(),buffer);buffer.seek(0)
    restored=EvolveGCNForecast(7,IdentityGraph()).to(device).eval()
    restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
    torch.testing.assert_close(restored(x),output,rtol=0,atol=0)
    amp_cases=[]
    matrix=np.eye(96,dtype=np.float32)*.5+np.roll(np.eye(96,dtype=np.float32),1,axis=1)*.5
    sparse_graph=SparseGraph(sparse.csr_matrix(matrix),device)
    for label,graph in [('identity',IdentityGraph()),('group',GroupGraph(np.arange(96)%8,device)),('sparse',sparse_graph)]:
        torch.manual_seed(42)
        net=EvolveGCNForecast(7,graph).to(device)
        own=torch.randn(2,14,96,device=device)
        with torch.autocast('cuda',dtype=torch.float16):pred=net(own)
        pred.float().square().mean().backward()
        assert torch.isfinite(pred).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        amp_cases.append(dict(graph=label,status='passed'))
    x=torch.randn(2,96,8,device=device,dtype=torch.float16,requires_grad=True)
    z=x.detach().clone().requires_grad_()
    with torch.autocast('cuda',dtype=torch.float16):actual=sparse_graph(x)
    expected=torch.as_tensor(matrix,device=device)@z.float()
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    actual.square().sum().backward();expected.square().sum().backward()
    torch.testing.assert_close(x.grad,z.grad,rtol=1e-3,atol=1e-3)
    report=dict(status='passed',architecture='EvolveGCN-H',device=torch.cuda.get_device_name(0),
                activation='torch.nn.RReLU() from author run_exp.build_gcn',
                author_factory_activation_calls=len(calls),training_equivalence_max_error=train_error,
                torch=torch.__version__,equivalence_cases=cases,checkpoint_roundtrip='bitwise_equal',
                amp_graph_cases=amp_cases,sparse_dense_forward_backward='passed',
                identity_scope='Identity removes graph-convolution edges; H weight evolution still conditions on pooled node embeddings.',
                source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [ROOT/'external/EvolveGCN/egcn_h.py',ROOT/'external/EvolveGCN/run_exp.py',(Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn.py'),Path(__file__)]})
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
