"""GPU-only comparison with a full, individual-output Jacobian."""
from pathlib import Path
from argparse import ArgumentParser
import hashlib,json,socket,sys
import torch
from exp.exp5.own_gradients import forward_with_support, diagonal_blocks


def main(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    sys.path.insert(0,str(args.source/'exp/common'))
    from exp.common.evolvegcn import EvolvingLayer, IdentityGraph
    torch.manual_seed(20261115)
    class TestForecast(torch.nn.Module):
        def __init__(self):
            super().__init__();self.graph=IdentityGraph()
            self.layers=torch.nn.ModuleList([EvolvingLayer(5,8),EvolvingLayer(8,8)])
            self.decoder=torch.nn.Linear(8,2)
    model=TestForecast().cuda().double()
    history=torch.randn(1,3,96,device='cuda',dtype=torch.float64,requires_grad=True)
    static=torch.randn(96,4,device='cuda',dtype=torch.float64,requires_grad=True)
    prediction,support=forward_with_support(model,history,static,args.physical_batch_size)
    assert len(support)<96
    errors=[];naive_errors=[]
    for lead in range(2):
        actual_h,actual_s=diagonal_blocks(prediction,history,static,support,lead,chunk_size=7)
        naive_h,naive_s=torch.autograd.grad(prediction[:,lead].sum(),(history,static),retain_graph=True)
        reference_h=torch.zeros_like(history);reference_s=torch.zeros_like(static)
        for j in range(96):
            gh,gs=torch.autograd.grad(prediction[0,lead,j],(history,static),retain_graph=True)
            reference_h[0,:,j]=gh[0,:,j];reference_s[j]=gs[j]
        torch.testing.assert_close(actual_h,reference_h,rtol=1e-9,atol=1e-11)
        torch.testing.assert_close(actual_s,reference_s,rtol=1e-9,atol=1e-11)
        errors.append(max(float((actual_h-reference_h).abs().max()),float((actual_s-reference_s).abs().max())))
        naive_errors.append(max(float((naive_h-reference_h).abs().max()),float((naive_s-reference_s).abs().max())))
    assert max(naive_errors)>1e-8
    report=dict(status='passed',device=torch.cuda.get_device_name(0),nodes=96,global_support_nodes=len(support),
                physical_batch_size=args.physical_batch_size,
                max_error_by_lead=errors,old_sum_gradient_error_by_lead=naive_errors,
                scope='Exact own-POI derivatives for identity convolution; scores describe own-input reliance, not the complete cross-POI attribution.',
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/exp5/own_gradients.py')]})
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--physical-batch-size',type=int,default=4)
    main(p.parse_args())
