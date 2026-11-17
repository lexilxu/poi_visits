"""Verify feature worlds against individual complete EvolveGCN-H forwards."""
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json,socket,sys
import numpy as np
import torch
from exp.exp5.feature_worlds import GROUPS, encoded_forecast, perturb_group
from exp.exp5.own_gradients import PHYSICAL_BATCH_SIZE, forward_with_support


def main(args):
    assert torch.cuda.is_available()
    sys.path.insert(0,str(args.source/'exp/common'))
    from exp.common.evolvegcn import EvolveGCNForecast, IdentityGraph
    torch.set_num_threads(4);torch.manual_seed(421);torch.set_float32_matmul_precision(args.precision)
    physical=args.physical_batch_size
    n,t,h=96,7,2
    static=torch.cat([torch.randn(n,4),torch.randint(0,6,(n,6)).float()],dim=1)
    model=EvolveGCNForecast(h,IdentityGraph(),static=static).cuda().eval()
    history=torch.randn(1,t,n,device='cuda');encoded=model.encode_static().detach()
    with torch.no_grad():
        base=encoded_forecast(model,history,encoded,2.,1.,physical)
        expected=torch.clamp(model(history.expand(physical,-1,-1).clone())[:1]+2.,min=0.)
    np.testing.assert_array_equal(base.cpu().numpy(),expected.cpu().numpy())
    differentiable_history=history.clone().requires_grad_(True)
    differentiable_static=encoded.clone().requires_grad_(True)
    logits,_=forward_with_support(model,differentiable_history,differentiable_static,physical)
    np.testing.assert_allclose(torch.clamp(logits+2.,min=0.).detach().cpu().numpy(),base.cpu().numpy(),atol=1e-5,rtol=1e-5)
    targets=np.array([2,7,42]);donors=np.array([3,9,51])
    dh=(history[0,:,targets]-history[0,:,donors]).T
    ds=encoded[targets]-encoded[donors]
    rows=[]
    for group,columns in GROUPS.items():
        for step in [.01,1.]:
            actual=perturb_group(model,history,encoded,targets,dh,ds,group,step,2.,1.,batch_size=3,physical_batch_size=physical)
            individual=[]
            for row,target in enumerate(targets):
                own=history.clone();features=encoded.clone()
                if columns is None:own[0,:,target]-=step*dh[row]
                else:features[target,columns]-=step*ds[row,columns]
                with torch.no_grad():individual.append(encoded_forecast(model,own,features,2.,1.,physical)[0,:,target].cpu().numpy())
            individual=np.stack(individual,axis=1)
            np.testing.assert_allclose(actual,individual,rtol=1e-5,atol=1e-5)
            rows.append(dict(group=group,step=step,max_error=float(np.abs(actual-individual).max()),
                maximum_forecast_change=float(np.abs(actual-base[0,:,targets].cpu().numpy()).max())))
        unchanged=perturb_group(model,history,encoded,targets,dh,ds,group,0.,2.,1.,batch_size=3,physical_batch_size=physical)
        np.testing.assert_array_equal(unchanged,base[0,:,targets].cpu().numpy())
    assert max(r['maximum_forecast_change'] for r in rows)>1e-4
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='passed',nodes=n,targets=targets.tolist(),rows=rows,physical_world_batch_size=physical,precision=args.precision,
        source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),(Path(__file__).resolve().parents[2] / 'exp/exp5/feature_worlds.py'),(Path(__file__).resolve().parents[2] / 'exp/exp5/own_gradients.py'),args.source/'exp/common/evolvegcn.py']})
    with args.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--physical-batch-size',type=int,default=4)
    p.add_argument('--precision',choices=['high','highest'],default='high')
    main(p.parse_args())
