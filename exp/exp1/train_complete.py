"""Run the prior complete-window trainer with an EvolveGCN-H model factory."""
from argparse import ArgumentParser, Namespace
from exp.common.provenance import source_hash
from pathlib import Path
from hashlib import sha256
import json, os, socket, sys, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'exp/common'),str(ROOT/'exp/common')]
import exp.common.controlled_training as controlled
from exp.common.original_protocol import load_panel
from exp.common.audit import EXPECTED_BUNDLE_SHA256
from exp.common.evolvegcn import EvolveGCNForecast
from exp.common.evolvegcn_training import graph_operator, write_json


class CompleteWindowEvolveGCN(EvolveGCNForecast):
    def forward(self,graph_input,own):
        return super().forward(own)


def run(a):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if a.protocol!='complete_h7' or a.H!=7 or a.adjacency not in ['Identity','NAICS-2','NAICS-4']:
        raise ValueError('Use the prespecified complete-H7 configurations')
    if a.static and a.adjacency!='Identity':raise ValueError('Static confirmation uses identity adjacency')
    gate=json.loads(a.verification.read_text())
    model_file=(Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn.py')
    assert gate['status']=='passed'
    assert source_hash(gate, 'exp/common/evolvegcn.py')==sha256(model_file.read_bytes()).hexdigest()
    panel=load_panel(a.bundle,EXPECTED_BUNDLE_SHA256)
    graph,graph_record=graph_operator(a.adjacency,panel,50,torch.device('cuda:0'))
    def factory(model_name,horizon,static,mask,nodes,t,unused,device,dropout):
        assert model_name in ['GraphGRU','GNN-static'] and np.all(mask==1)
        assert nodes==len(panel.attrs) and t==a.T
        return CompleteWindowEvolveGCN(horizon,graph,static=static)
    a.out.mkdir(parents=True,exist_ok=False)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    write_json(a.out/'running.json',dict(config=cfg,pid=os.getpid(),started_utc=time.time(),graph=graph_record))
    args=Namespace(bundle=a.bundle,out=a.out/'controlled_artifacts',model='GNN-static' if a.static else 'GraphGRU',
        adjacency=a.adjacency,variant='full',graph_file=None,sparse_graph=None,attributes_file=None,density=None,
        panel_file=None,T=a.T,H=a.H,train_end=50,seed=a.seed,lr=a.lr,dropout=.1,epochs=a.epochs,
        patience=10,batch_size=a.batch_size,loss='MAE',device='cuda:0',threads=4)
    torch.set_float32_matmul_precision('highest')
    controlled.run(args,model_factory=factory)
    child=args.out
    fit=json.loads((child/'fit.json').read_text())
    checkpoint=torch.load(child/'best_weights.pt',map_location='cpu',weights_only=True)
    mean,std=float(np.nanmean(panel.raw[:50])),float(np.nanstd(panel.raw[:50]))
    torch.save(dict(state_dict=checkpoint['state_dict'],config=cfg,mean=mean,
        std=std,graph=graph_record),a.out/'best_weights.pt')
    # Preserve the original trainer export exactly.
    import shutil
    shutil.copy2(child/'predictions.npz',a.out/'predictions.npz')
    shutil.copy2(child/'epoch_log.json',a.out/'epochs.json')
    with np.load(a.out/'predictions.npz') as z:
        test=z['dataset_split']=='test'
        keep=z['valid'][test]
        mae=float(np.abs(z['predicted_visits'][test]-np.nan_to_num(z['observed_visits'][test],nan=0.))[keep].mean())
        val=z['dataset_split']=='validation'
        scored=z['valid'][val].any(2).sum(0).tolist()
    sources=[Path(__file__),model_file,(Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn_training.py'),Path(controlled.__file__),
        ROOT/'exp/common/protocol.py',ROOT/'external/EvolveGCN/egcn_h.py']
    write_json(a.out/'complete.json',dict(status='complete',architecture='EvolveGCN-H',config=cfg,
        train_loop='prior_train_controlled.run_with_model_factory',prediction_arithmetic='torch_fp32',
        validation_loss=fit['validation_MAE'],best_epoch=fit['best_epoch'],test_MAE=mae,
        epochs=len(fit['epochs']),seconds=fit['seconds'],parameters=fit['parameters'],origins=fit['origins'],
        scored_validation_origins_by_lead=scored,graph=graph_record,density=None,bundle_sha256=EXPECTED_BUNDLE_SHA256,
        environment=fit['environment'],source_sha256={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in sources},
        checkpoint_sha256=sha256((a.out/'best_weights.pt').read_bytes()).hexdigest(),
        predictions_sha256=sha256((a.out/'predictions.npz').read_bytes()).hexdigest(),
        controlled_fit_sha256=sha256((child/'fit.json').read_bytes()).hexdigest()))
    print(json.dumps(dict(status='complete',test_MAE=mae)),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for key in ['bundle','out','verification']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--protocol',default='complete_h7')
    for key in ['T','H','seed']:p.add_argument('--'+key,type=int,required=True)
    p.add_argument('--adjacency',required=True);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--static',action='store_true');p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--precision',default='fp32')
    run(p.parse_args())
