"""Use the archived training loop with the corrected EvolveGCN-H constructor.

The adapter preserves the artifact format used by the analysis tools. Window
construction, optimization, validation, early stopping and test prediction are
executed by the unchanged archived train_formal.py functions.
"""
from exp.common.provenance import source_hash
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json, os, socket, sys, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'exp/common'),str(ROOT/'exp/common')]
import exp.common.original_protocol as archived
from exp.common.audit import EXPECTED_BUNDLE_SHA256
from exp.common.evolvegcn import EvolveGCNForecast
from exp.common.evolvegcn_training import graph_operator, write_json


def run(a):
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA')
    if a.protocol!='original_masked' or a.batch_size!=6 or a.precision!='legacy_amp':
        raise ValueError('The archived protocol uses masked windows, batch6, and AMP training/validation')
    gate=json.loads(a.verification.read_text())
    assert gate['status']=='passed'
    model_file=(Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn.py')
    assert sha256(model_file.read_bytes()).hexdigest()==source_hash(gate, 'exp/common/evolvegcn.py')
    panel=archived.load_panel(a.bundle,EXPECTED_BUNDLE_SHA256)
    density=None
    if a.density:
        assert a.adjacency=='Identity' and not a.static
        from exp.exp2.density_protocol import selected_indices
        ids,density=selected_indices(panel.raw,a.density)
        panel.raw=panel.raw[:,ids]
        panel.attrs=panel.attrs.iloc[ids].reset_index(drop=True)
    w=archived.make_windows(panel.raw,a.T,a.H)
    device=torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    archived.SEED=a.seed
    graph,graph_record=graph_operator(a.adjacency,panel,57,device)
    static=archived.static_features(panel.attrs)[0] if a.static else None
    own=panel.filled[w.history_idx]
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    a.out.mkdir(parents=True,exist_ok=False)
    write_json(a.out/'running.json',dict(config=cfg,pid=os.getpid(),started_utc=time.time(),graph=graph_record))
    started=time.monotonic()
    model,test_predictions,fit=archived.train_model(
        lambda:EvolveGCNForecast(a.H,graph,static=static),[own],w,device,a.lr,a.batch_size,a.epochs)
    # Validation predictions are an extra export; checkpoint selection above
    # uses the original AMP loss. Test forecasts are returned unchanged.
    inputs=archived.tensor((own-w.mean)/w.std,device)
    val=[]
    with torch.no_grad():
        for ids in torch.arange(w.n_train,w.test_start,device=device).split(a.batch_size):
            val.append(model(inputs[ids]).cpu().float().numpy())
    validation=np.maximum(np.concatenate(val)*w.std+w.mean,0)
    prediction=np.concatenate([validation,test_predictions])
    selected=np.arange(w.n_train,len(w.origins))
    split=np.asarray(['validation']*w.n_val+['test']*(len(w.origins)-w.test_start))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(dict(state_dict=state,config=cfg,mean=w.mean,std=w.std,graph=graph_record),a.out/'best_weights.pt')
    np.savez_compressed(a.out/'predictions.npz',predicted_visits=prediction,
        observed_visits=panel.raw[w.target_idx[selected]],valid=w.valid[selected],
        forecast_origin=np.asarray(panel.dates)[w.origins[selected]],dataset_split=split,
        poi_id=panel.attrs.placekey.to_numpy(dtype=str),horizon_step=np.arange(1,a.H+1))
    write_json(a.out/'epochs.json',fit['epochs'])
    write_json(a.out/'archived_fit.json',fit)
    files=[Path(__file__),model_file,(Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn_training.py'),Path(archived.__file__),
           ROOT/'external/EvolveGCN/egcn_h.py']
    mae=float(np.abs(test_predictions-w.y[w.test_start:]).mean())
    write_json(a.out/'complete.json',dict(status='complete',architecture='EvolveGCN-H',config=cfg,
        train_loop='unchanged_archived_train_formal.train_model',prediction_arithmetic='legacy_numpy',
        validation_loss=fit['validation_loss'],best_epoch=fit['best_epoch'],test_MAE=mae,
        epochs=len(fit['epochs']),seconds=time.monotonic()-started,
        parameters=sum(p.numel() for p in model.parameters()),
        origins=dict(train=w.n_train,validation=w.n_val,test=len(w.origins)-w.test_start),
        scored_validation_origins_by_lead=w.valid[w.n_train:w.test_start].any(2).sum(0).tolist(),
        graph=graph_record,density=density,bundle_sha256=EXPECTED_BUNDLE_SHA256,
        environment=dict(host=socket.gethostname(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__),
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in files},
        checkpoint_sha256=sha256((a.out/'best_weights.pt').read_bytes()).hexdigest(),
        predictions_sha256=sha256((a.out/'predictions.npz').read_bytes()).hexdigest()))
    print(json.dumps(dict(status='complete',test_MAE=mae,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for key in ['bundle','out','verification']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--protocol',default='original_masked')
    for key in ['T','H','seed']:p.add_argument('--'+key,type=int,required=True)
    p.add_argument('--adjacency',required=True)
    p.add_argument('--lr',type=float,required=True)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=6)
    p.add_argument('--precision',default='legacy_amp')
    p.add_argument('--static',action='store_true')
    p.add_argument('--density')
    run(p.parse_args())
