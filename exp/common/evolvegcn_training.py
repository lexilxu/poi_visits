"""Fit corrected EvolveGCN-H on CUDA in a new, immutable run directory."""
from exp.common.provenance import source_hash
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
import json,os,socket,sys,time
import numpy as np
import torch
from scipy import sparse
from exp.common.evolvegcn import EvolveGCNForecast, IdentityGraph, GroupGraph, SparseGraph

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'exp/common'),str(ROOT/'exp/common')]
from exp.common.original_protocol import load_panel, make_windows, group_codes, spatial_matrix, correlation_matrix, static_features
from exp.common.audit import EXPECTED_BUNDLE_SHA256
from exp.common.protocol import Protocol


def write_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')
    temporary.replace(path)


def graph_operator(name,panel,train_end,device):
    if name=='Identity':return IdentityGraph(),{'kind':'identity'}
    if name in ('NAICS-2','NAICS-4','Top category'):
        codes=group_codes(panel.attrs)[name]
        return GroupGraph(codes,device),dict(kind='half_identity_half_group_mean',
            group_sha256=sha256(codes.astype('<i8').tobytes()).hexdigest())
    adjacency=correlation_matrix(panel.raw[:train_end]) if name=='Visit correlation' else spatial_matrix(panel.attrs,name)
    matrix=adjacency+sparse.eye(len(panel.attrs),format='csr',dtype=np.float32)
    scale=sparse.diags(np.asarray(matrix.sum(1)).ravel()**-.5)
    matrix=(scale@matrix@scale).astype(np.float32).tocsr()
    record=dict(kind='original_symmetric_degree_normalization',
                csr_sha256=sha256(matrix.data.tobytes()+matrix.indices.tobytes()+matrix.indptr.tobytes()).hexdigest())
    return SparseGraph(matrix,device),record


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA')
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.set_float32_matmul_precision('high' if args.precision=='legacy_amp' else 'highest')
    torch.backends.cuda.matmul.allow_tf32=args.precision=='legacy_amp'
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda:0')
    gate=json.loads(args.verification.read_text())
    current=sha256((Path(__file__).resolve().parents[2] / 'exp/common/evolvegcn.py').read_bytes()).hexdigest()
    if gate['status']!='passed' or source_hash(gate, 'exp/common/evolvegcn.py')!=current:
        raise RuntimeError('The model must pass the current source equivalence gate')
    panel=load_panel(args.bundle,EXPECTED_BUNDLE_SHA256)
    density_record=None
    if args.density:
        if args.protocol!='original_masked' or args.adjacency!='Identity' or args.static:
            raise ValueError('Original density refits use history-only identity EvolveGCN-H')
        from exp.exp2.density_protocol import selected_indices
        ids,density_record=selected_indices(panel.raw,args.density)
        panel.raw=panel.raw[:,ids]
        panel.attrs=panel.attrs.iloc[ids].reset_index(drop=True)
    if args.protocol=='original_masked':
        w=make_windows(panel.raw,args.T,args.H)
        origins,history,target,mask=w.origins,w.history_idx,w.target_idx,w.valid
        split=np.array(['train']*w.n_train+['validation']*w.n_val+['test']*(len(origins)-w.test_start))
        mean,std=w.mean,w.std;train_end=57
    else:
        origins,history,target,split,mask=Protocol(args.T,args.H,50,64).windows(panel.raw)
        mean,std=float(np.nanmean(panel.raw[:50])),float(np.nanstd(panel.raw[:50]));train_end=50
    graph,graph_record=graph_operator(args.adjacency,panel,train_end,device)
    static=None
    if args.static:
        static,_=static_features(panel.attrs)
        if args.protocol=='complete_h7':
            mu=static[:,:4].mean(0);sigma=static[:,:4].std(0)
            static[:,:4]=(static[:,:4]-mu)/np.where(sigma>0,sigma,1.)
    args.out.mkdir(parents=True,exist_ok=False)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    write_json(args.out/'running.json',dict(config=config,pid=os.getpid(),started_utc=time.time(),graph=graph_record))
    model=EvolveGCNForecast(args.H,graph,static=static).to(device)
    inputs=torch.as_tensor((panel.filled[history]-mean)/std,device=device,dtype=torch.float32)
    responses=torch.as_tensor(np.nan_to_num(panel.raw[target],nan=0.),device=device,dtype=torch.float32)
    observed=torch.as_tensor(mask,device=device)
    train_ids=np.flatnonzero(split=='train');val_ids=np.flatnonzero(split=='validation')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,
                               weight_decay=1e-4 if args.protocol=='complete_h7' else 0.)
    amp=args.precision=='legacy_amp'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    rng=np.random.default_rng(args.seed);logs=[];best=float('inf');best_epoch=0;started=time.monotonic()
    def batches(indices):
        return np.array_split(indices,int(np.ceil(len(indices)/args.batch_size)))
    for epoch in range(1,args.epochs+1):
        began=time.monotonic();model.train();train_sum=0.;train_count=0
        for ids in batches(rng.permutation(train_ids)):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                predictions=model(inputs[ids])
            delta=predictions.float()-(responses[ids]-mean)/std
            error=delta.square() if args.protocol=='original_masked' else delta.abs()
            values=error[observed[ids]];loss=values.mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.protocol=='complete_h7':torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not amp and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradient')
            scaler.step(optimizer);scaler.update()
            train_sum+=float(values.detach().sum());train_count+=values.numel()
        model.eval();numerator=0.;denominator=0
        with torch.no_grad():
            for ids in batches(val_ids):
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                    prediction=model(inputs[ids])
                prediction=prediction.float()
                if args.protocol=='original_masked':error=(prediction-(responses[ids]-mean)/std).square()
                else:error=(torch.clamp(prediction*std+mean,min=0.)-responses[ids]).abs()
                values=error[observed[ids]];numerator+=float(values.sum());denominator+=values.numel()
        score=numerator/denominator
        if not np.isfinite(score):raise RuntimeError('Nonfinite validation objective')
        logs.append(dict(epoch=epoch,train_loss=train_sum/train_count,validation_loss=score,
                         seconds=time.monotonic()-began))
        if score<best:
            best=score;best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,config=config,mean=mean,std=std,graph=graph_record),args.out/'best_weights.pt')
        write_json(args.out/'epochs.json',logs)
        print(json.dumps(logs[-1]),flush=True)
        if epoch-best_epoch>=10:break
    model.load_state_dict(state);model.eval()
    selected=np.flatnonzero(split!='train');outputs=[]
    with torch.no_grad():
        for ids in batches(selected):outputs.append(torch.clamp(model(inputs[ids])*std+mean,min=0.).cpu().numpy())
    prediction=np.concatenate(outputs)
    if not np.isfinite(prediction).all():raise RuntimeError('Nonfinite saved forecast')
    np.savez_compressed(args.out/'predictions.npz',predicted_visits=prediction,
        observed_visits=panel.raw[target[selected]],valid=mask[selected],
        forecast_origin=np.asarray(panel.dates)[origins[selected]],dataset_split=split[selected],
        poi_id=panel.attrs.placekey.to_numpy(dtype=str),horizon_step=np.arange(1,args.H+1))
    test=split[selected]=='test';valid=mask[selected][test]
    mae=float(np.abs(prediction[test]-np.nan_to_num(panel.raw[target[selected]][test],nan=0.))[valid].mean())
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'external/EvolveGCN/egcn_h.py',ROOT/'exp/common/original_protocol.py',ROOT/'exp/common/protocol.py']
    write_json(args.out/'complete.json',dict(status='complete',architecture='EvolveGCN-H',config=config,
        validation_loss=best,best_epoch=best_epoch,test_MAE=mae,epochs=len(logs),seconds=time.monotonic()-started,
        parameters=sum(p.numel() for p in model.parameters()),origins={s:int((split==s).sum()) for s in ['train','validation','test']},
        scored_validation_origins_by_lead=mask[split=='validation'].any(2).sum(0).tolist(),
        environment=dict(host=socket.gethostname(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__),
        graph=graph_record,bundle_sha256=EXPECTED_BUNDLE_SHA256,
        density=density_record,
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in files},
        checkpoint_sha256=sha256((args.out/'best_weights.pt').read_bytes()).hexdigest(),
        predictions_sha256=sha256((args.out/'predictions.npz').read_bytes()).hexdigest()))
    print(json.dumps(dict(status='complete',test_MAE=mae,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--verification',type=Path,required=True)
    p.add_argument('--protocol',choices=['original_masked','complete_h7'],required=True)
    p.add_argument('--T',type=int,required=True);p.add_argument('--H',type=int,required=True)
    p.add_argument('--adjacency',choices=['Identity','NAICS-2','NAICS-4','Top category','Distance KNN','Distance RBF','Visit correlation'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--lr',type=float,required=True)
    p.add_argument('--static',action='store_true');p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--precision',choices=['fp32','legacy_amp'],default='fp32')
    p.add_argument('--density')
    args=p.parse_args()
    if args.out.exists():p.error('Existing run directories are immutable')
    if args.protocol=='complete_h7' and args.H!=7:p.error('Complete-horizon confirmation is prespecified for H=7')
    run(args)
