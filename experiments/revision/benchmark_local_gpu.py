"""Short local GPU timing run on the actual approved Seattle panel.

This is a speed/memory benchmark, not a fitted result for the manuscript.
It uses the same FP32 models, complete-horizon windows, masked MAE, Adam,
batch size, and normalization as train_controlled.py. No test labels are used.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import os
import platform
import subprocess
import sys
import time
import numpy as np
from audit_protocol import load_panel,EXPECTED_BUNDLE_SHA256
from training_protocol import Protocol,feature_mask

ROOT=Path(__file__).resolve().parents[2]
MODELS=('GraphGRU','GNN-static','Shared-LSTM','Fixed-MTGNN')


def benchmark(args):
    import torch
    from train_controlled import make_model
    from train_formal import static_features
    if not torch.cuda.is_available():
        raise RuntimeError('A working CUDA PyTorch build is required; CPU timing is not substituted')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42); np.random.seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda:0')
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    raw,dates,attrs=load_panel(args.bundle)
    protocol=Protocol(args.T,7,50,64)
    origins,history,target,split,valid=protocol.windows(raw)
    selected=split!='test'
    history,target,split,valid=history[selected],target[selected],split[selected],valid[selected]
    mean,std=float(np.nanmean(raw[:50])),float(np.nanstd(raw[:50]))
    own=torch.as_tensor((np.nan_to_num(raw[history],nan=0.)-mean)/std,dtype=torch.float32,device=device)
    response=torch.as_tensor(np.nan_to_num(raw[target],nan=0.),dtype=torch.float32,device=device)
    observed=torch.as_tensor(valid,device=device)
    static=None
    if args.model=='GNN-static':
        static,_=static_features(attrs)
        eligible=np.isfinite(raw[:50]).any(0)
        center=static[eligible,:4].mean(0); scale=static[eligible,:4].std(0)
        static[:,:4]=(static[:,:4]-center)/np.where(scale>0,scale,1.)
    model=make_model(args.model,7,static,feature_mask('full'),len(attrs),args.T,
                     (lambda x:x,lambda x:x),device,.1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    train_ids=np.flatnonzero(split=='train'); val_ids=np.flatnonzero(split=='validation')
    def step(ids):
        optimizer.zero_grad(set_to_none=True)
        prediction=model(own[ids],own[ids])*std+mean
        loss=((prediction-response[ids])/std).abs()[observed[ids]].mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite warmup/training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        return loss.detach()
    model.train()
    for _ in range(args.warmup_steps):
        step(train_ids[:args.batch_size])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    epoch_rows=[]; rng=np.random.default_rng(42)
    for epoch in range(args.epochs):
        model.train(); torch.cuda.synchronize(); started=time.perf_counter()
        for start in range(0,len(train_ids),args.batch_size):
            # One permutation per epoch, shared over the full set of origins.
            if start==0:
                permutation=rng.permutation(train_ids)
            loss=step(permutation[start:start+args.batch_size])
        torch.cuda.synchronize(); train_seconds=time.perf_counter()-started
        model.eval(); started=time.perf_counter(); error=0.; count=0
        with torch.no_grad():
            for start in range(0,len(val_ids),args.batch_size):
                ids=val_ids[start:start+args.batch_size]
                prediction=(model(own[ids],own[ids])*std+mean).clamp_min(0)
                values=(prediction-response[ids]).abs()[observed[ids]]
                error+=float(values.sum().cpu()); count+=values.numel()
        torch.cuda.synchronize(); validation_seconds=time.perf_counter()-started
        epoch_rows.append(dict(epoch=epoch+1,training_seconds=train_seconds,
                               validation_seconds=validation_seconds,validation_MAE=error/count))
        print(json.dumps(dict(model=args.model,**epoch_rows[-1])),flush=True)
    result=dict(status='complete',purpose='speed_and_memory_only_not_manuscript_results',
                model=args.model,adjacency='Adaptive' if args.model=='Adaptive-MTGNN' else 'Identity',GPU=torch.cuda.get_device_name(),
                compute_capability=list(torch.cuda.get_device_capability()),
                GPU_total_MiB=torch.cuda.get_device_properties(0).total_memory/2**20,
                torch=torch.__version__,CUDA_runtime=torch.version.cuda,python=sys.version,
                executable=sys.executable,platform=platform.platform(),
                precision='FP32',batch_size=args.batch_size,T=args.T,H=7,POIs=len(attrs),
                training_origins=len(train_ids),validation_origins=len(val_ids),
                train_updates_per_epoch=int(np.ceil(len(train_ids)/args.batch_size)),
                parameters=sum(p.numel() for p in model.parameters()),
                warmup_steps=args.warmup_steps,measured_epochs=args.epochs,
                mean_training_seconds=float(np.mean([r['training_seconds'] for r in epoch_rows])),
                mean_validation_seconds=float(np.mean([r['validation_seconds'] for r in epoch_rows])),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                bundle_sha256=EXPECTED_BUNDLE_SHA256,epoch_details=epoch_rows)
    result['estimated_100_epoch_minutes']=(result['mean_training_seconds']+result['mean_validation_seconds'])*100/60
    result['estimate_scope']='Linear timing extrapolation only; excludes setup, saving, test inference, early stopping, and thermal changes.'
    (args.out/(args.model+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


def run_all(args):
    args.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name in MODELS:
        command=[sys.executable,str(Path(__file__).resolve()),'--model',name,'--bundle',str(args.bundle.resolve()),
                 '--out',str(args.out.resolve()),'--T',str(args.T),'--epochs',str(args.epochs),
                 '--batch-size',str(args.batch_size),'--warmup-steps',str(args.warmup_steps),'--threads',str(args.threads)]
        started=time.monotonic()
        with (args.out/(name+'.log')).open('w',encoding='utf8') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.case_timeout)
        path=args.out/(name+'.json')
        if completed.returncode==0 and path.is_file():
            result=json.loads(path.read_text(encoding='utf8'))
            fields=('model','status','parameters','mean_training_seconds','mean_validation_seconds',
                    'peak_allocated_MiB','peak_reserved_MiB','estimated_100_epoch_minutes')
            rows.append({key:result[key] for key in fields})
        else:
            rows.append(dict(model=name,status='failed',exit_code=completed.returncode))
        print(json.dumps({**rows[-1],'elapsed_seconds':time.monotonic()-started}),flush=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (args.out/'benchmark_summary.csv').open('w',newline='',encoding='utf8') as file:
        writer=csv.DictWriter(file,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    if any(row['status']!='complete' for row in rows):
        raise SystemExit('Some benchmark cases failed; inspect the per-model logs')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT.parent/'paper_figures/mechanism_figures/formal_2020q4/approved_panel.zip')
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--model',choices=(*MODELS,'Adaptive-MTGNN'))
    parser.add_argument('--memory-fraction',type=float)
    parser.add_argument('--T',type=int,default=14); parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--warmup-steps',type=int,default=2); parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4); parser.add_argument('--case-timeout',type=int,default=600)
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.threads)<1 or args.warmup_steps<0:
        parser.error('Positive epochs, batch size, and threads are required')
    if args.model:
        benchmark(args)
    else:
        run_all(args)
