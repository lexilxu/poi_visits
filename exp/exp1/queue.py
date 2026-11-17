"""Serial CUDA scheduler with pinned configurations and verified reuse."""
from pathlib import Path
from argparse import ArgumentParser
from hashlib import sha256
from datetime import datetime,timezone
import json,os,subprocess,sys,time


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def specifications():
    graphs=['Identity','NAICS-2','NAICS-4','Distance KNN','Distance RBF','Visit correlation']
    settings=[(14,7),(28,7),(28,14),(28,28)]
    specs=[]
    # Recover the original single-seed tables first; retain the fixed five seeds.
    for seed in range(42,47):
        for graph in graphs:
            for t,h in settings:
                specs.append(dict(protocol='original_masked',T=t,H=h,adjacency=graph,seed=seed,
                                  lr=.0005 if graph=='NAICS-2' else .001,static=False,epochs=50,
                                  batch_size=6,precision='legacy_amp'))
        specs.append(dict(protocol='original_masked',T=14,H=7,adjacency='Identity',seed=seed,
                          lr=.001,static=True,epochs=50,batch_size=6,precision='legacy_amp'))
    for seed in range(42,47):
        for graph in ['Identity','NAICS-2','NAICS-4']:
            for t in [14,28]:
                specs.append(dict(protocol='complete_h7',T=t,H=7,adjacency=graph,seed=seed,
                                  lr=.001,static=False,epochs=100))
        specs.append(dict(protocol='complete_h7',T=14,H=7,adjacency='Identity',seed=seed,
                          lr=.001,static=True,epochs=100))
    return specs


def key(s):
    return f"{s['protocol']}_T{s['T']}_H{s['H']}_{s['adjacency'].replace(' ','_')}_{'static' if s['static'] else 'history'}_seed{s['seed']}"


def main(args):
    args.root.mkdir(parents=True,exist_ok=True)
    specs=specifications()
    if args.phase=='original':specs=[s for s in specs if s['protocol']=='original_masked']
    plan=dict(version='EvolveGCN-H_author_RReLU_archived_trainer_v2',created_utc=datetime.now(timezone.utc).isoformat(),
              specs=specs,alias={'Top category':'NAICS-4'},
              architecture='Two EvolveGCN-H layers with widths 32 and 32, author-default RReLU, linear H-output head; no node GRU or last-visit skip.',
              purpose='Correct the mislabeled original architecture while preserving the five-model manuscript structure.',
              original_protocol='Unchanged archived train_formal.train_model: batch6, AMP training and validation, FP32 test prediction, masked splits, original scale, learning rates, MSE, 50 epochs and patience10.',
              confirmation_protocol='Existing complete-window T14/T28 H7 protocol: train end 50, validation end 64, MAE, lr .001, decay 1e-4, clipping 5; 100 epochs, patience 10.',
              seeds=[42,43,44,45,46],original_batch_size=6,
              interpretation='H weight evolution pools node embeddings even with identity graph convolution. Graph comparisons remain configuration contrasts.',
              further_required='Recompute affected density, perturbation and attribution summaries from corrected checkpoints; preserve all negative findings.',
              selection='All prespecified fits retained; no test-based selection; Top category reuses the identical NAICS-4 graph only after partition audit.')
    manifest=args.root/'plan.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['specs']!=plan['specs']:raise RuntimeError('Frozen plan mismatch')
    else:atomic(manifest,plan)
    if args.plan_only:
        print(json.dumps(dict(fits=len(plan['specs']),plan=str(manifest))));return
    lock=args.root/'queue.lock'
    with lock.open('x') as handle:handle.write(str(os.getpid()))
    started=time.monotonic();completed=[]
    try:
        for spec in plan['specs']:
            name=key(spec);out=args.root/'runs'/name
            if (out/'complete.json').exists():
                saved=json.loads((out/'complete.json').read_text())
                assert saved['status']=='complete'
                for file,field in [('best_weights.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256')]:
                    assert sha256((out/file).read_bytes()).hexdigest()==saved[field]
                for field,value in spec.items():assert saved['config'][field]==value
                completed.append(name);continue
            if out.exists():raise RuntimeError(f'Unfinished run preserved for investigation: {out}')
            status=dict(status='running',pid=os.getpid(),current=name,completed=len(completed),total=len(plan['specs']),updated_utc=time.time())
            atomic(args.root/'status.json',status)
            module='exp.exp1.train' if spec['protocol']=='original_masked' else 'exp.exp1.train_complete'
            command=[sys.executable,'-m',module,'--bundle',str(args.bundle),
                     '--verification',str(args.root/'verification_minimal.json'),'--out',str(out)]
            for field,value in spec.items():
                if field=='static':
                    if value:command.append('--static')
                else:command.extend(['--'+field.replace('_','-'),str(value)])
            logs=args.root/'logs';logs.mkdir(exist_ok=True)
            with (logs/(name+'.log')).open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Fit {name} exited {result.returncode}')
            if not (out/'complete.json').exists():raise RuntimeError('Missing completed fit record')
            completed.append(name)
            print(json.dumps(dict(completed=len(completed),total=len(plan['specs']),fit=name)),flush=True)
        atomic(args.root/'status.json',dict(status='complete',completed=len(completed),total=len(plan['specs']),seconds=time.monotonic()-started))
    except Exception as error:
        atomic(args.root/'status.json',dict(status='failed',completed=len(completed),current=locals().get('name'),error=str(error),updated_utc=time.time()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--phase',choices=['original','all'],default='original')
    p.add_argument('--plan-only',action='store_true');main(p.parse_args())
