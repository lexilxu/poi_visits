"""Bounded, resumable job queue for the prepared controlled neural experiments.

The default is a dry run. Execution requires CUDA preflight and a synthetic
forward/backward smoke check. No SSH, cloud startup, or package installation
is performed. Failed fits stop the queue; outputs are never overwritten.
"""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json
import math
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
TRAIN=Path(__file__).with_name('train_controlled.py')
SEEDS=(42,43,44,45,46)
LEARNING_RATES=(.0003,.001,.003)


def recipes():
    result=[]
    def add(stage,model,t=14,adjacency='Identity',variant='full',loss='MAE',graph=None,k=None):
        name=f'{model}_T{t}_{adjacency}_{variant}_{loss}'
        if graph:
            name+=f'_{graph}_k{k}'
        result.append(dict(id=name,stage=stage,model=model,T=t,H=7,adjacency=adjacency,
                           variant=variant,loss=loss,graph=graph,k=k))
    for t in (14,28):
        for model in ('GraphGRU','Fixed-MTGNN'):
            for graph in ('Identity','NAICS-2','NAICS-4'):
                add('core',model,t,graph)
        add('core','Shared-LSTM',t)
        add('core','GNN-static',t)
        add('adaptive','Adaptive-MTGNN',t,adjacency='Adaptive')
        for model in ('GraphGRU','Fixed-MTGNN','Shared-LSTM'):
            add('loss','%s'%model,t,loss='MSE')
    for model in ('GraphGRU','Fixed-MTGNN'):
        for k in (5,10,20):
            for graph in ('NAICS-2','NAICS-4','Random'):
                add('graph',model,adjacency='Controlled',graph=graph,k=k)
    variants=('history_only','without_location','without_identity','without_parent','without_others',
              'only_category','only_subcategory','only_naics2','only_naics4','without_naics2','without_naics4')
    for variant in variants:
        add('ablation','GNN-static',variant=variant)
    for adjacency in ('Distance KNN','Distance RBF','Visit correlation'):
        add('spatial','GraphGRU',adjacency=adjacency)
    add('spatial','Fixed-MTGNN',adjacency='Distance KNN')
    for kind in ('Dense','Recent-dense','Persist-dense'):
        for exclusion in (10,25,50,75):
            add('density','GraphGRU',t=28)
            result[-1]['density']=f'{kind}{exclusion}'
            result[-1]['id']+='_'+result[-1]['density']
    for variant in ('full','without_parent'):
        add('metadata','GNN-static',variant=variant)
        result[-1]['id']+='_early_metadata'
    return result


def plan(stage):
    selected=[r for r in recipes() if stage=='all' or r['stage']==stage or
              (stage=='extension' and r['stage'] in ('spatial','density','metadata','adaptive'))]
    if stage=='extension':
        order={'metadata':0,'density':1,'spatial':2,'adaptive':3}
        selected.sort(key=lambda r:order[r['stage']])
    return dict(status='prepared_not_executed',train_end=50,validation_end=64,
                seeds=list(SEEDS),learning_rates=list(LEARNING_RATES),recipes=selected,
                maximum_fits=len(selected)*(len(LEARNING_RATES)+len(SEEDS)-1),
                selection='For each recipe choose LR by seed-42 validation MAE; retain that seed-42 fit and refit seeds 43--46.',
                graph_seeds='Use the paired graph construction seed equal to the training seed for controlled graphs.',
                missing_adapters=['BysGNN'],
                missing_evaluations=['Integrated gradients','Matched-neighbor interventions','Independent period/city'])


def selected_fit(fits):
    """Only validation MAE selects a configuration; seeds are never selected."""
    if not fits or any(not math.isfinite(f['validation_MAE']) for f in fits):
        raise ValueError('All tuning fits must report finite validation MAE')
    return min(fits,key=lambda f:(f['validation_MAE'],f['lr']))


def persist(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temp.replace(path)


def command(args,recipe,seed,lr,destination):
    cmd=[sys.executable,str(TRAIN),'--bundle',str(args.bundle.resolve()),'--out',str(destination),
         '--model',recipe['model'],'--adjacency',recipe['adjacency'],'--variant',recipe['variant'],
         '--T',str(recipe['T']),'--H','7','--train-end','50','--seed',str(seed),'--lr',str(lr),
         '--loss',recipe['loss'],'--epochs',str(args.epochs),'--patience','10',
         '--batch-size',str(args.batch_size),'--device',args.device]
    if recipe['graph']:
        path=args.graph_dir/f"{recipe['graph']}_k{recipe['k']}_seed{seed}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        cmd+=['--graph-file',str(path.resolve())]
    if recipe['stage']=='spatial':
        path=args.extra_input_dir/(recipe['adjacency'].replace(' ','_')+'.npz')
        if not path.exists():raise FileNotFoundError(path)
        cmd+=['--sparse-graph',str(path.resolve())]
    if recipe.get('density'):cmd+=['--density',recipe['density']]
    if recipe['stage']=='metadata':
        path=args.extra_input_dir/'earliest_metadata.csv'
        if not path.exists():raise FileNotFoundError(path)
        cmd+=['--attributes-file',str(path.resolve())]
    return cmd


def run(args,manifest):
    from train_controlled import environment
    env=environment()
    if not env.get('cuda_available'):
        raise RuntimeError('CUDA preflight failed: '+json.dumps(env))
    if not args.bundle.is_file():
        raise FileNotFoundError(args.bundle)
    # Check inputs for every selected graph recipe before spending GPU time.
    for recipe in manifest['recipes']:
        for seed in SEEDS:
            command(args,recipe,seed,LEARNING_RATES[0],args.out/'preflight')
    started=time.monotonic(); deadline=started+args.max_hours*3600
    args.out.mkdir(parents=True,exist_ok=True)
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous!=manifest:
            raise ValueError('Existing queue manifest differs; choose another output directory')
    else:
        persist(manifest_path,manifest)
    subprocess.run([sys.executable,str(TRAIN),'--smoke','--device',args.device],check=True,timeout=min(300,deadline-time.monotonic()))
    completed=0
    def one(recipe,seed,lr):
        nonlocal completed
        destination=args.out/recipe['id']/f'seed{seed}_lr{lr:g}'
        result=destination/'fit.json'
        if result.is_file():
            fit=json.loads(result.read_text())
            if fit.get('status')!='complete':
                raise ValueError(f'Incomplete saved fit: {result}')
            return {**fit,'directory':str(destination)}
        if destination.exists():
            raise RuntimeError(f'Interrupted fit preserved at {destination}; inspect it before restarting')
        remaining=deadline-time.monotonic()
        if remaining<60 or completed>=args.max_new_fits:
            raise TimeoutError('Queue budget reached before the next fit; rerun to resume completed jobs')
        destination.parent.mkdir(parents=True,exist_ok=True)
        cmd=command(args,recipe,seed,lr,destination)
        log=destination.parent/(destination.name+'.console.log')
        print(json.dumps({'starting':recipe['id'],'seed':seed,'lr':lr,'log':str(log)}),flush=True)
        with log.open('w',encoding='utf8') as stream:
            subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,
                           timeout=min(args.max_fit_minutes*60,remaining))
        fit=json.loads(result.read_text()); completed+=1
        return {**fit,'directory':str(destination)}
    try:
        for recipe in manifest['recipes']:
            tuning=[one(recipe,42,lr) for lr in LEARNING_RATES]
            chosen=selected_fit(tuning)
            selected=[chosen]+[one(recipe,seed,chosen['lr']) for seed in SEEDS[1:]]
            persist(args.out/recipe['id']/'selected_runs.json',dict(
                recipe=recipe,selected_lr=chosen['lr'],selection_metric='seed42 validation MAE',
                runs=[{k:f[k] for k in ('seed','lr','validation_MAE','directory')} for f in selected]))
        persist(args.out/'queue_status.json',dict(status='complete',new_fits=completed,seconds=time.monotonic()-started))
    except Exception as error:
        persist(args.out/'queue_status.json',dict(status='stopped',reason=str(error),new_fits=completed,
                                                seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('core','loss','graph','ablation','adaptive','spatial','density','metadata','extension','all'),default='core')
    p.add_argument('--plan-out',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--bundle',type=Path); p.add_argument('--graph-dir',type=Path)
    p.add_argument('--extra-input-dir',type=Path)
    p.add_argument('--out',type=Path); p.add_argument('--max-hours',type=float)
    p.add_argument('--max-new-fits',type=int,default=7)
    p.add_argument('--max-fit-minutes',type=float,default=30)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args(); manifest=plan(args.stage)
    # Pin all executable sources as queue provenance, including external models.
    sources=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'experiments/archived_2020q4').glob('*.py'))
    sources+=list((ROOT/'external/MTGNN').glob('*.py'))+list((ROOT/'src').glob('*.py'))
    manifest['source_sha256']={str(f.relative_to(ROOT)).replace('\\','/'):sha256(f.read_bytes()).hexdigest() for f in sorted(sources)}
    manifest['fit_limits']={'epochs':args.epochs,'batch_size':args.batch_size,'max_fit_minutes':args.max_fit_minutes}
    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True,exist_ok=True); persist(args.plan_out,manifest)
    print(json.dumps({k:manifest[k] for k in ('status','maximum_fits','seeds','selection','missing_adapters')},indent=2),flush=True)
    if args.execute:
        if args.bundle is None or args.out is None or args.max_hours is None or args.max_hours<=0 or args.max_new_fits<1:
            p.error('Execution requires --bundle, --out, positive --max-hours, and --max-new-fits')
        if args.stage in ('graph','all') and args.graph_dir is None:
            p.error('Graph execution requires --graph-dir')
        run(args,manifest)
