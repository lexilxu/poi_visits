"""Refit the 12 prespecified visit-density subsets using the original training loop.

This is the density phase of run_minimal_followup_training.py: 60 fits, five
fixed seeds, T=28/H=14, Identity, batch 6, MSE, and legacy AMP. Full-cohort fits
are reused from exp1. Plans can be inspected without a GPU or data access.
"""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json, os, subprocess, sys
from exp.common.paths import REPO_ROOT

def specifications():
    return [dict(protocol='original_masked', T=28, H=14, adjacency='Identity',
                 seed=seed, lr=.001, static=False, epochs=50, batch_size=6,
                 precision='legacy_amp', density=f'{group}{percent}')
            for seed in range(42, 47)
            for group in ['Dense', 'Recent-dense', 'Persist-dense']
            for percent in [10, 25, 50, 75]]

def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)

def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='exp1 core queue root; writes density/ below it')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--verification', type=Path, help='Defaults to <root>/verification_minimal.json')
    p.add_argument('--plan-only', action='store_true')
    a=p.parse_args(); specs=specifications(); root=a.root/'density'
    verification=a.verification or a.root/'verification_minimal.json'
    plan=dict(specifications=specs, full_cohort='Reuse exp1 T28 H14 Identity fits')
    file=root/'plan.json'
    if file.exists():
        if json.loads(file.read_text()) != plan:
            raise ValueError('Existing density plan differs; use a new queue directory')
    else: write(file, plan)
    if a.plan_only:
        print(json.dumps(dict(fits=len(specs), plan=str(file))));return
    # Require only the five full-cohort baselines used in the density comparison.
    for seed in range(42, 47):
        directory=a.root/'runs'/f'original_masked_T28_H14_Identity_history_seed{seed}'
        record=json.loads((directory/'complete.json').read_text())
        assert record['status']=='complete'
        assert record['config']['seed']==seed and not record['config'].get('density')
        for filename, field in [('best_weights.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256')]:
            assert sha256((directory/filename).read_bytes()).hexdigest()==record[field]
    lock=root/'worker.lock'; completed=[]
    with lock.open('x') as handle:handle.write(str(os.getpid()))
    try:
        for spec in specs:
            name=f"{spec['density']}_seed{spec['seed']}"; out=root/'runs'/name
            if not (out/'complete.json').exists():
                if out.exists():raise RuntimeError(f'Preserving unfinished run: {out}')
                command=[sys.executable,'-m','exp.exp1.train','--bundle',str(a.bundle.resolve()),
                         '--verification',str(verification.resolve()),'--out',str(out.resolve())]
                for key,value in spec.items():
                    if key=='static':
                        if value:command.append('--static')
                    else:command.extend(['--'+key.replace('_','-'),str(value)])
                write(root/'status.json',dict(status='running',current=name,completed=len(completed),total=len(specs)))
                logs=root/'logs';logs.mkdir(exist_ok=True)
                with (logs/(name+'.log')).open('x') as log:
                    subprocess.run(command,cwd=REPO_ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            record=json.loads((out/'complete.json').read_text())
            assert record['status']=='complete'
            for key,value in spec.items():assert record['config'][key]==value
            for filename,field in [('best_weights.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256')]:
                assert sha256((out/filename).read_bytes()).hexdigest()==record[field]
            completed.append(name)
        write(root/'status.json',dict(status='complete',completed=len(completed),total=len(specs)))
    except Exception as error:
        write(root/'status.json',dict(status='failed',completed=len(completed),error=str(error)))
        raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
