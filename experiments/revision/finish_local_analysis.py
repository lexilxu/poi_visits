"""Run dependent analyses only when every required fitted seed is available.

This helper never trains, selects a seed, publishes a manuscript, or writes Git.
It waits for the independently running local training/evaluation workers and
records failures rather than treating a launched process as a completed result.
"""
from argparse import ArgumentParser
from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess
import sys
import time
from summarize_queue import write_json


def selected_ready(queue,recipe):
    path=queue/recipe/'selected_runs.json'
    if not path.exists():return False
    item=json.loads(path.read_text())
    return sorted(r['seed'] for r in item['runs'])==[42,43,44,45,46]


def main(args):
    root=args.root.resolve();v1=root/'private_local_runs/20261114_v1/queue';v2=root/'private_local_runs/20261114_v2/queue'
    results=root/'neural_results';logdir=root/'private_local_runs/dependent_analysis';logdir.mkdir(exist_ok=True)
    script=Path(__file__).resolve().parent;start=time.monotonic();done=[]
    def execute(name,command):
        with (logdir/(name+'.log')).open('w',encoding='utf8') as log:
            result=subprocess.run([sys.executable,'-u',str(script/command[0])]+command[1:],stdout=log,stderr=subprocess.STDOUT,
                                  timeout=max(60,args.max_hours*3600-(time.monotonic()-start)))
        if result.returncode:raise RuntimeError(f'{name} failed; inspect {logdir/(name+".log")}')
        done.append(name);print(json.dumps(dict(completed=name)),flush=True)
    def all_attribution(folder):return all((folder/f'seed{s}/attribution_protocol.json').exists() for s in range(42,47))
    while True:
        if time.monotonic()-start>args.max_hours*3600:raise TimeoutError('Dependent analysis time budget exhausted')
        if all_attribution(results/'attribution') and not all_attribution(results/'attribution_january'):
            execute('January_attribution',['attribution_audit.py','--queue',str(v1),'--bundle',str(args.bundle.resolve()),
                    '--out',str(results/'attribution_january'),'--device','cuda:0','--extended-panel',str(root/'private_validation_202101/extended_panel.npz')])
        full='GNN-static_T14_Identity_full_MAE';removed='GNN-static_T14_Identity_without_parent_MAE'
        for period,folder in [('december',results/'attribution'),('january',results/'attribution_january')]:
            destination=folder/'summary'
            if (destination/'summary_protocol.json').exists() or not all_attribution(folder):continue
            if not selected_ready(v1,removed):continue
            if period=='january' and not all((results/'january/run_records'/f'{r}_seed{s}.json').exists() for r in (full,removed) for s in range(42,47)):continue
            command=['summarize_attribution.py','--queue',str(v1),'--bundle',str(args.bundle.resolve()),'--attribution',str(folder),'--out',str(destination)]
            if period=='january':command+=['--january',str(results/'january')]
            execute(period+'_attribution_summary',command)
        states=[json.loads((q/'queue_status.json').read_text()) if (q/'queue_status.json').exists() else {} for q in (v1,v2)]
        if any(s.get('status')=='stopped' for s in states):raise RuntimeError('A training queue stopped before completion')
        if all(s.get('status')=='complete' for s in states):
            from summarize_queue import selected_jobs
            january_ready=all((results/'january/run_records'/(meta['id'].replace(' ','_')+f"_seed{meta['seed']}.json")).exists()
                              for _,meta,_ in selected_jobs([v1,v2]) if meta['stage']!='density')
            summaries_ready=all((results/folder/'summary/summary_protocol.json').exists() for folder in ('attribution','attribution_january'))
            if january_ready and summaries_ready:break
        write_json(logdir/'status.json',dict(status='waiting',completed=done,updated_utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(20)
    # Separate destination avoids racing the earlier streaming comparison worker.
    for period in ('december','january'):
        command=['paired_seed_analysis.py','--queues',str(v1),str(v2),'--bundle',str(args.bundle.resolve()),
                 '--baselines',str(root/'complete_horizon/private_predictions'),'--out',str(results/'final_uncertainty'/period)]
        if period=='january':command+=['--january',str(results/'january')]
        execute(period+'_paired_intervals',command)
    write_json(logdir/'status.json',dict(status='complete',completed=done,finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--max-hours',type=float,default=18)
    args=p.parse_args()
    try:main(args)
    except BaseException as error:
        path=args.root/'private_local_runs/dependent_analysis/status.json'
        write_json(path,dict(status='stopped',error=str(error),updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
