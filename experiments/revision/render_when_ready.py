"""Complete CPU summaries and paper figures after the local analyses finish.

This writes tables and figures only. It does not revise scientific conclusions,
publish files, or claim that the resulting manuscript has been reviewed.
"""
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime,timezone
import json
import subprocess
import sys
import time
from summarize_queue import write_json


def run(args):
    root=args.root.resolve();script=Path(__file__).resolve().parent
    v1=root/'private_local_runs/20261114_v1/queue';v2=root/'private_local_runs/20261114_v2/queue'
    result=root/'neural_results';folder=root/'private_local_runs/render_pipeline';folder.mkdir(exist_ok=True)
    start=time.monotonic();done=[]
    def execute(name,python,command):
        write_json(folder/'status.json',dict(status='running',step=name,completed=done,updated_utc=datetime.now(timezone.utc).isoformat()))
        with (folder/(name+'.log')).open('w',encoding='utf8') as log:
            outcome=subprocess.run([str(python),'-u',str(script/command[0])]+command[1:],stdout=log,stderr=subprocess.STDOUT,
                timeout=max(60,args.max_hours*3600-(time.monotonic()-start)))
        if outcome.returncode:raise RuntimeError(f'{name} failed; see {folder/(name+".log")}')
        done.append(name);print(json.dumps(dict(completed=name)),flush=True)
    while True:
        path=root/'private_local_runs/dependent_analysis/status.json'
        if path.exists():
            status=json.loads(path.read_text())
            if status['status']=='stopped':raise RuntimeError('Dependent analysis stopped: '+status.get('error',''))
            if status['status']=='complete':break
        if time.monotonic()-start>args.max_hours*3600:raise TimeoutError('Rendering wait budget exhausted')
        write_json(folder/'status.json',dict(status='waiting_for_complete_analyses',updated_utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(20)
    common=['--queues',str(v1),str(v2),'--bundle',str(args.bundle.resolve()),'--baselines',str(root/'complete_horizon/private_predictions')]
    execute('density',sys.executable,['summarize_density.py']+common+['--out',str(result/'density')])
    for period in ('december','january'):
        command=['spatial_priority_analysis.py']+common+['--out',str(result/'activity'/period)]
        if period=='january':command+=['--january',str(result/'january')]
        execute(period+'_coverage',sys.executable,command)
    # Rerun the aggregators once after every selected prediction is available.
    # Their original workers are read-only after no new files remain.
    execute('December_summary',sys.executable,['summarize_queue.py','--queues',str(v1),'--bundle',str(args.bundle.resolve()),
          '--baselines',str(root/'complete_horizon/private_predictions'),'--out',str(result)])
    execute('extension_summary',sys.executable,['summarize_queue.py','--queues',str(v2),'--bundle',str(args.bundle.resolve()),
          '--out',str(result/'extension')])
    for part in ('neighbors','forecasting','controls','ablations','density','diagnostics','activity','attribution'):
        execute('render_'+part,args.plot_python,['render_revision_results.py','--part',part])
    repo=script.parents[1]
    execute('attribution_map',args.plot_python,['render_attribution_map.py','--summary',str(result/'attribution/summary'),
          '--bundle',str(args.bundle.resolve()),'--geometry',str(repo/'artifacts/cartography/tiger2020_seattle.npz'),
          '--out',str(repo/'paper/revision/art/attribution_cluster_map.pdf'),'--qa',str(root/'qa/attribution_cluster_map.png')])
    execute('source_audit',sys.executable,['audit_release.py','--root',str(root)])
    write_json(folder/'status.json',dict(status='tables_and_figures_complete',completed=done,
          manuscript='Scientific interpretation, source consistency, PDF compilation and visual inspection remain required before publishing',
          finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--plot-python',type=Path,required=True)
    p.add_argument('--max-hours',type=float,default=18);args=p.parse_args()
    try:run(args)
    except BaseException as error:
        write_json(args.root/'private_local_runs/render_pipeline/status.json',dict(status='stopped',error=str(error)))
        raise
